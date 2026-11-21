"""Evaluation summaries and rescoring of published episode records with the shared scorer.

    python -m harness.report summarize runs/test.jsonl [--output test-summary.json]
    python -m harness.report rescore --repo nielsr/qwen35-4b-url-agent-eval-v2 --dataset nielsr/paper-url-extraction-v1

`rescore` recomputes every record's verification and protocol completeness from its stored answer and raw
OpenCode messages with `harness.scoring.score_episode`, and reports each difference against the stored
record and the published summary, with the reason.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}}
    # base/adapter always (paired comparison below); other labels (e.g. "teacher") when present.
    labels = ["base", "adapter"] + sorted({str(r["evaluation_model"]) for r in records if r.get("evaluation_model")} - {"base", "adapter"})
    for label in labels:
        selected = [record for record in records if record.get("evaluation_model") == label]
        field_correct = Counter()
        field_total = Counter()
        failures = Counter()
        tool_calls = 0
        tool_success = 0
        recovered_failures = 0
        total_failures = 0
        for record in selected:
            for field, value in (record.get("verification") or {}).get("fields", {}).items():
                if value is not None:
                    field_total[field] += 1
                    field_correct[field] += int(bool(value))
            events = record.get("tool_events") or []
            tool_calls += len(events)
            tool_success += sum(bool(event.get("success")) for event in events)
            for index, event in enumerate(events):
                if not event.get("success"):
                    total_failures += 1
                    if any(later.get("success") for later in events[index + 1 :]):
                        recovered_failures += 1
            if record.get("failure_reason"):
                failures[str(record["failure_reason"])] += 1
        count = len(selected)
        scoreable = sum(field_total.values())
        summary["models"][label] = {
            "episodes": count,
            "complete_episodes": sum(bool(record.get("attempt_complete")) for record in selected),
            "exact_match_episodes": sum(
                bool((record.get("verification") or {}).get("all_scoreable_correct")) for record in selected
            ),
            "field_accuracy": (sum(field_correct.values()) / scoreable) if scoreable else None,
            "fields": {
                field: {"correct": field_correct[field], "total": field_total[field]} for field in sorted(field_total)
            },
            "average_tool_calls": (tool_calls / count) if count else None,
            "tool_success_rate": (tool_success / tool_calls) if tool_calls else None,
            "failed_tool_recovery_rate": (recovered_failures / total_failures) if total_failures else None,
            "failure_reasons": dict(failures.most_common()),
        }
    tiers = sorted({record.get("difficulty_tier") for record in records if record.get("difficulty_tier") is not None})
    for label in labels:
        by_tier = {}
        for tier in tiers:
            selected = [r for r in records if r.get("evaluation_model") == label and r.get("difficulty_tier") == tier]
            exact = sum(bool((r.get("verification") or {}).get("all_scoreable_correct")) for r in selected)
            by_tier[str(tier)] = {"episodes": len(selected), "exact_match_episodes": exact}
        summary["models"][label]["by_difficulty_tier"] = by_tier
    base = summary["models"]["base"]
    adapter = summary["models"]["adapter"]
    exact_by_paper = {
        label: {
            int(r["paper_id"]): bool((r.get("verification") or {}).get("all_scoreable_correct"))
            for r in records
            if r.get("evaluation_model") == label and r.get("paper_id") is not None
        }
        for label in ("base", "adapter")
    }
    shared = sorted(set(exact_by_paper["base"]) & set(exact_by_paper["adapter"]))
    gained = sum(exact_by_paper["adapter"][p] and not exact_by_paper["base"][p] for p in shared)
    lost = sum(exact_by_paper["base"][p] and not exact_by_paper["adapter"][p] for p in shared)
    summary["paired"] = {
        "papers": len(shared),
        "adapter_only_correct": gained,
        "base_only_correct": lost,
        "mcnemar_exact_p": mcnemar_exact(gained, lost),
    }
    summary["comparison"] = {
        "field_accuracy_delta": (
            adapter["field_accuracy"] - base["field_accuracy"]
            if adapter["field_accuracy"] is not None and base["field_accuracy"] is not None
            else None
        ),
        "exact_match_delta": (
            adapter["exact_match_episodes"] / adapter["episodes"] - base["exact_match_episodes"] / base["episodes"]
            if adapter["episodes"] and base["episodes"]
            else None
        ),
    }
    return summary




# --------------------------------------------------------------------------------------------------
# Rescoring
# --------------------------------------------------------------------------------------------------


def rescore_record(record: dict[str, Any], row: dict[str, Any], *, max_tool_calls: int | None) -> dict[str, Any]:
    """The record as the shared scorer sees it now, plus `rescore_notes` explaining any difference."""
    from pydantic import ValidationError

    from harness.opencode import findings_from_messages, tool_events
    from harness.scoring import canonical_url, score_episode
    from harness.task import ToolEvent, URLFindings

    notes: list[str] = []
    findings = None
    if record.get("findings"):
        try:
            findings = URLFindings.model_validate(record["findings"])
        except ValidationError:
            notes.append("stored answer fails the strict answer schema")
    if findings is None and record.get("opencode_messages"):
        findings = findings_from_messages(record["opencode_messages"])
        if findings is not None:
            notes.append("answer recovered from the final message by the shared parser")
    if record.get("opencode_messages"):
        events = tool_events(record["opencode_messages"])
    else:
        events = [ToolEvent.model_validate(event) for event in record.get("tool_events") or []]
    score = score_episode(row, findings, events, max_tool_calls=max_tool_calls)
    old_fields = (record.get("verification") or {}).get("fields") or {}
    for name, new in score["verification"]["fields"].items():
        old = old_fields.get(name)
        if old == new:
            continue
        if old is None and new is False and findings is None:
            notes.append(f"{name}: no answer now counts as wrong (was unscored)")
        elif old is None:
            notes.append(f"{name}: now scored (label quality {row.get(f'{name}_label_quality')})")
        elif new is None:
            notes.append(f"{name}: no longer scored (label quality {row.get(f'{name}_label_quality')})")
        elif new and findings is not None:
            predicted = canonical_url(getattr(findings, f"{name}_url"))
            aliases = {canonical_url(alias) for alias in row.get(f"reference_{name}_aliases") or []}
            if predicted in aliases and predicted != canonical_url(row.get(f"reference_{name}_url")):
                notes.append(f"{name}: matches a reference alias")
            else:
                notes.append(f"{name}: {old} -> {new}")
        else:
            notes.append(f"{name}: {old} -> {new}")
    if bool(record.get("attempt_complete")) != score["attempt_complete"]:
        notes.append(f"complete {record.get('attempt_complete')} -> {score['attempt_complete']}: {score['incomplete_reason']}")
    return {
        **record,
        "verification": score["verification"],
        "attempt_complete": score["attempt_complete"],
        "findings": findings.model_dump(mode="json") if findings else None,
        "rescore_notes": notes,
    }


def _headline(model: dict[str, Any]) -> str:
    accuracy = model.get("field_accuracy")
    fields = sum(field["total"] for field in (model.get("fields") or {}).values())
    return (
        f"field acc {100 * accuracy:.1f}% ({fields} fields), exact {model.get('exact_match_episodes')}/{model.get('episodes')}, "
        f"complete {model.get('complete_episodes')}"
        if accuracy is not None
        else "no scored fields"
    )


def rescore_repo(
    repo: str, dataset: str, *, files: list[str] | None, max_tool_calls: int | None, revision: str | None = None
) -> dict[str, Any]:
    from datasets import get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    rows: dict[int, dict[str, Any]] = {}
    for split in get_dataset_split_names(dataset):
        for row in load_dataset(dataset, split=split):
            rows[int(row["paper_id"])] = dict(row)
    names = files or sorted(
        name
        for name in api.list_repo_files(repo, repo_type="dataset", revision=revision)
        if name.endswith(".jsonl") and not name.startswith("sessions/")
    )
    report: dict[str, Any] = {"repo": repo, "dataset": dataset, "files": {}}
    for name in names:
        local = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))
        # Teacher records (traces, baselines) carry no evaluation_model; summarize them as "teacher".
        records = [{**r, "evaluation_model": r.get("evaluation_model") or "teacher"} for r in load_records(local)]
        if not records or "verification" not in records[0]:
            continue
        rescored = [rescore_record(record, rows[int(record["paper_id"])], max_tool_calls=max_tool_calls) for record in records]
        published = None
        summary_name = name.removesuffix(".jsonl") + "-summary.json"
        if summary_name in api.list_repo_files(repo, repo_type="dataset", revision=revision):
            published = json.loads(Path(hf_hub_download(repo, summary_name, repo_type="dataset", revision=revision)).read_text())
        before, after = summarize(records), summarize(rescored)
        entry: dict[str, Any] = {"models": {}, "differences": []}
        for label, model in after["models"].items():
            if not model["episodes"]:
                continue
            entry["models"][label] = {
                "published": (published or {}).get("models", {}).get(label),
                "stored_records": before["models"][label],
                "rescored": model,
            }
        for record in rescored:
            if record["rescore_notes"]:
                entry["differences"].append(
                    {"label": record.get("evaluation_model"), "arxiv_id": record["arxiv_id"], "notes": record["rescore_notes"]}
                )
        report["files"][name] = entry
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"# {report['repo']} (labels: {report['dataset']})")
    for name, entry in report["files"].items():
        print(f"\n## {name}")
        for label, models in entry["models"].items():
            published = models["published"]
            print(f"- {label}: published {_headline(published) if published else 'n/a'}")
            print(f"  {' ' * len(label)}  stored    {_headline(models['stored_records'])}")
            print(f"  {' ' * len(label)}  rescored  {_headline(models['rescored'])}")
        for difference in entry["differences"]:
            print(f"  * [{difference['label']}] {difference['arxiv_id']}: {'; '.join(difference['notes'])}")
        if not entry["differences"]:
            print("  no per-episode differences")


def results_table(repo: str, dataset: str, *, labels: dict[str, str] | None = None, revision: str | None = None) -> str:
    """Markdown table per (model, split): all evaluated papers and the fully scoreable subset.

    `labels` maps "<dir>/<label>" (e.g. "sft-grpo/adapter") to a display name; others use that key.
    """
    from datasets import get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    from harness.scoring import fully_scoreable

    rows = {}
    for split in get_dataset_split_names(dataset):
        for row in load_dataset(dataset, split=split):
            rows[int(row["paper_id"])] = dict(row)
    api = HfApi()
    files = sorted(
        name
        for name in api.list_repo_files(repo, repo_type="dataset", revision=revision)
        if name.endswith(".jsonl") and "/" in name and not name.startswith(("sessions/", "difficulty/"))
    )
    lines = [
        "| Model | Split | Field acc. (all) | Exact (all) | Field acc. (fully scoreable) | Exact (fully scoreable) | Tool calls |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in files:
        directory, split = name.rsplit("/", 1)
        split = split.removesuffix(".jsonl")
        records = [
            {**r, "evaluation_model": r.get("evaluation_model") or "teacher"}
            for r in load_records(Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision)))
        ]
        for label in sorted({r["evaluation_model"] for r in records}):
            mine = [r for r in records if r["evaluation_model"] == label]
            full = [r for r in mine if fully_scoreable(rows[int(r["paper_id"])])]
            every, subset = summarize(mine)["models"][label], summarize(full)["models"][label]

            def pct(model: dict[str, Any]) -> str:
                fields = sum(f["total"] for f in model["fields"].values())
                return f"{100 * model['field_accuracy']:.1f}% ({fields})" if model["field_accuracy"] is not None else "-"

            key = f"{directory}/{label}"
            lines.append(
                f"| {(labels or {}).get(key, key)} | {split} | {pct(every)} | {every['exact_match_episodes']}/{every['episodes']} "
                f"| {pct(subset)} | {subset['exact_match_episodes']}/{subset['episodes']} "
                f"| {every['average_tool_calls']:.1f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    summarize_parser = commands.add_parser("summarize", help="Summary JSON of episode record files")
    summarize_parser.add_argument("files", nargs="+", type=Path)
    summarize_parser.add_argument("--output", type=Path)
    rescore_parser = commands.add_parser("rescore", help="Rescore a published evaluation repo with the shared scorer")
    rescore_parser.add_argument("--repo", required=True)
    rescore_parser.add_argument("--dataset", default="nielsr/paper-url-extraction-v1")
    rescore_parser.add_argument("--files", nargs="*")
    rescore_parser.add_argument("--revision")
    rescore_parser.add_argument("--max-tool-calls", type=int, default=15, help="The limit the episodes ran with")
    rescore_parser.add_argument("--json", type=Path, help="Also write the full report here")
    table_parser = commands.add_parser("table", help="Markdown results table: all papers and fully scoreable subset")
    table_parser.add_argument("--repo", required=True)
    table_parser.add_argument("--dataset", required=True)
    table_parser.add_argument("--label", action="append", default=[], help="dir/label=Display name, repeatable")
    args = parser.parse_args()
    if args.command == "table":
        print(results_table(args.repo, args.dataset, labels=dict(item.split("=", 1) for item in args.label)))
        return
    if args.command == "summarize":
        records = [record for path in args.files for record in load_records(path)]
        text = json.dumps(summarize(records), indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(text)
        print(text, end="")
        return
    report = rescore_repo(args.repo, args.dataset, files=args.files, max_tool_calls=args.max_tool_calls, revision=args.revision)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print_report(report)


if __name__ == "__main__":
    main()
