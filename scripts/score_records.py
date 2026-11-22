#!/usr/bin/env python3
"""Score trace/evaluation JSONL files with evaluate_vllm.summarize: field accuracy, exact matches, tool calls.

    python scripts/score_records.py [--fully-scoreable] [--rescore DATASET] FILE.jsonl [FILE.jsonl ...]

`--fully-scoreable` keeps only papers whose every requested field has a scoreable label (the default evaluation
set; records from `--include-partial` runs also contain papers with an unverifiable field).

`--rescore` recomputes each record's verification from its stored findings against DATASET's rows (e.g. the
validated dataset with `reference_*_aliases`), using the current alias-aware `score_findings`.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_vllm import SCOREABLE_QUALITIES, URLFindings, score_findings, summarize  # noqa: E402

ROWS: dict[int, dict] = {}


def rescored(record: dict) -> dict:
    row = ROWS[int(record["paper_id"])]
    findings = URLFindings.model_validate(record["findings"]) if record.get("findings") else None
    if findings is not None:
        verification = score_findings(row, findings)
    else:
        verification = {"fields": {}, "num_scoreable": 0, "all_scoreable_correct": False}
    qualities = {name: row.get(f"{name}_label_quality") if row.get(f"missing_{name}") else None for name in ("github", "project_page")}
    return dict(record, verification=verification, label_quality=qualities)


def normalized(record: dict) -> dict:
    """Count every scoreable label; a field without a verified answer (e.g. no valid output) counts as wrong."""
    verification = dict(record.get("verification") or {})
    fields = dict(verification.get("fields") or {})
    for field, quality in (record.get("label_quality") or {}).items():
        if quality in SCOREABLE_QUALITIES and fields.get(field) is None:
            fields[field] = False
    verification["fields"] = fields
    verification["all_scoreable_correct"] = bool(fields) and all(v for v in fields.values() if v is not None)
    return dict(record, verification=verification, evaluation_model="base")


paths = sys.argv[1:]
FULL_ONLY = "--fully-scoreable" in paths
paths = [p for p in paths if p != "--fully-scoreable"]


def fully_scoreable(record: dict) -> bool:
    qualities = [q for q in (record.get("label_quality") or {}).values() if q is not None]
    return bool(qualities) and all(q in SCOREABLE_QUALITIES for q in qualities)


if paths and paths[0] == "--rescore":
    from datasets import load_dataset

    for split in ("train", "validation", "test"):
        for row in load_dataset(paths[1], split=split):
            ROWS[int(row["paper_id"])] = dict(row)
    paths = paths[2:]
for path in paths:
    records = [json.loads(line) for line in open(path)]
    if ROWS:
        records = [rescored(r) for r in records]
    if FULL_ONLY:
        records = [r for r in records if fully_scoreable(r)]
    labels = {record.get("evaluation_model") for record in records}
    for label in sorted(label for label in labels if label) or [None]:
        selected = [normalized(r) for r in records if label is None or r.get("evaluation_model") == label]
        m = summarize(selected)["models"]["base"]
        f = m["fields"]
        fields = sum(v["total"] for v in f.values())
        print(
            f"{path} [{label or 'all'}]: field acc {100 * m['field_accuracy']:.1f}% ({fields} fields), "
            f"exact {m['exact_match_episodes']}/{m['episodes']}, complete {m['complete_episodes']}, "
            f"tool calls {m['average_tool_calls']:.1f}"
        )
