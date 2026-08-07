#!/usr/bin/env python3
"""Run and summarize held-out URL-agent evaluation in the OpenCode harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from opencode_ai import AsyncOpencode

from generate_opencode_traces import (
    SCORED_QUALITIES,
    cached_hf_cli_token,
    configure_huggingface_billing,
    opencode_server,
    run_one,
    split_model,
    validate_huggingface_token,
)

DEFAULT_DATASET = "nielsr/paper-url-extraction-v1"
DEFAULT_TRAINING_TRACES = "nielsr/paper-url-opencode-traces"


def load_training_ids(dataset_id: str, split: str) -> set[int]:
    """Return every paper ID present in the trace corpus, accepted or not."""
    return {int(row["paper_id"]) for row in load_dataset(dataset_id, split=split)}


def load_completed_ids(path: Path) -> set[int]:
    """Avoid paying for duplicate held-out episodes when an evaluation resumes."""
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open() as handle:
        for line in handle:
            try:
                completed.add(int(json.loads(line)["paper_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def row_has_scoreable_target(row: dict[str, Any]) -> bool:
    return any(
        bool(row[f"missing_{name}"]) and row.get(f"{name}_label_quality") in SCORED_QUALITIES
        for name in ("github", "project_page")
    )


def select_rows(
    rows: list[dict[str, Any]], training_ids: set[int], completed_ids: set[int], start: int, limit: int
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row_has_scoreable_target(row)
        and int(row["paper_id"]) not in training_ids
        and int(row["paper_id"]) not in completed_ids
    ]
    return eligible[start : start + limit]


def summarize(records: list[dict[str, Any]], *, split: str, model: str, training_ids: set[int]) -> dict[str, Any]:
    field_scores = [
        value
        for record in records
        for value in (record.get("verification") or {}).get("fields", {}).values()
        if value is not None
    ]
    episodes_with_tools = 0
    successful_tool_episodes = 0
    stopping_successes = 0
    recovery_opportunities = 0
    recoveries = 0
    tool_calls = 0
    failed_tool_calls = 0
    for record in records:
        events = record.get("tool_events") or []
        tool_calls += len(events)
        failures = [index for index, event in enumerate(events) if not event.get("success")]
        failed_tool_calls += len(failures)
        episodes_with_tools += bool(events)
        successful_tool_episodes += bool(events) and any(event.get("success") for event in events)
        failure = str(record.get("failure_reason") or "")
        stopped_cleanly = record.get("findings") is not None and not any(
            marker in failure for marker in ("timed out", "tool-call limit", "provider error")
        )
        stopping_successes += stopped_cleanly
        if failures:
            recovery_opportunities += 1
            last_failure = failures[-1]
            recoveries += any(event.get("success") for event in events[last_failure + 1 :])
    count = len(records)
    evaluated_ids = {int(record["paper_id"]) for record in records}
    overlap = sorted(evaluated_ids & training_ids)
    return {
        "schema_version": 1,
        "model": model,
        "split": split,
        "episodes": count,
        "training_paper_ids": len(training_ids),
        "training_overlap_count": len(overlap),
        "training_overlap_ids": overlap,
        "url_accuracy": {
            "correct": sum(field_scores),
            "scoreable_fields": len(field_scores),
            "accuracy": sum(field_scores) / len(field_scores) if field_scores else None,
            "all_scoreable_correct_episodes": sum(
                bool((record.get("verification") or {}).get("all_scoreable_correct")) for record in records
            ),
        },
        "agent_behavior": {
            "episodes_with_tool_use": episodes_with_tools,
            "tool_use_rate": episodes_with_tools / count if count else None,
            "episodes_with_successful_tool": successful_tool_episodes,
            "tool_calls": tool_calls,
            "failed_tool_calls": failed_tool_calls,
            "clean_stops": stopping_successes,
            "clean_stop_rate": stopping_successes / count if count else None,
            "policy_complete_episodes": sum(bool(record.get("attempt_complete")) for record in records),
            "recovery_opportunities": recovery_opportunities,
            "recoveries": recoveries,
            "recovery_rate": recoveries / recovery_opportunities if recovery_opportunities else None,
        },
    }


async def async_main(args: argparse.Namespace) -> None:
    model = split_model(args.model)
    validate_huggingface_token(model, args.hf_bill_to)
    configure_huggingface_billing(model, args.hf_bill_to)
    training_ids = load_training_ids(args.training_traces, args.training_trace_split)
    completed_ids = load_completed_ids(args.output) if args.resume else set()
    rows = [dict(row) for row in load_dataset(args.dataset, split=args.split)]
    selected = select_rows(rows, training_ids, completed_ids, args.start, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with (
        opencode_server(
            args.base_url,
            args.opencode_command,
            custom_structured_output=not args.native_structured_output,
        ) as base_url,
        AsyncOpencode(base_url=base_url, timeout=httpx.Timeout(args.timeout, connect=10), max_retries=1) as client,
    ):
        for number, row in enumerate(selected, 1):
            print(f"[{number}/{len(selected)}] {row['arxiv_id']}", flush=True)
            record = await run_one(
                client,
                row,
                model,
                episode_timeout=args.episode_timeout,
                max_tool_calls=args.max_tool_calls,
                native_structured_output=args.native_structured_output,
            )
            record["evaluation_split"] = args.split
            record["excluded_from_training"] = int(row["paper_id"]) not in training_ids
            with args.output.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"  complete={record['attempt_complete']} score={record['verification']['fields']} "
                f"tools={len(record['tool_events'])}",
                flush=True,
            )
    with args.output.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    summary = summarize(records, split=args.split, model=args.model, training_ids=training_ids)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--training-traces", default=DEFAULT_TRAINING_TRACES)
    parser.add_argument("--training-trace-split", default="train")
    parser.add_argument("--model", required=True, help="OpenCode PROVIDER_ID/MODEL_ID")
    parser.add_argument("--hf-bill-to", default="huggingface")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument(
        "--native-structured-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose OpenCode's terminal StructuredOutput tool (disable only for incompatible OpenCode builds)",
    )
    parser.add_argument("--episode-timeout", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/heldout.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/eval/summary.json"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-url")
    parser.add_argument("--opencode-command", default="opencode")
    parser.add_argument("--use-hf-cli-token", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.use_hf_cli_token and not os.environ.get("HF_TOKEN"):
        token = cached_hf_cli_token()
        if not token:
            raise RuntimeError("No active token found in the Hugging Face CLI cache")
        os.environ["HF_TOKEN"] = token
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
