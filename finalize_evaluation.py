#!/usr/bin/env python3
"""Validate and rescore a completed endpoint evaluation without new inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from url_extraction_agent.batch import load_jsonl, result_record, summarize, write_jsonl_atomic
from url_extraction_agent.schema import AgentAttempt


def unique_by_arxiv(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        arxiv_id = row.get("arxiv_id")
        if not isinstance(arxiv_id, str) or not arxiv_id:
            raise ValueError(f"{label} contains a row without an arxiv_id")
        if arxiv_id in indexed:
            raise ValueError(f"{label} contains duplicate arxiv_id {arxiv_id}")
        indexed[arxiv_id] = row
    return indexed


def finalize(
    inputs: list[dict[str, Any]], raw_results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_id = unique_by_arxiv(inputs, "input")
    result_by_id = unique_by_arxiv(raw_results, "results")
    missing = sorted(set(source_by_id) - set(result_by_id))
    extra = sorted(set(result_by_id) - set(source_by_id))
    if missing or extra:
        raise ValueError(
            f"input/result ID mismatch: {len(missing)} missing and {len(extra)} extra; "
            f"missing sample={missing[:3]}, extra sample={extra[:3]}"
        )

    rescored: list[dict[str, Any]] = []
    for source in inputs:
        raw = result_by_id[source["arxiv_id"]]
        record = result_record(source, AgentAttempt.model_validate(raw["attempt"]))
        if raw.get("evaluated_at"):
            record["evaluated_at"] = raw["evaluated_at"]
        rescored.append(record)
    summary = summarize(rescored)
    summary["input_records"] = len(inputs)
    summary["unique_arxiv_ids"] = len(source_by_id)
    return rescored, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rescored, summary = finalize(load_jsonl(args.input), load_jsonl(args.results))
    write_jsonl_atomic(args.output_results, rescored)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
