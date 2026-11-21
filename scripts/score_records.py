#!/usr/bin/env python3
"""Score trace/evaluation JSONL files with evaluate_vllm.summarize: field accuracy, exact matches, tool calls.

    uv run --no-project --with ... python scripts/score_records.py FILE.jsonl [FILE.jsonl ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_vllm import SCOREABLE_QUALITIES, summarize  # noqa: E402


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


for path in sys.argv[1:]:
    records = [json.loads(line) for line in open(path)]
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
