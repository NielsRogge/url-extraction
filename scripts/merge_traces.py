#!/usr/bin/env python3
"""Merge distillation worker outputs: accepted records only, one per paper_id, without raw OpenCode messages.

    python scripts/merge_traces.py OUT.jsonl IN.jsonl [IN.jsonl ...]
"""
import json
import sys

out, inputs = sys.argv[1], sys.argv[2:]
seen: set[int] = set()
total = kept = 0
with open(out, "w") as handle:
    for path in inputs:
        for line in open(path):
            total += 1
            record = json.loads(line)
            if not record.get("accepted") or int(record["paper_id"]) in seen:
                continue
            seen.add(int(record["paper_id"]))
            record.pop("opencode_messages", None)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
print(f"{kept} accepted unique trajectories from {total} records -> {out}")
