"""Select arXiv IDs from a dataset split, e.g. the candidate pool for the difficulty probe.

    python -m harness.select --dataset nielsr/paper-url-extraction-v2 --split train --exclude-tiers 7 --output ids.txt

Keeps rows with at least one verified requested label (what the probe and GRPO can score).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harness.scoring import partly_scoreable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--exclude-tiers", default="", help="Comma-separated difficulty tiers to drop, e.g. 7")
    parser.add_argument("--include-ids-file", type=Path, help="Restrict to these arXiv IDs first")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    drop = {int(tier) for tier in args.exclude_tiers.split(",") if tier.strip()}
    include = set(args.include_ids_file.read_text().split()) if args.include_ids_file else None
    ids = sorted(
        str(row["arxiv_id"])
        for row in map(dict, load_dataset(args.dataset, split=args.split))
        if partly_scoreable(row)
        and (row.get("difficulty_tier") is None or int(row["difficulty_tier"]) not in drop)
        and (include is None or str(row["arxiv_id"]) in include)
    )
    args.output.write_text("\n".join(ids) + "\n")
    print(f"{len(ids)} ids from {args.dataset}:{args.split} -> {args.output}")


if __name__ == "__main__":
    main()
