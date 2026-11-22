#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets>=4.0.0", "huggingface-hub>=1.13.0"]
# ///
"""Build the difficulty-balanced paper URL dataset v2.

Inputs:
- `nielsr/paper-url-extraction-v1-validated` (labels re-checked against HF metadata, difficulty tiers).
- Mined rows from `mine_papers.py` after `recheck_nulls.py` (verified-null GitHub labels contradicted by a
  repository that names the paper are no longer scored).

Steps:
1. Drop duplicates and papers already in v1, and silver project pages on generic hosts (product pages,
   model hubs, demo studios) that are not paper-specific project pages.
2. Drop rows with no scoreable requested field left.
3. Hold out a hard-heavy evaluation split, `test_hard`, from the mined papers, stratified by difficulty
   tier, so that every teacher/student pair is compared on papers that need more than one API call.
4. Splits: `train` (v1 train + remaining mined papers), `validation` and `test` (v1, unchanged, for
   comparison with earlier results), `test_hard` (new).

    uv run build_dataset_v2.py --mined runs/part-1.rechecked.jsonl runs/part-2.rechecked.jsonl ... \\
        --output-repo nielsr/paper-url-extraction-v2
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import Dataset, DatasetDict, load_dataset

SCOREABLE = {"gold", "silver", "verified_null"}
FIELDS = ("github", "project_page")
# Hosts whose pages are products, model hubs or demos rather than a paper's own project page.
GENERIC_PROJECT_HOSTS = {
    "deepmind.google", "modelscope.cn", "huggingface.co", "openai.com", "research.google", "blog.google",
    "x.com", "twitter.com", "youtube.com", "colab.research.google.com", "arxiv.org", "paperswithcode.com",
    "openreview.net", "microsoft.com", "aws.amazon.com", "nvidia.com", "docs.google.com", "drive.google.com",
}  # fmt: skip
# Held-out mined papers per difficulty tier for `test_hard` (tier 1 kept small: it is one API call).
TEST_HARD_QUOTA = {0: 30, 1: 20, 2: 25, 3: 40, 4: 10, 5: 30, 6: 20}
TIER_NAMES = {0: "verified no URL", 1: "HF paper API", 2: "abstract", 3: "PDF page 1", 4: "PDF page 2",
              5: "GitHub only via project page", 6: "later in the paper", 7: "label not found in any source (weak label)"}  # fmt: skip


def scoreable(row: dict[str, Any]) -> bool:
    return any(row[f"missing_{n}"] and row.get(f"{n}_label_quality") in SCOREABLE for n in FIELDS)


def generic_project_page(url: str | None) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return any(host == h or host.endswith("." + h) for h in GENERIC_PROJECT_HOSTS)


def load_mined(paths: list[Path], exclude: set[str], columns: list[str]) -> tuple[list[dict[str, Any]], collections.Counter]:
    rows: dict[str, dict[str, Any]] = {}
    stats: collections.Counter = collections.Counter()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats["mined"] += 1
            arxiv = str(row["arxiv_id"])
            if arxiv in exclude or arxiv in rows:
                stats["duplicate"] += 1
                continue
            if row.get("github_validation") == "null_contradicted_by_github_search":
                stats["github_null_contradicted"] += 1
            if row.get("project_page_label_quality") == "silver" and generic_project_page(row.get("reference_project_page_url")):
                row["project_page_label_quality"] = "unknown"
                row["project_page_validation"] = "generic_host"
                stats["generic_project_page"] += 1
            if not scoreable(row):
                stats["no_scoreable_field"] += 1
                continue
            notes = {k: row[k] for k in ("adjudication", "github_null_recheck") if k in row}
            out = {c: row.get(c) for c in columns}
            out["split"] = "mined"
            out["mining_notes"] = json.dumps(notes, ensure_ascii=False) if notes else ""
            for name in FIELDS:
                out[f"reference_{name}_aliases"] = list(row.get(f"reference_{name}_aliases") or [])
                out[f"evidence_{name}"] = str(row.get(f"evidence_{name}") or "")
            rows[arxiv] = out
    return list(rows.values()), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="nielsr/paper-url-extraction-v1-validated")
    parser.add_argument("--mined", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset-v2"))
    parser.add_argument("--output-repo")
    args = parser.parse_args()

    base = load_dataset(args.base)
    features = base["train"].features.copy()
    columns = list(features)
    exclude = {str(r["arxiv_id"]) for split in base.values() for r in split}
    mined, stats = load_mined(args.mined, exclude, columns)

    rng = random.Random(args.seed)
    rng.shuffle(mined)
    held: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    taken: collections.Counter = collections.Counter()
    for row in mined:
        tier = row["difficulty_tier"]
        if taken[tier] < TEST_HARD_QUOTA.get(tier, 0):
            taken[tier] += 1
            held.append({**row, "split": "test_hard"})
        else:
            rest.append({**row, "split": "train"})

    def with_notes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**r, "mining_notes": r.get("mining_notes", "")} for r in rows]

    from datasets import Value

    features["mining_notes"] = Value("string")
    for name in ("known_github_url", "known_project_page_url"):
        features[name] = Value("string")
    splits = {
        "train": with_notes([dict(r) for r in base["train"]]) + rest,
        "validation": with_notes([dict(r) for r in base["validation"]]),
        "test": with_notes([dict(r) for r in base["test"]]),
        "test_hard": held,
    }
    dataset = DatasetDict({name: Dataset.from_list(rows, features=features) for name, rows in splits.items()})

    def tiers(rows: list[dict[str, Any]], only_scoreable: bool = True) -> dict[int, int]:
        return dict(sorted(collections.Counter(r["difficulty_tier"] for r in rows if not only_scoreable or scoreable(r)).items()))

    summary = {
        "mining": dict(stats),
        "rows": {name: len(rows) for name, rows in splits.items()},
        "scoreable_by_tier": {name: tiers(rows) for name, rows in splits.items()},
        "train_mined_by_tier": tiers(rest),
    }
    print(json.dumps(summary, indent=2), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    dataset.save_to_disk(str(args.output_dir / "dataset"))
    if args.output_repo:
        dataset.push_to_hub(args.output_repo, private=False)
        from huggingface_hub import DatasetCard, HfApi

        api = HfApi()
        api.upload_file(path_or_fileobj=str(args.output_dir / "build-summary.json"), path_in_repo="build-summary.json",
                        repo_id=args.output_repo, repo_type="dataset")  # fmt: skip
        card = DatasetCard.load(args.output_repo, repo_type="dataset")
        card.text = card_text(summary)
        card.data.tags = sorted(set((getattr(card.data, "tags", None) or []) + ["agents", "url-extraction", "opencode"]))
        card.push_to_hub(args.output_repo, repo_type="dataset")
        print(f"published {args.output_repo}", flush=True)


def card_text(summary: dict[str, Any]) -> str:
    tiers = sorted({t for split in summary["scoreable_by_tier"].values() for t in split})
    head = "| Tier | Where the URLs are found | " + " | ".join(summary["scoreable_by_tier"]) + " |\n"
    head += "|---|---|" + "---:|" * len(summary["scoreable_by_tier"]) + "\n"
    body = "".join(
        f"| {t} | {TIER_NAMES.get(t, '')} | " + " | ".join(str(split.get(t, 0)) for split in summary["scoreable_by_tier"].values()) + " |\n"
        for t in tiers
    )
    return f"""
# Paper URL extraction v2 (difficulty-balanced)

Find the official GitHub repository and project page of an arXiv paper. v2 extends
[`nielsr/paper-url-extraction-v1-validated`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v1-validated)
with papers mined from Hugging Face daily papers to cover the hard cases the original benchmark lacks: URLs
that only appear in the PDF, GitHub repositories reachable only through the project page, and papers with no
official URL.

## Splits

- `train`: v1 train plus mined papers.
- `validation`, `test`: v1, unchanged, so earlier results stay comparable.
- `test_hard`: held-out mined papers, stratified by difficulty tier.

Scoreable papers per difficulty tier (the hardest source an agent must reach):

{head}{body}
## Labels

Each URL field has a label quality. Only `gold`, `silver` and `verified_null` are scored; `unknown` fields
are never treated as negatives.

- `gold`: the URL is in the paper's Hugging Face metadata.
- `silver`: found in the paper, abstract or project page, chosen by GLM-5.3 among the collected URLs only,
  and verified to resolve. GitHub redirects are accepted as aliases.
- `verified_null`: no official URL in HF metadata, the abstract, the PDF or the project page. For GitHub,
  papers whose title search finds a repository naming the paper in its README were downgraded to
  `unknown` (about a third of mined nulls in a spot check had code released elsewhere).

`mining_notes` records the adjudication and the GitHub re-check for mined rows.

Part of the [paper URL agent collection](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e).
"""


if __name__ == "__main__":
    main()
