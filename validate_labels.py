#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets>=4.0.0", "httpx>=0.27.0", "huggingface-hub>=1.13.0", "pydantic>=2.10,<2.13", "opencode-ai==0.1.0a36"]
# ///
"""Validate the reward before RL: re-check every scoreable label and test the scorer itself.

The harness-adapters recipe validates every RL task before training (`openenv validate`): the reference
answer must score full reward, doing nothing must score clearly lower, and known-wrong answers must fail.
Our "verifier" is URL matching against dataset labels, so validating it means validating the labels:

1. Label refresh. For each scoreable requested field, compare the dataset label with the paper's current
   HF metadata (`githubRepo` / `projectPage`). GitHub URLs are resolved through redirects first, so a
   renamed or transferred repository counts as the same repository.
     - agree               -> keep, with the redirect target as an accepted alias
     - label only          -> keep (metadata has no URL for this field)
     - metadata only        -> verified_null label contradicted by metadata -> stop scoring the field
     - conflict             -> stop scoring the field (quality set to "unknown")
2. Scorer checks on every remaining task (the openenv-validate analogues):
     - oracle:  answering the reference (and each alias) scores 1
     - floor:   answering nothing scores 0 on fields whose label is a URL
     - canary:  a plausible wrong answer (a different repo / another page) scores 0
   Any failure is a bug in the scorer or the data and is reported.

Writes a dataset with the original columns plus `reference_*_aliases` and `*_validation`, ready for
`--dataset` in the evaluator and the GRPO trainer.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset, DatasetDict, Features, List, Value, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_vllm import SCOREABLE_QUALITIES, URLFindings, canonical_url, score_findings  # noqa: E402

FIELDS = ("github", "project_page")


async def hf_metadata(client: httpx.AsyncClient, arxiv_id: str) -> dict[str, Any] | None:
    for attempt in range(4):
        try:
            response = await client.get(f"https://huggingface.co/api/papers/{arxiv_id}")
            if response.status_code == 404:
                return {}
            if response.status_code == 429:
                await asyncio.sleep(2 ** attempt * 5)
                continue
            response.raise_for_status()
            data = response.json()
            return {"github": data.get("githubRepo"), "project_page": data.get("projectPage")}
        except httpx.HTTPError:
            await asyncio.sleep(2 ** attempt)
    return None  # unknown: never treated as evidence


async def resolve_github(client: httpx.AsyncClient, url: str | None, cache: dict[str, str | None]) -> str | None:
    """Canonical URL GitHub redirects `url` to (renamed / transferred repos), or the URL itself."""
    key = canonical_url(url)
    if not key or not key.startswith("https://github.com/"):
        return key
    if key in cache:
        return cache[key]
    final = key
    for attempt in range(3):
        try:
            response = await client.head(key, follow_redirects=True)
            if response.status_code < 400:
                final = canonical_url(str(response.url)) or key
            break
        except httpx.HTTPError:
            await asyncio.sleep(2 ** attempt)
    cache[key] = final
    return final


def decide(label: str | None, quality: str, meta: str | None, label_res: str | None, meta_res: str | None) -> str:
    if quality not in SCOREABLE_QUALITIES:
        return "unscored"
    if label is None:  # verified_null
        return "contradicted" if meta else "agree_null"
    if not meta:
        return "label_only"
    if canonical_url(label) == canonical_url(meta) or (label_res and label_res == meta_res):
        return "agree"
    return "conflict"


async def validate_rows(rows: list[dict[str, Any]], concurrency: int) -> list[dict[str, Any]]:
    token = os.environ.get("HF_TOKEN")
    hf = httpx.AsyncClient(timeout=20, headers={"Authorization": f"Bearer {token}"} if token else {})
    gh = httpx.AsyncClient(timeout=20, headers={"User-Agent": "paper-url-label-check"})
    cache: dict[str, str | None] = {}
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        requested = [n for n in FIELDS if row[f"missing_{n}"] and row.get(f"{n}_label_quality") in SCOREABLE_QUALITIES]
        meta = None
        if requested:
            async with sem:
                meta = await hf_metadata(hf, row["arxiv_id"])
        for name in FIELDS:
            out[f"reference_{name}_aliases"] = []
            out[f"{name}_validation"] = "unscored"
            if name not in requested:
                continue
            if meta is None:
                out[f"{name}_validation"] = "metadata_unavailable"  # keep the original label
                continue
            label = row.get(f"reference_{name}_url")
            current = meta.get(name)
            label_res = meta_res = None
            if name == "github":
                async with sem:
                    label_res = await resolve_github(gh, label, cache)
                    meta_res = await resolve_github(gh, current, cache)
            status = decide(label, row[f"{name}_label_quality"], current, label_res, meta_res)
            out[f"{name}_validation"] = status
            if status in {"conflict", "contradicted"}:
                out[f"{name}_label_quality"] = "unknown"  # stop scoring this field
                continue
            aliases = {canonical_url(label)} if label else set()
            for extra in (label_res, current if status == "agree" else None, meta_res if status == "agree" else None):
                if extra and canonical_url(extra) not in aliases:
                    aliases.add(canonical_url(extra))
            aliases.discard(canonical_url(label))
            out[f"reference_{name}_aliases"] = sorted(a for a in aliases if a)
        return out

    results = await asyncio.gather(*(one(row) for row in rows))
    await hf.aclose()
    await gh.aclose()
    return list(results)


def scorer_checks(rows: list[dict[str, Any]]) -> collections.Counter:
    """Oracle / floor / canary checks of the reward on every task that still has a scoreable field."""
    report: collections.Counter = collections.Counter()

    def answer(**urls: str | None) -> URLFindings:
        return URLFindings(reasoning="check", github_url=urls.get("github"), project_page_url=urls.get("project_page"))

    for row in rows:
        scored = [n for n in FIELDS if row[f"missing_{n}"] and row.get(f"{n}_label_quality") in SCOREABLE_QUALITIES]
        if not scored:
            continue
        report["tasks"] += 1
        oracle = {n: row.get(f"reference_{n}_url") for n in FIELDS}
        result = score_findings(row, answer(**oracle))
        report["oracle_ok" if result["all_scoreable_correct"] else "oracle_FAIL"] += 1
        for name in scored:
            for alias in row.get(f"reference_{name}_aliases") or []:
                alt = dict(oracle, **{name: alias})
                ok = score_findings(row, answer(**alt))["fields"][name]
                report["alias_ok" if ok else "alias_FAIL"] += 1
            if row.get(f"reference_{name}_url"):
                floor = score_findings(row, answer())["fields"][name]
                report["floor_ok" if floor is False else "floor_FAIL"] += 1
                canary_url = (
                    "https://github.com/not-the-authors/fork-of-this-paper"
                    if name == "github"
                    else "https://example.org/unrelated-project/"
                )
                canary = score_findings(row, answer(**dict(oracle, **{name: canary_url})))["fields"][name]
                report["canary_ok" if canary is False else "canary_FAIL"] += 1
            else:  # verified_null: any URL is wrong, null is right
                canary = score_findings(row, answer(**dict(oracle, **{name: "https://example.org/x"})))["fields"][name]
                report["null_canary_ok" if canary is False else "null_canary_FAIL"] += 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="nielsr/paper-url-extraction-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("validated"))
    parser.add_argument("--output-repo", help="Publish the validated dataset to this Hub dataset repo")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--limit", type=int, help="Rows per split, for a quick check")
    args = parser.parse_args()

    splits = {}
    summary: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        source = load_dataset(args.dataset, split=split)
        features = Features(
            {
                **source.features,
                **{f"reference_{n}_aliases": List(Value("string")) for n in FIELDS},
                **{f"{n}_validation": Value("string") for n in FIELDS},
            }
        )
        rows = [dict(r) for r in source]
        if args.limit:
            rows = rows[: args.limit]
        validated = asyncio.run(validate_rows(rows, args.concurrency))
        counts = collections.Counter(
            f"{n}:{r[f'{n}_validation']}" for r in validated for n in FIELDS if r[f"{n}_validation"] != "unscored"
        )
        checks = scorer_checks(validated)
        summary[split] = {"rows": len(rows), "label_status": dict(sorted(counts.items())), "scorer_checks": dict(checks)}
        print(split, json.dumps(summary[split], indent=2), flush=True)
        splits[split] = Dataset.from_list(validated, features=features)

    failures = {s: {k: v for k, v in d["scorer_checks"].items() if k.endswith("FAIL")} for s, d in summary.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    DatasetDict(splits).save_to_disk(str(args.output_dir / "dataset"))
    print("scorer check failures:", failures, flush=True)
    if args.output_repo:
        DatasetDict(splits).push_to_hub(args.output_repo, private=not args.public)
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(args.output_dir / "validation-summary.json"),
            path_in_repo="validation-summary.json",
            repo_id=args.output_repo,
            repo_type="dataset",
        )
        print(f"published {args.output_repo}", flush=True)


if __name__ == "__main__":
    main()
