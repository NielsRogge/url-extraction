#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets>=4.0.0", "httpx>=0.27.0", "huggingface-hub>=1.13.0", "pypdf>=5.0.0",
#                 "pydantic>=2.10,<2.13", "opencode-ai==0.1.0a36"]
# ///
"""Rank every labelled paper by how hard its URLs are to find.

Difficulty = the hardest source an agent must reach to find every labelled URL of a paper:

  1 api          the HF paper API (`hf papers info`) already returns them
  2 abstract     they are in the abstract (which the agent's prompt already contains)
  3 page1        they are on the first PDF page (visible text or hyperlink annotations)
  4 page2        ... on the second PDF page
  5 project_page the GitHub URL is only reachable through the project page's HTML
  6 deeper       they appear later in the paper
  7 unfound      a labelled URL appears in none of these sources (weak label; excluded from scoring)
  0 null         every requested field is a verified "no URL" (the agent must read to confirm absence)

Finding a labelled URL in the paper or on its project page is also strong evidence that the label is
right, so this doubles as label verification. PDFs are fetched from arXiv only when the API and the
abstract do not already explain the labels, one request per three seconds as arXiv asks.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset, DatasetDict, Features, Value, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_vllm import SCOREABLE_QUALITIES, canonical_url  # noqa: E402

FIELDS = ("github", "project_page")
TIERS = {"null": 0, "api": 1, "abstract": 2, "page1": 3, "page2": 4, "project_page": 5, "deeper": 6, "unfound": 7}
UA = {"User-Agent": "paper-url-difficulty-tiering (research; polite 1 req/3s to arXiv)"}


def url_key(url: str | None) -> str | None:
    """Host+path key used for matching inside free text (lowercase, no scheme/www/trailing slash)."""
    canon = canonical_url(url)
    if not canon:
        return None
    return canon.removeprefix("https://")


def squash(text: str) -> str:
    """Lowercase and drop whitespace (PDF text breaks URLs across lines)."""
    return re.sub(r"\s+", "", text.lower()).replace("-\n", "")


def contains(haystack: str, url: str | None, aliases: list[str]) -> bool:
    keys = [url_key(u) for u in [url, *aliases] if u]
    return any(k and squash(k) in haystack for k in keys)


def pdf_pages(data: bytes) -> list[str]:
    """Per-page text plus that page's link annotations, squashed."""
    from pypdf import PdfReader

    pages = []
    reader = PdfReader(io.BytesIO(data))
    for page in reader.pages:
        text = ""
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - a bad page should not drop the paper
            pass
        links = []
        for annot in page.get("/Annots") or []:
            try:
                obj = annot.get_object()
                uri = (obj.get("/A") or {}).get("/URI")
                if uri:
                    links.append(str(uri))
            except Exception:  # noqa: BLE001
                continue
        pages.append(squash(text + " " + " ".join(links)))
    return pages


class Arxiv:
    def __init__(self) -> None:
        self.client = httpx.Client(timeout=60, headers=UA, follow_redirects=True)
        self.last = 0.0

    def pdf(self, arxiv_id: str) -> bytes | None:
        for attempt in range(3):
            wait = 3.0 - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            try:
                response = self.client.get(f"https://arxiv.org/pdf/{arxiv_id}")
                if response.status_code == 200 and response.content[:4] == b"%PDF":
                    return response.content
                if response.status_code in (403, 429, 503):
                    time.sleep(30 * (attempt + 1))
            except httpx.HTTPError:
                time.sleep(5)
        return None


def hf_meta(client: httpx.Client, arxiv_id: str) -> dict[str, Any]:
    for attempt in range(3):
        try:
            response = client.get(f"https://huggingface.co/api/papers/{arxiv_id}")
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            data = response.json()
            return {"github": data.get("githubRepo"), "project_page": data.get("projectPage"), "summary": data.get("summary") or ""}
        except httpx.HTTPError:
            time.sleep(2 ** attempt)
    return {}


def page_html(client: httpx.Client, url: str) -> str:
    try:
        response = client.get(url)
        if response.status_code == 200:
            return squash(response.text[:2_000_000])
    except httpx.HTTPError:
        pass
    return ""


def tier_row(row: dict[str, Any], hf: httpx.Client, web: httpx.Client, arxiv: Arxiv) -> dict[str, Any]:
    out = dict(row)
    labelled = {
        n: (row.get(f"reference_{n}_url"), list(row.get(f"reference_{n}_aliases") or []))
        for n in FIELDS
        if row[f"missing_{n}"] and row.get(f"{n}_label_quality") in SCOREABLE_QUALITIES
    }
    evidence: dict[str, str] = {}
    urls = {n: v for n, v in labelled.items() if v[0]}
    for n in labelled:
        if n not in urls:
            evidence[n] = "null"
    if urls:
        meta = hf_meta(hf, row["arxiv_id"])
        abstract = squash((row.get("abstract") or "") + " " + meta.get("summary", ""))
        pending = {}
        for n, (url, aliases) in urls.items():
            if meta.get(n) and canonical_url(meta[n]) in {canonical_url(u) for u in [url, *aliases]}:
                evidence[n] = "api"
            elif contains(abstract, url, aliases):
                evidence[n] = "abstract"
            else:
                pending[n] = (url, aliases)
        if pending:
            data = arxiv.pdf(row["arxiv_id"])
            pages = pdf_pages(data) if data else []
            for n, (url, aliases) in list(pending.items()):
                for index, page in enumerate(pages):
                    if contains(page, url, aliases):
                        evidence[n] = "page1" if index == 0 else "page2" if index == 1 else "deeper"
                        break
                else:
                    evidence[n] = "unfound" if pages else "pdf_unavailable"
            # GitHub reachable only through the project page?
            gh = urls.get("github")
            pp_url = urls.get("project_page", (None, []))[0] or row.get("known_project_page_url") or meta.get("project_page")
            if gh and evidence.get("github") in {"unfound", "deeper", "pdf_unavailable"} and pp_url:
                if contains(page_html(web, pp_url), gh[0], gh[1]):
                    evidence["github"] = "project_page"
    out["evidence_github"] = evidence.get("github", "")
    out["evidence_project_page"] = evidence.get("project_page", "")
    found = [e for e in evidence.values() if e in TIERS]
    if not evidence:
        out["difficulty_tier"] = -1  # nothing scoreable
    elif any(e in {"unfound", "pdf_unavailable"} for e in evidence.values()):
        out["difficulty_tier"] = TIERS["unfound"]
    else:
        out["difficulty_tier"] = max(TIERS[e] for e in found)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="nielsr/paper-url-extraction-v1-validated")
    parser.add_argument("--output-dir", type=Path, default=Path("tiered"))
    parser.add_argument("--output-repo")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    hf = httpx.Client(timeout=20)
    web = httpx.Client(timeout=20, headers=UA, follow_redirects=True)
    arxiv = Arxiv()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits, summary = {}, {}
    for split in ("train", "validation", "test"):
        source = load_dataset(args.dataset, split=split)
        features = Features(
            {
                **source.features,
                "evidence_github": Value("string"),
                "evidence_project_page": Value("string"),
                "difficulty_tier": Value("int32"),
            }
        )
        rows = [dict(r) for r in source]
        if args.limit:
            rows = rows[: args.limit]
        tiered = []
        started = time.monotonic()
        for i, row in enumerate(rows, 1):
            tiered.append(tier_row(row, hf, web, arxiv))
            if i % 200 == 0:
                print(f"[tier] {split} {i}/{len(rows)} ({(time.monotonic() - started) / 60:.0f} min)", flush=True)
        counts = collections.Counter(r["difficulty_tier"] for r in tiered if r["difficulty_tier"] >= 0)
        evid = collections.Counter(f"{n}:{r[f'evidence_{n}']}" for r in tiered for n in FIELDS if r[f"evidence_{n}"])
        summary[split] = {
            "tiers": {name: counts.get(t, 0) for name, t in sorted(TIERS.items(), key=lambda kv: kv[1])},
            "evidence": dict(sorted(evid.items())),
        }
        print(split, json.dumps(summary[split], indent=2), flush=True)
        splits[split] = Dataset.from_list(tiered, features=features)
        (args.output_dir / "tier-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    DatasetDict(splits).save_to_disk(str(args.output_dir / "dataset"))
    if args.output_repo:
        DatasetDict(splits).push_to_hub(args.output_repo, private=not args.public)
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(args.output_dir / "tier-summary.json"), path_in_repo="tier-summary.json",
            repo_id=args.output_repo, repo_type="dataset",
        )  # fmt: skip
        print(f"[tier] published {args.output_repo}", flush=True)


if __name__ == "__main__":
    main()
