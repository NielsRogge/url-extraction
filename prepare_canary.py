#!/usr/bin/env python3
# /// script
# dependencies = [
#   "datasets>=4.0.0",
#   "huggingface-hub>=1.13.0",
#   "httpx>=0.27.0",
# ]
# ///
"""Build a deduplicated, consensus-labelled canary set from URL traces."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def canonical_arxiv_id(value: Any) -> str | None:
    match = ARXIV_ID_RE.fullmatch(str(value or "").strip())
    return match.group(1) if match else None


def canonical_url(value: Any) -> str | None:
    text = str(value or "").strip().strip("`").rstrip(".,;)")
    if not text or text.lower() in {"null", "none"}:
        return None
    parsed = urlparse(text if "://" in text else f"https://{text}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    host = parsed.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if host == "github.com":
        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            return None
        path = "/" + "/".join(parts[:2])
    return urlunparse(("https", host, path, "", "", ""))


def parse_output_raw(raw: str | None) -> tuple[str | None, str | None] | None:
    candidates = JSON_FENCE_RE.findall(raw or "")
    if (raw or "").lstrip().startswith("{"):
        candidates.append((raw or "").strip())
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        return canonical_url(payload.get("github_url")), canonical_url(payload.get("project_page_url"))
    return None


def load_source(args: argparse.Namespace) -> Dataset:
    if args.source_parquet:
        shards = [Dataset.from_parquet(str(path)) for path in sorted(args.source_parquet)]
        return concatenate_datasets(shards)
    return load_dataset(args.source_dataset, args.source_config, split=args.source_split)


def seen_arxiv_ids(dataset_id: str) -> set[str]:
    dataset = load_dataset(dataset_id)
    return {
        canonical
        for split in dataset.values()
        for value in split["arxiv_id"]
        if (canonical := canonical_arxiv_id(value)) is not None
    }


def consensus(value_votes: list[str | None], min_votes: int, min_agreement: float) -> dict[str, Any]:
    counts = Counter(value_votes)
    value, votes = counts.most_common(1)[0]
    total = len(value_votes)
    agreement = votes / total
    accepted = votes >= min_votes and agreement >= min_agreement
    return {
        "value": value if accepted else None,
        "votes": votes,
        "total": total,
        "agreement": agreement,
        "accepted": accepted,
    }


def aggregate_rows(rows: Dataset, excluded: set[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    votes: dict[str, dict[str, list[str | None]]] = defaultdict(lambda: {"github": [], "project_page": []})
    for row in rows:
        arxiv_id = canonical_arxiv_id(row.get("arxiv_id"))
        parsed = parse_output_raw(row.get("output_raw"))
        if arxiv_id is None or arxiv_id in excluded or parsed is None:
            continue
        votes[arxiv_id]["github"].append(parsed[0])
        votes[arxiv_id]["project_page"].append(parsed[1])

    aggregated: list[dict[str, Any]] = []
    for arxiv_id, fields in sorted(votes.items()):
        github = consensus(fields["github"], args.min_votes, args.min_agreement)
        project_page = consensus(fields["project_page"], args.min_votes, args.min_agreement)
        if not github["accepted"] or not project_page["accepted"]:
            continue
        aggregated.append({"arxiv_id": arxiv_id, "github": github, "project_page": project_page})
    return aggregated


def stratum(row: dict[str, Any]) -> str:
    github = row["github"]["value"] is not None
    project = row["project_page"]["value"] is not None
    if github and project:
        return "both"
    if github:
        return "github_only"
    if project:
        return "project_only"
    return "neither"


def stratified_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    selected: list[dict[str, Any]] = []
    order = ("both", "github_only", "project_only", "neither")
    while len(selected) < limit and any(groups.values()):
        for name in order:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop(0))
    return selected


def arxiv_metadata(arxiv_id: str) -> tuple[str, str]:
    response = httpx.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": arxiv_id},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", namespace)
    if entry is None:
        raise LookupError(f"arXiv returned no metadata for {arxiv_id}")
    title = " ".join((entry.findtext("atom:title", default="", namespaces=namespace)).split())
    abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=namespace)).split())
    if not title or not abstract:
        raise LookupError(f"arXiv returned incomplete metadata for {arxiv_id}")
    return title, abstract


def paper_metadata(api: HfApi, arxiv_id: str) -> tuple[str, str]:
    try:
        paper = api.paper_info(arxiv_id)
        return paper.title.strip(), paper.summary.strip()
    except HfHubHTTPError as error:
        if error.response is None or error.response.status_code != 404:
            raise
        return arxiv_metadata(arxiv_id)


def add_paper_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    api = HfApi()
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        title, abstract = paper_metadata(api, row["arxiv_id"])
        candidates.append(
            {
                "paper_id": -(index + 1),
                "arxiv_id": row["arxiv_id"],
                "title": title,
                "abstract": abstract,
                "missing_github": True,
                "missing_project_page": True,
                "known_github_url": None,
                "known_project_page_url": None,
                "priority": 0,
                "priority_source": "community_science_trace_consensus",
                "reference_github_url": row["github"]["value"],
                "reference_project_page_url": row["project_page"]["value"],
                "github_label_quality": "consensus",
                "project_page_label_quality": "consensus",
                "trace_consensus": {
                    "github": {key: row["github"][key] for key in ("votes", "total", "agreement")},
                    "project_page": {key: row["project_page"][key] for key in ("votes", "total", "agreement")},
                },
            }
        )
    return candidates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-parquet", type=Path, nargs="+")
    source.add_argument("--source-dataset", default="huggingface/community-science-llm-traces")
    parser.add_argument("--source-config", default="step_1_url_extraction")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--seen-dataset", default="nielsr/paper-url-extraction-v1")
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--min-agreement", type=float, default=0.6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("canary-candidates.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_source(args)
    excluded = seen_arxiv_ids(args.seen_dataset)
    aggregated = aggregate_rows(source, excluded, args)
    selected = stratified_limit(aggregated, args.limit)
    candidates = add_paper_metadata(selected)
    write_jsonl(args.output, candidates)
    print(
        {
            "source_rows": len(source),
            "excluded_arxiv_ids": len(excluded),
            "unseen_consensus_papers": len(aggregated),
            "written": len(candidates),
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()
