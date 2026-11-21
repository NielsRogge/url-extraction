#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Re-check mined `verified_null` GitHub labels against GitHub search.

`mine_papers.py` labels a paper's GitHub field `verified_null` when no repository appears in HF metadata, the
abstract, the PDF or the project page. A spot check found that about a third of those papers do have an
official repository that the paper never links (released later, or linked only from an org page). Scoring
"no repository" as correct would then punish an agent that finds the real one.

For every `verified_null` GitHub field this script searches GitHub for the paper title, skips list-style
repositories (awesome lists, paper collections), and reads the README of the top candidates. A README that
contains the paper's arXiv id or exact title in its first 3,000 characters is evidence of an implementation, and the field is
downgraded to `unknown` (no longer scored). The candidate is recorded in `github_null_recheck` for review.

Uses the `gh` CLI (GitHub search allows ~30 requests per minute).

    uv run recheck_nulls.py --input mined/part-1.jsonl --output mined/part-1.rechecked.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

README_HEAD_CHARS = 3000
LIST_LIKE = re.compile(r"awesome|paper|survey|reading|list|daily|arxiv|collection|resources|notes|study|hub|blog", re.IGNORECASE)


def gh(path: str) -> Any:
    for attempt in range(4):
        result = subprocess.run(["gh", "api", "-X", "GET", path], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return json.loads(result.stdout)
        if "rate limit" in result.stderr.lower():
            time.sleep(30 * (attempt + 1))
            continue
        return None
    return None


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def candidates(title: str, limit: int) -> list[str]:
    # GitHub search rejects long queries with qualifiers; search the title words, then the short name before ":".
    clean = re.sub(r"[\"\n]+", " ", title).strip()[:200]
    queries = [f'"{clean}" in:readme,description,name']
    seen: list[str] = []
    for query in queries:
        data = gh(f"search/repositories?q={urllib.parse.quote(query)}&per_page=10")
        time.sleep(2.5)
        for item in (data or {}).get("items", []):
            name = item["full_name"]
            if LIST_LIKE.search(name) or name in seen:
                continue
            seen.append(name)
    return seen[:limit]


def readme_mentions(repo: str, arxiv_id: str, title: str) -> bool:
    """Read the default-branch README from raw.githubusercontent.com (no API quota)."""
    for name in ("README.md", "readme.md", "README.rst", "README"):
        try:
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/HEAD/{name}", timeout=20) as response:
                raw = response.read(README_HEAD_CHARS * 4)
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue
        else:
            # Official repositories name the paper at the top; lists that cite it bury it further down.
            text = raw.decode("utf-8", "replace")[:README_HEAD_CHARS]
            return arxiv_id in text or normalize(title) in normalize(text)
    return False


def recheck(row: dict[str, Any], limit: int) -> dict[str, Any]:
    if row.get("github_label_quality") != "verified_null":
        return row
    hits = [repo for repo in candidates(row["title"], limit) if readme_mentions(repo, str(row["arxiv_id"]), row["title"])]
    row = dict(row)
    row["github_null_recheck"] = {"candidates_with_paper_in_readme": hits}
    if hits:
        row["github_label_quality"] = "unknown"
        row["github_validation"] = "null_contradicted_by_github_search"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=5, help="READMEs to read per paper")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    done = {}
    if args.output.exists():  # resume: rows already re-checked are kept as written
        for line in args.output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["arxiv_id"])] = row
    downgraded = checked = 0
    with args.output.open("a") as handle:
        for row in rows:
            if str(row["arxiv_id"]) in done:
                continue
            new = recheck(row, args.candidates)
            if row.get("github_label_quality") == "verified_null":
                checked += 1
                hits = new["github_null_recheck"]["candidates_with_paper_in_readme"]
                downgraded += bool(hits)
                print(f"[recheck] {checked} {row['arxiv_id']} {row['title'][:50]!r} -> {hits}", flush=True)
            handle.write(json.dumps(new) + "\n")
            handle.flush()
    print(f"[recheck] {downgraded}/{checked} verified_null GitHub labels contradicted this run; wrote {args.output}", flush=True)

if __name__ == "__main__":
    main()
