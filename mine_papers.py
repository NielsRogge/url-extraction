#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets>=4.0.0", "httpx>=0.27.0", "huggingface-hub>=1.13.0", "pypdf>=5.0.0",
#                 "pydantic>=2.10,<2.13", "opencode-ai==0.1.0a36"]
# ///
"""Mine new papers into a difficulty-balanced URL-extraction dataset.

Most papers in the original dataset are easy: the HF paper API already returns their URLs. This miner
looks for the hard cases (URLs only in the PDF's first pages, or GitHub reachable only through the
project page) and for verified "no URL" papers, and keeps an easy share for balance.

Per candidate paper (from HF daily papers, minus any paper already in the source dataset):
  1. HF metadata (githubRepo, projectPage, abstract).
  2. Evidence: every GitHub/web URL in the abstract and the full arXiv PDF (page text + link
     annotations, page index recorded), plus GitHub links on the project page's HTML.
  3. Adjudication: an LLM (default GLM-5.3 via Inference Providers) picks the official repository and
     project page *from the collected URLs only*, or none.
  4. Verification: chosen URLs must resolve (GitHub redirects become aliases).
  5. Labels: HF metadata -> gold; adjudicated + live -> silver; nothing official anywhere -> verified_null.
     Tier = hardest source needed (same scale as tier_labels.py).
Sampling stops filling a tier once its quota is reached. Output rows match the validated dataset schema.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_vllm import canonical_url  # noqa: E402

TIERS = {"null": 0, "api": 1, "abstract": 2, "page1": 3, "page2": 4, "project_page": 5, "deeper": 6}
UA = {"User-Agent": "paper-url-dataset-miner (research; 1 req/3s to arXiv)"}
URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.IGNORECASE)
# Never the paper's own repo/page: generic hosts, dependencies, datasets, preprint servers.
SKIP_HOSTS = ("arxiv.org", "doi.org", "huggingface.co", "hf.co", "openreview.net", "aclanthology.org", "creativecommons.org",
              "proceedings.", "ieee.org", "acm.org", "springer.com", "neurips.cc", "nips.cc", "icml.cc", "iclr.cc",
              "cvf.com", "thecvf.com", "orcid.org", "twitter.com", "x.com", "youtube.com", "wikipedia.org",
              "pytorch.org", "tensorflow.org", "kaggle.com", "zenodo.org", "paperswithcode.com", "google.com/scholar")
GENERIC_REPOS = {"huggingface/transformers", "pytorch/pytorch", "openai/gpt-2", "facebookresearch/fairseq", "google-research/bert",
                 "vllm-project/vllm", "huggingface/datasets", "huggingface/peft", "huggingface/trl", "microsoft/deepspeed"}

ADJUDICATION_PROMPT = """You label research papers with their official code repository and project page.

Paper: {title} (arXiv {arxiv_id})
Abstract: {abstract}

Candidate URLs found in the paper and on its project page, each with where it appeared and the text around it:
{candidates}

Rules:
- github_url: the authors' own official implementation of THIS paper, on github.com. Not a dependency, a
  baseline, a dataset, a fork, or a general library the paper merely uses. null if none is official.
- project_page_url: an official page specific to THIS paper (often *.github.io or a lab site). Not arXiv,
  Hugging Face, a GitHub repository, a generic lab/author homepage, or a venue page. null if none.
- Choose ONLY from the candidate URLs above (copy them exactly), or null. Never invent a URL.
Answer with one JSON object: {{"github_url": ..., "project_page_url": ..., "reasoning": "<one sentence>"}}"""


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


# ------------------------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------------------------


def daily_ids(client: httpx.Client, dates: list[dt.date]) -> list[str]:
    ids: list[str] = []
    for day in dates:
        try:
            response = client.get("https://huggingface.co/api/daily_papers", params={"date": day.isoformat(), "limit": 100})
            if response.status_code == 200 and isinstance(response.json(), list):
                ids += [item["paper"]["id"] for item in response.json() if item.get("paper", {}).get("id")]
        except (httpx.HTTPError, ValueError, KeyError):
            continue
    return list(dict.fromkeys(i for i in ids if re.fullmatch(r"\d{4}\.\d{4,5}", i)))


def hf_paper(client: httpx.Client, arxiv_id: str) -> dict[str, Any] | None:
    for attempt in range(3):
        try:
            response = client.get(f"https://huggingface.co/api/papers/{arxiv_id}")
            if response.status_code == 404:
                return None
            if response.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            return {
                "arxiv_id": arxiv_id,
                "title": data.get("title") or "",
                "abstract": " ".join((data.get("summary") or "").split()),
                "github": data.get("githubRepo"),
                "project_page": data.get("projectPage"),
            }
        except (httpx.HTTPError, ValueError):
            time.sleep(2 ** attempt)
    return None


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


def pdf_pages(data: bytes) -> list[tuple[str, list[str]]]:
    """(page text, link annotation URIs) per page."""
    from pypdf import PdfReader

    pages = []
    for page in PdfReader(io.BytesIO(data)).pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        links = []
        for annot in page.get("/Annots") or []:
            try:
                uri = (annot.get_object().get("/A") or {}).get("/URI")
                if uri:
                    links.append(str(uri))
            except Exception:  # noqa: BLE001
                continue
        pages.append((text, links))
    return pages


# ------------------------------------------------------------------------------------------------
# Evidence and adjudication
# ------------------------------------------------------------------------------------------------


def normalize_candidate(url: str) -> str | None:
    url = url.rstrip(".,;:")
    m = GITHUB_RE.search(url)
    if m:
        owner, repo = m.group(1), re.sub(r"\.git$", "", m.group(2))
        if owner.lower() in {"orgs", "features", "topics", "sponsors", "settings", "login"}:
            return None
        return f"https://github.com/{owner}/{repo}"
    canon = canonical_url(url)
    if not canon or any(h in canon for h in SKIP_HOSTS):
        return None
    if re.search(r"\.(pdf|png|jpg|jpeg|gif|zip|tar|gz|json|csv|txt)$", canon):
        return None
    return canon


def collect(text: str, links: list[str], where: str, found: dict[str, dict[str, Any]]) -> None:
    flat = " ".join(text.split())
    # PDF extraction splits URLs across lines; re-join "github.com/ owner/ repo" style breaks.
    joined = flat
    for _ in range(3):  # only a URL broken right after "/" or "-" is re-joined
        joined = re.sub(r"((?:https?://)\S*-)\s+(?=\w)", r"\1", joined)  # hyphenated line break
        joined = re.sub(r"(github\.com/[\w.-]+/)\s+(?=\w)", r"\1", joined)  # owner/ <break> repo
    for raw in [*URL_RE.findall(joined), *[f"https://{m.group(0)}" for m in GITHUB_RE.finditer(joined)], *links]:
        url = normalize_candidate(raw)
        if not url or url in found:
            continue
        key = url.split("://", 1)[-1][:40]
        index = joined.lower().find(key.lower()[:25])
        snippet = joined[max(0, index - 160): index + 160] if index >= 0 else "(hyperlink)"
        found[url] = {"where": where, "context": snippet}


def adjudicate(client: httpx.Client, model: str, bill_to: str | None, paper: dict[str, Any], found: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not found:
        return {"github_url": None, "project_page_url": None, "reasoning": "no candidate URLs"}
    lines = [f"- {url}  [{info['where']}]  ...{info['context']}..." for url, info in list(found.items())[:40]]
    prompt = ADJUDICATION_PROMPT.format(
        title=paper["title"], arxiv_id=paper["arxiv_id"], abstract=paper["abstract"][:1500], candidates="\n".join(lines)
    )
    headers = {"X-HF-Bill-To": bill_to} if bill_to else {}
    for attempt in range(3):
        try:
            response = client.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                      "max_tokens": 1500, "response_format": {"type": "json_object"}},
            )  # fmt: skip
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            answer = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
            # Enforce "choose only from candidates".
            for key in ("github_url", "project_page_url"):
                value = normalize_candidate(str(answer.get(key) or "")) if answer.get(key) else None
                answer[key] = value if value in found else None
            return answer
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return {"github_url": None, "project_page_url": None, "reasoning": "adjudication failed", "failed": True}


def live(client: httpx.Client, url: str) -> tuple[bool, str | None]:
    """(reachable, canonical redirect target if different)."""
    try:
        response = client.get(url, follow_redirects=True)
        final = canonical_url(str(response.url))
        return response.status_code < 400, final if final != canonical_url(url) else None
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):  # malformed URLs from PDF text
        return False, None


# ------------------------------------------------------------------------------------------------
# Per-paper labelling
# ------------------------------------------------------------------------------------------------


def label_paper(paper: dict[str, Any], arxiv: Arxiv, web: httpx.Client, llm: httpx.Client, args: argparse.Namespace) -> dict[str, Any] | None:
    labels: dict[str, dict[str, Any]] = {}
    for name in ("github", "project_page"):
        if paper.get(name):
            labels[name] = {"url": canonical_url(paper[name]), "quality": "gold", "evidence": "api", "aliases": []}
    found: dict[str, dict[str, Any]] = {}
    pages: list[tuple[str, list[str]]] = []
    if len(labels) < 2:
        collect(paper["abstract"], [], "abstract", found)
        data = arxiv.pdf(paper["arxiv_id"])
        if data is None:
            return None
        try:
            pages = pdf_pages(data)
        except Exception:  # noqa: BLE001
            return None
        for index, (text, links) in enumerate(pages):
            collect(text, links, "page1" if index == 0 else "page2" if index == 1 else "deeper", found)
        # GitHub links reachable only through a project page (known or found in the paper).
        project_pages = [labels["project_page"]["url"]] if "project_page" in labels else [
            u for u, i in found.items() if "github.com" not in u and i["where"] in {"abstract", "page1", "page2"}
        ][:3]
        for page_url in project_pages:
            try:
                html = web.get(page_url, follow_redirects=True).text[:2_000_000]
            except (httpx.HTTPError, httpx.InvalidURL, ValueError):
                continue
            for m in GITHUB_RE.finditer(html):
                url = normalize_candidate(f"https://{m.group(0)}")
                if url and url not in found:
                    found[url] = {"where": "project_page", "context": f"linked from {page_url}"}
        if "github" in labels:
            found = {u: i for u, i in found.items() if "github.com" not in u}
        if "project_page" in labels:
            found = {u: i for u, i in found.items() if "github.com" in u}
        answer = adjudicate(llm, args.model, args.bill_to, paper, found)
        if answer.get("failed"):
            return None
        for name, key in (("github", "github_url"), ("project_page", "project_page_url")):
            if name in labels:
                continue
            url = answer.get(key)
            if url:
                if f"{url.split('github.com/')[-1]}".lower() in GENERIC_REPOS:
                    labels[name] = {"url": None, "quality": "unknown", "evidence": "", "aliases": []}
                    continue
                ok, redirect = live(web, url)
                if not ok:
                    labels[name] = {"url": url, "quality": "unknown", "evidence": "", "aliases": []}  # unverifiable
                    continue
                labels[name] = {"url": url, "quality": "silver", "evidence": found[url]["where"],
                                "aliases": [redirect] if redirect else []}  # fmt: skip
            else:
                # "No URL" is only verified if the paper offers no candidate of that kind at all.
                kind = [u for u in found if ("github.com" in u) == (name == "github")]
                labels[name] = {"url": None, "quality": "verified_null" if not kind else "unknown", "evidence": "null" if not kind else "", "aliases": []}  # fmt: skip
    scored = [lab for lab in labels.values() if lab["quality"] in {"gold", "silver", "verified_null"}]
    if not scored:
        return None
    tier = max(TIERS[lab["evidence"]] for lab in scored)
    row = {
        "paper_id": int(paper["arxiv_id"].replace(".", "")) + 900_000_000_000,  # disjoint from source ids
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "missing_github": True,
        "missing_project_page": True,
        "known_github_url": None,
        "known_project_page_url": None,
        "source": "hf-daily-papers-mined",
        "stratum": f"tier{tier}",
        "sampling_weight": 1.0,
        "priority": 0,
        "priority_source": "mined",
        "difficulty_tier": tier,
    }
    for name in ("github", "project_page"):
        lab = labels.get(name, {"url": None, "quality": "unknown", "evidence": "", "aliases": []})
        row[f"reference_{name}_url"] = lab["url"]
        row[f"{name}_label_quality"] = lab["quality"]
        row[f"reference_{name}_aliases"] = lab["aliases"]
        row[f"{name}_validation"] = "mined"
        row[f"evidence_{name}"] = lab["evidence"]
    row["adjudication"] = json.dumps({"candidates": len(found)})
    return row


# ------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exclude-dataset", default="nielsr/paper-url-extraction-v1-validated")
    parser.add_argument("--start", default="2023-06-01")
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--days", type=int, default=400, help="Random days sampled between start and end")
    parser.add_argument("--quota", default="1:300,2:200,3:450,4:200,5:150,6:100,0:250",
                        help="Target papers per tier (tier:count)")  # fmt: skip
    parser.add_argument("--model", default="zai-org/GLM-5.3")
    parser.add_argument("--bill-to", default=os.environ.get("HF_BILL_TO"))
    parser.add_argument("--max-hours", type=float, default=7.0)
    parser.add_argument("--output", type=Path, default=Path("mined/rows.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-repo", help="Dataset repo to upload the rows file to (periodically)")
    parser.add_argument("--path-in-repo", default="mined/rows.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    hf = httpx.Client(timeout=20, headers={"Authorization": f"Bearer {token}"} if token else {})
    web = httpx.Client(timeout=20, headers=UA)
    llm = httpx.Client(timeout=180, headers={"Authorization": f"Bearer {token}"})
    arxiv = Arxiv()
    quota = {int(k): int(v) for k, v in (item.split(":") for item in args.quota.split(","))}
    have: collections.Counter = collections.Counter()
    seen = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():  # resume
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            seen.add(row["arxiv_id"])
            have[row["difficulty_tier"]] += 1
    excluded = {str(r["arxiv_id"]) for split in ("train", "validation", "test") for r in load_dataset(args.exclude_dataset, split=split)}

    rng = random.Random(args.seed)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    span = (end - start).days
    dates = sorted({start + dt.timedelta(days=rng.randrange(span)) for _ in range(args.days)})
    ids = [i for i in daily_ids(hf, dates) if i not in excluded and i not in seen]
    rng.shuffle(ids)
    print(f"[mine] {len(ids)} candidate papers from {len(dates)} days", flush=True)

    with ThreadPoolExecutor(8) as pool:
        metas = [m for m in pool.map(lambda i: hf_paper(hf, i), ids) if m and m["abstract"]]
    # Hard / no-URL candidates first (HF lacks a URL), tier-5 candidates (project page, no repo) first of all.
    metas.sort(key=lambda m: (0 if (m["project_page"] and not m["github"]) else 1 if not m["github"] else 2 if not m["project_page"] else 3))
    print(f"[mine] metadata for {len(metas)} papers; quota {quota}", flush=True)

    deadline = time.monotonic() + args.max_hours * 3600
    for n, paper in enumerate(metas, 1):
        if time.monotonic() > deadline or all(have[t] >= q for t, q in quota.items()):
            break
        # Easy papers (both URLs in HF metadata) are cheap; skip them once tier 1 is full.
        if paper["github"] and paper["project_page"] and have[1] >= quota.get(1, 0):
            continue
        try:
            row = label_paper(paper, arxiv, web, llm, args)
        except Exception as exc:  # noqa: BLE001 - one bad paper must not end a multi-hour job
            print(f"[mine] skipped {paper['arxiv_id']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if row is None or have[row["difficulty_tier"]] >= quota.get(row["difficulty_tier"], 0):
            continue
        have[row["difficulty_tier"]] += 1
        with args.output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if sum(have.values()) % 50 == 0:
            upload(args)
        if n % 50 == 0:
            print(f"[mine] {n}/{len(metas)} examined; per tier {dict(sorted(have.items()))}", flush=True)
    upload(args)
    print(f"[mine] done: per tier {dict(sorted(have.items()))}", flush=True)


def upload(args: argparse.Namespace) -> None:
    if not args.output_repo or not args.output.exists():
        return
    from huggingface_hub import HfApi

    try:
        HfApi().upload_file(path_or_fileobj=str(args.output), path_in_repo=args.path_in_repo,
                            repo_id=args.output_repo, repo_type="dataset", commit_message="Mined rows (checkpoint)")  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - keep mining; the next checkpoint retries
        print(f"[mine] upload failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
