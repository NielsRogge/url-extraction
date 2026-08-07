"""Run a resumable URL-extraction batch through OpenCode and a managed Endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from huggingface_hub import HfApi

from .opencode_endpoint import DEFAULT_MODEL_ID, DEFAULT_PROVIDER_ID, run_candidates
from .schema import AgentAttempt, PaperCandidate

CANDIDATE_FIELDS = set(PaperCandidate.model_fields)


def canonical_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"https://{text}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    host = parsed.netloc.lower().removeprefix("www.")
    path = "/".join(part for part in parsed.path.split("/") if part)
    if host == "github.com":
        parts = path.split("/")
        if len(parts) >= 2:
            path = "/".join(parts[:2])
    path = f"/{path.lower()}" if path else ""
    return urlunparse(("https", host, path, "", "", ""))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def score_attempt(source: dict[str, Any], attempt: AgentAttempt) -> dict[str, bool | None]:
    scores: dict[str, bool | None] = {}
    for name in ("github", "project_page"):
        if source.get(f"{name}_label_quality") != "consensus":
            scores[name] = None
            continue
        predicted = canonical_url(getattr(attempt.findings, f"{name}_url", None)) if attempt.findings else None
        expected = canonical_url(source.get(f"reference_{name}_url"))
        scores[name] = predicted == expected
    return scores


def result_record(source: dict[str, Any], attempt: AgentAttempt) -> dict[str, Any]:
    scores = score_attempt(source, attempt)
    scoreable = [score for score in scores.values() if score is not None]
    return {
        "arxiv_id": attempt.arxiv_id,
        "paper_id": attempt.paper_id,
        "attempt": attempt.model_dump(mode="json"),
        "references": {
            "github_url": source.get("reference_github_url"),
            "project_page_url": source.get("reference_project_page_url"),
        },
        "trace_consensus": source.get("trace_consensus"),
        "field_scores": scores,
        "exact_match": bool(scoreable) and all(scoreable),
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        name: [row["field_scores"][name] for row in rows if row["field_scores"].get(name) is not None]
        for name in ("github", "project_page")
    }
    return {
        "episodes": len(rows),
        "complete_episodes": sum(bool(row["attempt"]["attempt_complete"]) for row in rows),
        "exact_match_episodes": sum(bool(row["exact_match"]) for row in rows),
        "fields": {name: {"correct": sum(values), "total": len(values)} for name, values in fields.items()},
        "failure_reasons": dict(
            Counter(row["attempt"].get("failure_reason") for row in rows if row["attempt"].get("failure_reason"))
        ),
    }


def publish(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if not args.output_repo:
        return
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.output_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(args.output),
        path_in_repo=args.repo_results_path,
        repo_id=args.output_repo,
        repo_type="dataset",
        commit_message="Publish URL-agent canary results",
    )
    api.upload_file(
        path_or_fileobj=(json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
        path_in_repo=args.repo_summary_path,
        repo_id=args.output_repo,
        repo_type="dataset",
        commit_message="Publish URL-agent canary summary",
    )


async def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_jsonl(args.input)
    prior = {row["arxiv_id"]: row for row in load_jsonl(args.output)}
    pending_rows = [
        row
        for row in inputs
        if row["arxiv_id"] not in prior
        or (args.retry_incomplete and not prior[row["arxiv_id"]]["attempt"]["attempt_complete"])
    ]
    candidates = [PaperCandidate.model_validate({key: row[key] for key in CANDIDATE_FIELDS}) for row in pending_rows]
    source_by_arxiv = {row["arxiv_id"]: row for row in inputs}
    lock = asyncio.Lock()

    async def checkpoint(attempt: AgentAttempt) -> None:
        async with lock:
            prior[attempt.arxiv_id] = result_record(source_by_arxiv[attempt.arxiv_id], attempt)
            ordered = [prior[row["arxiv_id"]] for row in inputs if row["arxiv_id"] in prior]
            write_jsonl_atomic(args.output, ordered)
            print(
                f"checkpoint {len(ordered)}/{len(inputs)} {attempt.arxiv_id} complete={attempt.attempt_complete}",
                flush=True,
            )

    if candidates:
        await run_candidates(args, candidates, checkpoint)
    results = [prior[row["arxiv_id"]] for row in inputs if row["arxiv_id"] in prior]
    summary = summarize(results)
    publish(args, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-url", default=os.environ.get("HF_ENDPOINT_URL"))
    parser.add_argument("--provider-id", default=DEFAULT_PROVIDER_ID)
    parser.add_argument("--model-id", default=os.environ.get("HF_ENDPOINT_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--opencode-binary", default="opencode")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--episode-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--output-repo")
    parser.add_argument("--repo-results-path", default="canary-results.jsonl")
    parser.add_argument("--repo-summary-path", default="canary-summary.json")
    args = parser.parse_args()
    if not args.endpoint_url:
        parser.error("set HF_ENDPOINT_URL or pass --endpoint-url")
    if not os.environ.get("HF_TOKEN"):
        parser.error("set HF_TOKEN; it is used at runtime and never persisted")
    return args


def main() -> None:
    load_dotenv()
    args = parse_args()
    print(json.dumps(asyncio.run(run_batch(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
