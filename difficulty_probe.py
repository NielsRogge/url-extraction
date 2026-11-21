#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "datasets>=4.0.0",
#   "httpx>=0.27.0",
#   "huggingface-hub>=1.13.0",
#   "opencode-ai==0.1.0a36",
#   "pydantic>=2.10.0,<2.13",
# ]
# ///
"""Difficulty check with the policy itself, before GRPO.

GRPO learns only from groups whose rollouts disagree: a paper the policy always solves, or never
solves, contributes nothing. Following the harness-adapters recipe, the starting policy attempts every
training paper K times under the exact RL conditions (same OpenCode harness, prompt, tool budget and
sampling temperature), each attempt is scored with the RL success criterion (every verified field
correct and the research protocol followed), and papers solved on some but not all attempts form the
qualified set used as the GRPO prompt pool.

vLLM runs from its own `uvx` environment (as in train_grpo_harness.py). Results are written as one
JSONL record per attempt plus a per-paper summary and the qualified ID list, and can be published.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset
from opencode_ai import AsyncOpencode

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_vllm import SCOREABLE_QUALITIES, install_opencode, opencode_server, run_one  # noqa: E402

VLLM_VERSION = "0.33.0"
SERVED = "policy"


def scoreable(row: dict[str, Any]) -> bool:
    return any(row[f"missing_{n}"] and row.get(f"{n}_label_quality") in SCOREABLE_QUALITIES for n in ("github", "project_page"))


def start_vllm(model: str, args: argparse.Namespace) -> subprocess.Popen:
    command = [
        shutil.which("uvx") or "uvx", "--python", "3.12", "--from", f"vllm=={VLLM_VERSION}", "vllm", "serve", model,
        "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", "8000",
        "--max-model-len", str(args.max_model_len), "--gpu-memory-utilization", "0.9",
        "--language-model-only", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--max-num-seqs", str(max(8, args.concurrency)),
        # Same sampling as GRPO rollouts; OpenCode sends no temperature of its own.
        "--override-generation-config", json.dumps({"temperature": args.temperature, "top_p": 1.0, "top_k": -1}),
    ]  # fmt: skip
    print("[probe] " + " ".join(command), flush=True)
    return subprocess.Popen(command, start_new_session=True)


def wait_for(url: str, process: subprocess.Popen, timeout: float = 2400) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with {process.returncode}")
        with suppress(Exception), urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                return
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy")


def success(record: dict[str, Any]) -> bool:
    """The RL success criterion: every verified field right and the research protocol followed."""
    return bool(record["verification"].get("all_scoreable_correct")) and bool(record.get("attempt_complete"))


async def probe(rows: list[dict[str, Any]], args: argparse.Namespace, out: Path) -> None:
    done = collections.Counter()
    if out.exists():
        for line in out.read_text().splitlines():
            with suppress(Exception):
                done[json.loads(line)["arxiv_id"]] += 1
    jobs = [(row, k) for row in rows for k in range(args.k) if done[str(row["arxiv_id"])] <= k]
    print(f"[probe] {len(rows)} papers x {args.k} attempts; {len(jobs)} episodes to run", flush=True)
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "local-vllm": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Local vLLM",
                    "options": {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "EMPTY"},
                    "models": {SERVED: {"name": "policy", "limit": {"context": args.max_model_len, "output": 1024}}},
                }
            },
        }
    )
    sem = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    finished = 0
    async with opencode_server(str(args.opencode_binary)) as base_url, AsyncOpencode(
        base_url=base_url, timeout=httpx.Timeout(args.episode_timeout + 120, connect=10), max_retries=1
    ) as client:

        async def one(row: dict[str, Any], attempt: int) -> None:
            nonlocal finished
            async with sem:
                record = await run_one(
                    client, row, ("local-vllm", SERVED),
                    episode_timeout=args.episode_timeout, max_tool_calls=args.max_tool_calls,
                )  # fmt: skip
            slim = {
                "arxiv_id": str(row["arxiv_id"]),
                "attempt": attempt,
                "success": success(record),
                "all_scoreable_correct": record["verification"].get("all_scoreable_correct"),
                "attempt_complete": record.get("attempt_complete"),
                "failure_reason": record.get("failure_reason"),
                "tool_calls": len(record.get("tool_events") or []),
                "elapsed_seconds": record.get("elapsed_seconds"),
            }
            with out.open("a") as handle:
                handle.write(json.dumps(slim) + "\n")
            finished += 1
            if finished % 100 == 0:
                rate = finished / (time.monotonic() - started)
                print(f"[probe] {finished}/{len(jobs)} episodes, {rate * 60:.0f}/min", flush=True)

        await asyncio.gather(*(one(row, k) for row, k in jobs))


def summarize(out: Path, k: int, summary_path: Path, qualified_path: Path) -> dict[str, Any]:
    per_paper: dict[str, list[bool]] = collections.defaultdict(list)
    for line in out.read_text().splitlines():
        record = json.loads(line)
        per_paper[record["arxiv_id"]].append(record["success"])
    histogram = collections.Counter(sum(v[:k]) for v in per_paper.values() if len(v) >= k)
    qualified = sorted(p for p, v in per_paper.items() if len(v) >= k and 0 < sum(v[:k]) < k)
    summary = {
        "papers": len(per_paper),
        "attempts_per_paper": k,
        "successes_histogram": {str(s): histogram.get(s, 0) for s in range(k + 1)},
        "qualified": len(qualified),
        "mean_success_rate": round(sum(sum(v) for v in per_paper.values()) / max(1, sum(len(v) for v in per_paper.values())), 4),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    qualified_path.write_text("\n".join(qualified) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Merged checkpoint (path or repo) of the starting policy")
    parser.add_argument("--dataset", default="nielsr/paper-url-extraction-v1-validated")
    parser.add_argument("--split", default="train")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--episode-timeout", type=float, default=180)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--output-dir", type=Path, default=Path("difficulty"))
    parser.add_argument("--output-repo", help="Dataset repo to upload results to (under difficulty/)")
    args = parser.parse_args()

    rows = [dict(r) for r in load_dataset(args.dataset, split=args.split) if scoreable(dict(r))]
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.opencode_binary = shutil.which("opencode") or install_opencode(Path(tempfile.mkdtemp()) / "opencode")
    os.environ["PATH"] = str(Path(args.opencode_binary).parent) + os.pathsep + os.environ["PATH"]
    vllm = start_vllm(args.model, args)
    try:
        wait_for("http://127.0.0.1:8000/health", vllm)
        asyncio.run(probe(rows, args, args.output_dir / "attempts.jsonl"))
    finally:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(vllm.pid), 15)
    summarize(args.output_dir / "attempts.jsonl", args.k, args.output_dir / "summary.json", args.output_dir / "qualified-ids.txt")
    if args.output_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=str(args.output_dir), path_in_repo="difficulty", repo_id=args.output_repo, repo_type="dataset",
            commit_message=f"Difficulty probe: {args.k} attempts per {args.split} paper",
        )  # fmt: skip
        print(f"[probe] published to {args.output_repo}/difficulty", flush=True)


if __name__ == "__main__":
    main()
