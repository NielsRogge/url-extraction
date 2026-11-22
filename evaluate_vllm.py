#!/usr/bin/env python3
# /// script
# dependencies = [
#   "datasets>=4.0.0",
#   "httpx>=0.27.0",
#   "huggingface-hub>=1.13.0",
#   "opencode-ai==0.1.0a36",
#   "pydantic>=2.10.0,<2.13",
#   "python-dotenv>=1.1.0",
#   "safetensors>=0.4.0",
#   "vllm>=0.22.0",
# ]
# ///
"""Evaluate a student (base model and optionally a LoRA adapter) in OpenCode through a local vLLM server.

Serves the base model and adapter together, runs both on the same fully scoreable held-out papers of any
split (`test`, `validation`, `test_hard`, ...), persists each episode as JSONL, and publishes the records,
a summary and agent-trace sessions to a Hub dataset. The task, the episode loop and the scorer come from
the shared `harness` package, so these scores are computed exactly as the GRPO reward and teacher-trace
acceptance are. Model-specific serving flags come from the student profile via `pipeline.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from datasets import load_dataset  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from opencode_ai import AsyncOpencode  # noqa: E402

# Re-exported for scripts and tests that import the harness through the evaluator.
from harness.opencode import (  # noqa: E402,F401
    ALLOWED_TOOLS,
    MAX_PORTABLE_TOOL_OUTPUT_CHARS,
    OPENCODE_TOOLS,
    OPENCODE_VERSION,
    findings_from_messages,
    findings_from_text,
    install_opencode,
    opencode_server,
    portable_trajectory,
    harness_failure,
    raw_json,
    record_line,
    tool_events,
)
from harness.opencode import run_one as _run_one  # noqa: E402
from harness.report import mcnemar_exact, summarize  # noqa: E402,F401
from harness.scoring import (  # noqa: E402,F401
    SCOREABLE_QUALITIES,
    canonical_url,
    derive_attempt_complete,
    fully_scoreable,
    partly_scoreable,
    score_findings,
    score_missing_findings,
)
from harness.task import (  # noqa: E402,F401
    SYSTEM_PROMPT,
    PaperCandidate,
    ToolEvent,
    URLFindings,
    candidate_from_row,
    user_prompt,
)

DATASET_ID = "nielsr/paper-url-extraction-v1"
ADAPTER_ID = "nielsr/qwen35-4b-url-agent-lora"
BASE_MODEL = "Qwen/Qwen3.5-4B"


async def run_one(client: Any, row: dict[str, Any], model: tuple[str, str], *, episode_timeout: float,
                  max_tool_calls: int) -> dict[str, Any]:  # fmt: skip
    return await _run_one(
        client, row, model, episode_timeout=episode_timeout, max_tool_calls=max_tool_calls, title="URL evaluation"
    )


def wait_for_server(process: subprocess.Popen, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy within the startup timeout")


def start_vllm(args: argparse.Namespace, adapter_path: str) -> subprocess.Popen:
    command = [
        shutil.which(args.vllm_binary) or args.vllm_binary,
        "serve",
        args.base_model,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        args.tool_call_parser,
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.language_model_only:
        command.append("--language-model-only")
    if args.reasoning_parser:
        command += ["--reasoning-parser", args.reasoning_parser]
    if adapter_path:
        command += ["--enable-lora", "--max-lora-rank", str(args.max_lora_rank), "--lora-modules", f"url-agent={adapter_path}"]
    command += shlex.split(args.vllm_extra_args)
    what = "base model and URL-agent LoRA" if adapter_path else "base model only"
    print(f"Starting vLLM ({what}): {shlex.join(command)}", flush=True)
    return subprocess.Popen(command, start_new_session=True)


def prepare_text_adapter(source: str, destination: Path, converter: str = "") -> str:
    """Make a trained adapter servable by text-only vLLM.

    With `converter` (a student profile's `adapter_converter`, `module:function(source, destination)`), that
    function rewrites the adapter (e.g. renaming modules to the checkpoint names vLLM expects). Otherwise
    inert vision-tower tensors, which text-only vLLM rejects, are removed.
    """
    from safetensors.torch import load_file, save_file

    source_path = Path(source)
    if converter:
        from harness.hooks import resolve

        return str(resolve(converter)(source_path, destination))
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_path, destination, ignore=shutil.ignore_patterns("adapter_model.safetensors"))
    tensors = load_file(source_path / "adapter_model.safetensors", device="cpu")
    text_tensors = {name: tensor for name, tensor in tensors.items() if ".visual." not in name}
    removed = set(tensors) - set(text_tensors)
    if not text_tensors:
        raise RuntimeError("Expected text LoRA tensors in the trained adapter")
    if not removed:
        # Adapters trained on the language model only (e.g. the GRPO stage) need no filtering.
        print(f"Adapter has {len(text_tensors)} text-only tensors; no visual tensors to remove", flush=True)
    nonzero_visual_b = [name for name in removed if ".lora_B." in name and bool(tensors[name].count_nonzero().item())]
    if nonzero_visual_b:
        raise RuntimeError("Refusing to remove trained visual LoRA tensors")
    save_file(text_tensors, destination / "adapter_model.safetensors")
    config_path = destination / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["target_modules"] = sorted({name.split(".")[-3] for name in text_tensors})
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(
        f"Prepared text-only adapter with {len(text_tensors)} tensors; removed {len(removed)} inert visual tensors",
        flush=True,
    )
    return str(destination)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def configure_opencode(args: argparse.Namespace) -> None:
    model_limit = {"context": args.max_model_len, "output": args.max_output_tokens}
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "local-vllm": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Local vLLM",
                    "options": {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "EMPTY"},
                    "models": {
                        args.base_model: {"name": f"{args.base_model} base", "limit": model_limit},
                        "url-agent": {"name": f"{args.base_model} URL-agent LoRA", "limit": model_limit},
                    },
                }
            },
        }
    )


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            with suppress(ValueError, KeyError, json.JSONDecodeError):
                record = json.loads(line)
                keys.add((str(record["evaluation_model"]), int(record["paper_id"])))
    return keys


async def evaluate(args: argparse.Namespace, output_path: Path) -> None:
    keep = partly_scoreable if args.include_partial else fully_scoreable
    rows = [dict(row) for row in load_dataset(args.dataset, split=args.split) if keep(dict(row))]
    if args.paper_ids:
        requested_ids = {int(value) for value in args.paper_ids.split(",")}
        rows = [row for row in rows if int(row["paper_id"]) in requested_ids]
        found_ids = {int(row["paper_id"]) for row in rows}
        if missing_ids := requested_ids - found_ids:
            raise ValueError(f"Requested paper IDs were not scoreable in {args.split}: {sorted(missing_ids)}")
    else:
        rows = rows[args.start : args.start + args.limit]
    if not rows:
        raise ValueError(f"No scoreable rows selected from {args.dataset}:{args.split}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = [("base", args.base_model), ("adapter", "url-agent")] if args.adapter else [("base", args.base_model)]
    if args.adapter and args.adapter_only:
        models = [("adapter", "url-agent")]

    for label, model_id in models:
        for restart in range(args.max_harness_restarts + 1):
            pending = [row for row in rows if (label, int(row["paper_id"])) not in completed_keys(output_path)]
            if not pending:
                break
            if restart:
                print(f"[{label}] restarting OpenCode after a harness failure ({len(pending)} papers left)", flush=True)
            if not await run_model(args, label, model_id, pending, output_path):
                break
        else:
            raise RuntimeError(f"OpenCode kept failing for {label} after {args.max_harness_restarts} restarts")


async def run_model(
    args: argparse.Namespace, label: str, model_id: str, pending: list[dict[str, Any]], output_path: Path
) -> bool:
    """Run `pending` papers on one fresh OpenCode server. Returns True when the server broke.

    An episode with a harness failure (see `harness.opencode.harness_failure`) is not recorded and stops the
    remaining queue, so the caller can retry those papers on a new server instead of scoring the breakage.
    """
    broken = asyncio.Event()
    async with (
        # The student runs on local vLLM and needs no token; keep the job's HF_TOKEN out of the agent's shell.
        opencode_server(str(args.opencode_binary), strip_credentials=True) as base_url,
        AsyncOpencode(
            base_url=base_url, timeout=httpx.Timeout(args.request_timeout, connect=10), max_retries=1
        ) as client,
    ):
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_row(index: int, row: dict[str, Any]) -> None:
            async with semaphore:
                if broken.is_set():
                    return
                print(f"[{label} {index}/{len(pending)}] {row['arxiv_id']}", flush=True)
                record = await run_one(
                    client,
                    row,
                    ("local-vllm", model_id),
                    episode_timeout=args.episode_timeout,
                    max_tool_calls=args.max_tool_calls,
                )
                if failure := harness_failure(record):
                    print(f"  harness failure, not recorded: {failure}", flush=True)
                    broken.set()
                    return
                record["evaluation_model"] = label
                record["difficulty_tier"] = row.get("difficulty_tier")
                record["served_model"] = model_id
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(record_line(record))
                    handle.flush()
                print(
                    f"  complete={record['attempt_complete']} correct={record['verification']['all_scoreable_correct']} "
                    f"tools={len(record['tool_events'])}",
                    flush=True,
                )

        await asyncio.gather(*(run_row(i, row) for i, row in enumerate(pending, 1)))
        if not broken.is_set():
            # Explicitly verify that the OpenCode service remained responsive through the final episode.
            await raw_json(client, "get", "/global/health")
    return broken.is_set()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def publish(args: argparse.Namespace, output_path: Path, summary_path: Path) -> None:
    if not args.output_repo:
        print(f"Results kept at {output_path} and {summary_path}; no output repo", flush=True)
        return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.output_repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo=f"{args.tag}{args.split}.jsonl",
        repo_id=args.output_repo,
        repo_type="dataset",
        commit_message=f"Add {args.split} OpenCode evaluation episodes",
    )
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo=f"{args.tag}{args.split}-summary.json",
        repo_id=args.output_repo,
        repo_type="dataset",
        commit_message=f"Add {args.split} evaluation summary",
    )
    from trace_export import publish_sessions  # required: evaluation episodes are published as viewer sessions

    count = publish_sessions(api, args.output_repo, output_path, f"{args.tag}{args.split}.jsonl")
    print(f"published {count} agent-trace sessions to {args.output_repo}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--paper-ids", help="Comma-separated paper IDs; overrides --start and --limit")
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Also evaluate papers where only some requested fields have verified labels",
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER_ID, help="LoRA adapter repo or dir; '' evaluates the base model only")
    parser.add_argument("--adapter-only", action="store_true", help="Skip the base model (already evaluated)")
    parser.add_argument("--adapter-converter", default="", help="module:function preparing the adapter for vLLM")
    parser.add_argument("--output", type=Path, default=Path("/artifacts/qwen35-4b-url-agent-eval/validation.jsonl"))
    parser.add_argument("--output-repo", default="nielsr/qwen35-4b-url-agent-eval")
    parser.add_argument("--tag", default="", help="Prefix for published file names, e.g. 'sft-v2/'")
    parser.add_argument("--public", action="store_true", help="Create --output-repo as public")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--episode-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-harness-restarts", type=int, default=3, help="Fresh OpenCode servers after harness failures")
    # Serving flags; `pipeline.py` fills them from the student profile.
    parser.add_argument("--vllm-binary", default="vllm", help="vLLM executable, e.g. the one in a model-specific image")
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument("--reasoning-parser", default="", help="e.g. 'inkling'; empty disables reasoning parsing")
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-extra-args", default="", help="Extra arguments appended to `vllm serve` (shell-quoted)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be passed as a Job secret")
    workdir = Path(tempfile.mkdtemp(prefix="url-agent-eval-"))
    args.opencode_binary = install_opencode(workdir / "bin" / "opencode")
    os.environ["PATH"] = str(args.opencode_binary.parent) + os.pathsep + os.environ["PATH"]
    configure_opencode(args)
    adapter_path = ""
    if args.adapter:
        adapter_snapshot = (
            args.adapter if Path(args.adapter).is_dir() else snapshot_download(args.adapter, token=os.environ["HF_TOKEN"])
        )
        adapter_path = prepare_text_adapter(adapter_snapshot, workdir / "text-adapter", args.adapter_converter)
    server = start_vllm(args, adapter_path)
    try:
        wait_for_server(server)
        asyncio.run(evaluate(args, args.output))
        records = load_records(args.output)
        summary = summarize(records)
        summary_path = args.output.with_name(f"{args.split}-summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        publish(args, args.output, summary_path)
    finally:
        stop_process(server)


if __name__ == "__main__":
    main(parse_args())
