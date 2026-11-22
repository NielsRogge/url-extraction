#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "datasets>=4.0.0",
#   "httpx>=0.27.0",
#   "huggingface-hub>=1.13.0",
#   "opencode-ai==0.1.0a36",
#   "pydantic>=2.10.0,<2.13",
#   "python-dotenv>=1.1.0",
# ]
# ///
"""Generate verified URL-research trajectories with the OpenCode Python SDK.

Run an existing server with ``--base-url`` or let this script launch one in an
isolated temporary workspace. Each JSONL record contains the untouched
OpenCode messages, a portable chat trajectory, tool events, and label-aware
verification results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import get_token

from harness.opencode import (  # noqa: F401 - re-exported for tests and older imports
    ALLOWED_TOOLS,
    OPENCODE_TOOLS,
    OPENCODE_VERSION,
    SUPPORTED_OPENCODE_VERSIONS,
    find_or_install_opencode,
    findings_from_messages,
    findings_from_text,
    opencode_client,
    opencode_server,
    portable_trajectory,
    raw_json,
    record_line,
    tool_events,
)
from harness.opencode import run_one as run_episode
from harness.scoring import SCOREABLE_QUALITIES, fully_scoreable, partly_scoreable
from harness.task import candidate_from_row  # noqa: F401
from trace_export import publish_sessions, write_sessions

DEFAULT_DATASET = "nielsr/paper-url-extraction-v1"
DEFAULT_MODEL = "huggingface/zai-org/GLM-5.3"
SCORED_QUALITIES = SCOREABLE_QUALITIES


def split_model(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError("model must be PROVIDER_ID/MODEL_ID")
    return tuple(value.split("/", 1))  # type: ignore[return-value]


def validate_huggingface_token(model: tuple[str, str], bill_to: str | None = None) -> None:
    """Fail early when a fine-grained token cannot invoke the selected billing account."""
    if model[0] != "huggingface":
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the OpenCode Hugging Face provider")
    response = httpx.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    access = (payload.get("auth") or {}).get("accessToken") or {}
    if access.get("role") != "fineGrained":
        return
    fine_grained = access.get("fineGrained") or {}
    required = "inference.serverless.write"
    if required in (fine_grained.get("global") or []):
        return
    username = str(payload.get("name"))
    entity_type = "org" if bill_to else "user"
    entity_name = bill_to or username
    owner_permissions = {
        permission
        for scope in fine_grained.get("scoped") or []
        if (scope.get("entity") or {}).get("type") == entity_type
        and (scope.get("entity") or {}).get("name") == entity_name
        for permission in scope.get("permissions") or []
    }
    if required not in owner_permissions:
        target = f"organization {bill_to}" if bill_to else f"user {username}"
        raise RuntimeError(
            f"HF_TOKEN is valid for {username}, but lacks '{required}' for {target}; "
            "create a token with permission to make calls to Inference Providers"
        )


def configure_huggingface_billing(model: tuple[str, str], bill_to: str | None) -> None:
    """Add HF organization billing to OpenCode's inline provider configuration."""
    if model[0] != "huggingface" or not bill_to:
        return
    try:
        config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("OPENCODE_CONFIG_CONTENT must contain valid JSON") from exc
    headers = (
        config.setdefault("provider", {})
        .setdefault("huggingface", {})
        .setdefault("options", {})
        .setdefault("headers", {})
    )
    headers["X-HF-Bill-To"] = bill_to
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)


def cached_hf_cli_token() -> str | None:
    """Read the active cached token while ignoring token environment variables."""
    keys = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        return get_token()
    finally:
        os.environ.update(saved)


def row_all_requested_scoreable(row: dict[str, Any]) -> bool:
    return fully_scoreable(row)


def row_any_requested_scoreable(row: dict[str, Any]) -> bool:
    return partly_scoreable(row)


# Set from --accept-partial: also accept rows where only some requested fields have verified labels.
ACCEPT_PARTIAL = False


def configure_model_entry(model: tuple[str, str], entry: dict[str, Any] | None) -> None:
    """Declare `model` in OpenCode's inline config, e.g. a provider-pinned id such as `org/model:deepinfra`."""
    if not entry:
        return
    config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{}") or "{}")
    models = config.setdefault("provider", {}).setdefault(model[0], {}).setdefault("models", {})
    models[model[1]] = {**models.get(model[1], {}), **entry}
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)


async def run_one(
    client: Any,
    row: dict[str, Any],
    model: tuple[str, str],
    *,
    episode_timeout: float,
    max_tool_calls: int,
) -> dict[str, Any]:
    """One teacher episode, scored by the shared harness; `accepted` decides whether it is an SFT example."""
    record = await run_episode(
        client, row, model, episode_timeout=episode_timeout, max_tool_calls=max_tool_calls, title="URL trace"
    )
    all_requested_scoreable = row_all_requested_scoreable(row)
    label_ok = all_requested_scoreable or (ACCEPT_PARTIAL and record["verification"]["num_scoreable"] > 0)
    record["accepted"] = bool(record["accepted"] and label_ok)
    record["all_requested_scoreable"] = all_requested_scoreable
    return record


def accepted_ids(path: Path, *, any_attempt: bool = False) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if any_attempt or record.get("accepted"):
                    ids.add(int(record["paper_id"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return ids


async def async_main(args: argparse.Namespace) -> None:
    model = split_model(args.model)
    validate_huggingface_token(model, args.hf_bill_to)
    configure_huggingface_billing(model, args.hf_bill_to)
    configure_model_entry(model, json.loads(args.opencode_model_entry) if args.opencode_model_entry else None)
    opencode_command = (
        args.opencode_command if args.base_url else find_or_install_opencode(args.opencode_command, args.opencode_version)
    )
    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(row) for row in dataset]
    global ACCEPT_PARTIAL
    ACCEPT_PARTIAL = args.accept_partial
    if args.accept_partial:
        rows = [row for row in rows if row_any_requested_scoreable(row)]
    elif not args.include_unscoreable:
        rows = [row for row in rows if row_all_requested_scoreable(row)]
    rows = rows[args.start : args.start + args.limit]
    if args.shard:  # disjoint slices for parallel workers / jobs
        index, count = (int(value) for value in args.shard.split("/"))
        rows = rows[index::count]
    done = accepted_ids(args.output, any_attempt=args.resume_attempted) if args.resume else set()
    rows = [row for row in rows if int(row["paper_id"]) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with (
        opencode_server(opencode_command, base_url=args.base_url, version=args.opencode_version) as server,
        opencode_client(server, timeout=httpx.Timeout(args.timeout, connect=10)) as client,
    ):
        semaphore = asyncio.Semaphore(args.concurrency)

        async def process_row(number: int, row: dict[str, Any]) -> None:
            async with semaphore:
                print(f"[{number}/{len(rows)}] {row['arxiv_id']}", flush=True)
                for attempt in range(1, args.attempts_per_row + 1):
                    record = await run_one(
                        client,
                        row,
                        model,
                        episode_timeout=args.episode_timeout,
                        max_tool_calls=args.max_tool_calls,
                    )
                    record["attempt"] = attempt
                    if args.evaluation_label:  # e.g. "teacher": a teacher baseline in the evaluator's summary
                        # Same columns as evaluate_vllm.py records, so one dataset config can hold both.
                        for key in ("attempt", "all_requested_scoreable"):
                            record.pop(key, None)
                        record["evaluation_model"] = args.evaluation_label
                        record["difficulty_tier"] = row.get("difficulty_tier")
                        record["served_model"] = args.model
                    with args.output.open("a", encoding="utf-8") as handle:
                        handle.write(record_line(record))
                        handle.flush()
                    if args.sessions_dir:  # one Pi-format session per episode, for the Hub's agent-traces viewer
                        write_sessions([record], args.sessions_dir)
                    print(
                        f"  {row['arxiv_id']} attempt={attempt} accepted={record['accepted']} "
                        f"complete={record['attempt_complete']} score={record['verification']['fields']}",
                        flush=True,
                    )
                    if record["accepted"]:
                        break

        results = await asyncio.gather(
            *(process_row(number, row) for number, row in enumerate(rows, 1)),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(f"{len(errors)} row workers failed; rerun with resume enabled") from errors[0]
    if args.output_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.output_repo, repo_type="dataset", private=not args.public, exist_ok=True)
        path_in_repo = args.path_in_repo or args.output.name
        api.upload_file(path_or_fileobj=str(args.output), path_in_repo=path_in_repo, repo_id=args.output_repo,
                        repo_type="dataset", commit_message=f"Add {path_in_repo}")  # fmt: skip
        count = publish_sessions(api, args.output_repo, args.output, path_in_repo)
        print(f"published {path_in_repo} and {count} agent-trace sessions to {args.output_repo}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--shard", help="i/n: keep every n-th selected row starting at i (parallel workers)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/traces/train.jsonl"))
    parser.add_argument("--model", help=f"OpenCode PROVIDER/MODEL (default: {DEFAULT_MODEL})")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Load provider credentials from this file")
    parser.add_argument(
        "--use-hf-cli-token",
        action="store_true",
        help="Use the cached `hf auth login` token in memory instead of HF_TOKEN from .env",
    )
    parser.add_argument(
        "--hf-bill-to",
        help="HF organization to bill (or set HF_BILL_TO; default: huggingface)",
    )
    parser.add_argument("--base-url", help="Attach to an existing `opencode serve` instance")
    parser.add_argument(
        "--opencode-command", default="opencode", help="Used if it is --opencode-version, else the pinned release is installed"
    )
    parser.add_argument("--opencode-version", default=OPENCODE_VERSION, choices=SUPPORTED_OPENCODE_VERSIONS,
                        help="OpenCode release to run (recorded in every episode record)")
    parser.add_argument(
        "--opencode-model-entry",
        help="JSON model entry added to OpenCode's inline config for --model (needed for provider-pinned ids)",
    )
    parser.add_argument("--evaluation-label", help="Store as evaluation_model (e.g. 'teacher' for a baseline run)")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--episode-timeout", type=float, default=180)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--attempts-per-row", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1, help="Number of simultaneous OpenCode sessions")
    parser.add_argument(
        "--include-unscoreable",
        action="store_true",
        help="Also research rows with unknown requested labels (never accepted for SFT)",
    )
    parser.add_argument(
        "--accept-partial",
        action="store_true",
        help="Research rows with at least one verified requested label; accept when every verified field matches",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        help="Also write each episode as a Pi session file here (default: <output stem>.sessions/; '' disables)",
    )
    parser.add_argument("--output-repo", help="Upload the records and their agent-trace sessions to this dataset repo")
    parser.add_argument("--path-in-repo", help="Records path in --output-repo (default: the --output file name)")
    parser.add_argument("--public", action="store_true", help="Create --output-repo as public")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--resume-attempted",
        action="store_true",
        help="On resume, skip every paper already in --output, not only accepted ones (avoids re-paying for rejects)",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    if args.use_hf_cli_token:
        token = cached_hf_cli_token()
        if not token:
            parser.error("--use-hf-cli-token requires an authenticated `hf auth login` session")
        os.environ["HF_TOKEN"] = token
    args.hf_bill_to = args.hf_bill_to or os.environ.get("HF_BILL_TO") or "huggingface"
    args.model = args.model or os.environ.get("OPENCODE_TEACHER_MODEL", DEFAULT_MODEL)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.sessions_dir is None:
        args.sessions_dir = args.output.with_suffix(".sessions")
    elif str(args.sessions_dir) in {"", "."}:
        args.sessions_dir = None
    asyncio.run(async_main(args))
