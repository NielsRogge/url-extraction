#!/usr/bin/env python3
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
import re
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import get_token
from opencode_ai import AsyncOpencode
from pydantic import ValidationError

from trace_export import publish_sessions, write_sessions
from url_extraction_agent.research import (
    SYSTEM_PROMPT,
    derive_attempt_complete,
    findings_json_schema,
    json_text,
    score_findings,
    user_prompt,
)
from url_extraction_agent.schema import PaperCandidate, ToolEvent, URLFindings

# Credentials an agent can print from its environment (e.g. `env | grep HF`); never written to records.
CREDENTIAL_RE = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,})")


def redact_credentials(text: str) -> str:
    return CREDENTIAL_RE.sub(lambda match: match.group(0)[:3] + "***REDACTED***", text)

DEFAULT_DATASET = "nielsr/paper-url-extraction-v1"
DEFAULT_MODEL = "huggingface/zai-org/GLM-5.3"
ALLOWED_TOOLS = {"bash", "read", "grep", "glob", "webfetch"}
OPENCODE_TOOLS = {
    "invalid",
    "question",
    "bash",
    "read",
    "glob",
    "grep",
    "edit",
    "write",
    "task",
    "webfetch",
    "todowrite",
    "websearch",
    "codesearch",
    "skill",
    "apply_patch",
}
SCORED_QUALITIES = {"gold", "silver", "verified_null"}
MAX_PORTABLE_TOOL_OUTPUT_CHARS = 8_000


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


def candidate_from_row(row: dict[str, Any]) -> PaperCandidate:
    return PaperCandidate(
        paper_id=int(row["paper_id"]),
        arxiv_id=str(row["arxiv_id"]),
        title=str(row["title"]),
        abstract=str(row["abstract"]),
        missing_github=bool(row["missing_github"]),
        missing_project_page=bool(row["missing_project_page"]),
        known_github_url=row.get("known_github_url"),
        known_project_page_url=row.get("known_project_page_url"),
        priority=int(row.get("priority", 0)),
        priority_source=str(row.get("priority_source", "dataset")),
    )


def requested_qualities(row: dict[str, Any]) -> list[Any]:
    return [row.get(f"{name}_label_quality") for name in ("github", "project_page") if bool(row[f"missing_{name}"])]


def row_all_requested_scoreable(row: dict[str, Any]) -> bool:
    qualities = requested_qualities(row)
    return bool(qualities) and all(quality in SCORED_QUALITIES for quality in qualities)


def row_any_requested_scoreable(row: dict[str, Any]) -> bool:
    return any(quality in SCORED_QUALITIES for quality in requested_qualities(row))


# Set from --accept-partial: also accept rows where only some requested fields have verified labels.
ACCEPT_PARTIAL = False


def tool_events(messages: list[dict[str, Any]]) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    for message in messages:
        for part in message.get("parts", []):
            if part.get("type") != "tool":
                continue
            state = part.get("state") or {}
            status = state.get("status")
            events.append(
                ToolEvent(
                    tool_name=str(part.get("tool", "")),
                    tool_input=state.get("input") or {},
                    success=status == "completed",
                    error=state.get("error"),
                    response=state.get("output"),
                )
            )
    return events


def portable_trajectory(messages: list[dict[str, Any]], final: URLFindings | None) -> list[dict[str, Any]]:
    """Convert OpenCode parts into ordered OpenAI-style chat messages."""
    result: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in messages:
        info = message.get("info") or {}
        role = info.get("role")
        if role == "user":
            text = "\n".join(part.get("text", "") for part in message.get("parts", []) if part.get("type") == "text")
            if text:
                result.append({"role": "user", "content": text})
            continue
        if role != "assistant":
            continue
        text_parts: list[str] = []
        # The teacher's reasoning is kept as `reasoning_content` on the assistant message that follows it
        # (the field chat templates such as Inkling 2's and Qwen3.5's render as a thinking block).
        reasoning_parts: list[str] = []

        def assistant(content: str, **extra: Any) -> dict[str, Any]:
            entry: dict[str, Any] = {"role": "assistant", "content": content, **extra}
            if reasoning_parts:
                entry["reasoning_content"] = "\n\n".join(reasoning_parts)
                reasoning_parts.clear()
            return entry

        for part in message.get("parts", []):
            if part.get("type") == "reasoning" and str(part.get("text") or "").strip():
                reasoning_parts.append(str(part["text"]).strip())
            elif part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif part.get("type") == "tool":
                if text_parts:
                    result.append(assistant("\n".join(text_parts)))
                    text_parts.clear()
                state = part.get("state") or {}
                call_id = str(part.get("callID") or part.get("id"))
                result.append(
                    assistant(
                        "",
                        tool_calls=[
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": str(part.get("tool", "")),
                                    "arguments": json_text(state.get("input") or {}),
                                },
                            }
                        ],
                    )
                )
                output = str(state.get("output") if state.get("status") == "completed" else state.get("error") or "")
                if len(output) > MAX_PORTABLE_TOOL_OUTPUT_CHARS:
                    output = (
                        output[:MAX_PORTABLE_TOOL_OUTPUT_CHARS] + "\n[tool output truncated in portable trajectory]"
                    )
                result.append({"role": "tool", "tool_call_id": call_id, "content": output})
        if text_parts or reasoning_parts:
            result.append(assistant("\n".join(text_parts)))
    used_structured_output = any(
        call.get("function", {}).get("name", "").lower() == "structuredoutput"
        for message in result
        for call in message.get("tool_calls", [])
    )
    if final is not None and not used_structured_output:
        final_text = final.model_dump_json(exclude_none=False)
        existing_final = None
        if result and result[-1].get("role") == "assistant" and isinstance(result[-1].get("content"), str):
            existing_final = findings_from_text(result[-1]["content"])
        if existing_final != final:
            result.append({"role": "assistant", "content": final_text})
    return result


def findings_from_text(text: str) -> URLFindings | None:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return URLFindings.model_validate_json(candidate)
        except (ValueError, TypeError):
            pass
    return None


def findings_from_messages(messages: list[dict[str, Any]]) -> URLFindings | None:
    """Parse a JSON-only final answer when provider-native schema output is unavailable."""
    texts = [
        str(part.get("text", ""))
        for message in messages
        if (message.get("info") or {}).get("role") == "assistant"
        for part in message.get("parts", [])
        if part.get("type") == "text" and part.get("text")
    ]
    for text in reversed(texts):
        findings = findings_from_text(text)
        if findings is not None:
            return findings
    return None


async def raw_json(client: AsyncOpencode, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    request = getattr(client, method)
    kwargs: dict[str, Any] = {"cast_to": httpx.Response}
    if body is not None:
        kwargs["body"] = body
    response = await request(path, **kwargs)
    response.raise_for_status()
    return response.json()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@asynccontextmanager
async def opencode_server(base_url: str | None, command: str) -> AsyncIterator[str]:
    if base_url:
        yield base_url.rstrip("/")
        return
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError(f"OpenCode executable not found: {command}")
    port = free_port()
    workspace = tempfile.TemporaryDirectory(prefix="url-traces-")
    process = subprocess.Popen(  # noqa: ASYNC220 - startup is immediate; process remains asynchronous
        [executable, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
        cwd=workspace.name,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient() as health:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f"OpenCode exited during startup with status {process.returncode}")
                try:
                    response = await health.get(f"{url}/global/health", timeout=1)
                    if response.is_success:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
            else:
                raise TimeoutError("OpenCode did not become healthy within 10 seconds")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        workspace.cleanup()


async def run_one(
    client: AsyncOpencode,
    row: dict[str, Any],
    model: tuple[str, str],
    *,
    episode_timeout: float,
    max_tool_calls: int,
) -> dict[str, Any]:
    candidate = candidate_from_row(row)
    started = time.monotonic()
    session = await raw_json(
        client,
        "post",
        "/session",
        {
            "title": f"URL trace {candidate.arxiv_id}",
            "permission": [
                {"permission": "*", "pattern": "*", "action": "allow"},
                {"permission": "external_directory", "pattern": "*", "action": "deny"},
                {"permission": "doom_loop", "pattern": "*", "action": "deny"},
            ],
        },
    )
    session_id = session["id"]
    failure: str | None = None
    response: dict[str, Any] = {}
    watchdog_reason: str | None = None

    async def watchdog() -> None:
        nonlocal watchdog_reason
        while True:
            await asyncio.sleep(2)
            current = await raw_json(client, "get", f"/session/{session_id}/message")
            calls = sum(part.get("type") == "tool" for message in current for part in message.get("parts", []))
            if calls >= max_tool_calls:
                watchdog_reason = f"tool-call limit reached ({max_tool_calls})"
                await raw_json(client, "post", f"/session/{session_id}/abort")
                return

    watchdog_task = asyncio.create_task(watchdog())
    try:
        async with asyncio.timeout(episode_timeout):
            response = await raw_json(
                client,
                "post",
                f"/session/{session_id}/message",
                {
                    "model": {"providerID": model[0], "modelID": model[1]},
                    "system": SYSTEM_PROMPT,
                    "tools": {name: name in ALLOWED_TOOLS for name in OPENCODE_TOOLS},
                    "format": {"type": "json_schema", "schema": findings_json_schema(), "retryCount": 2},
                    "parts": [{"type": "text", "text": user_prompt(candidate)}],
                },
            )
            if (response.get("info") or {}).get("error"):
                response = await raw_json(
                    client,
                    "post",
                    f"/session/{session_id}/message",
                    {
                        "model": {"providerID": model[0], "modelID": model[1]},
                        "system": SYSTEM_PROMPT,
                        "tools": {name: name in ALLOWED_TOOLS for name in OPENCODE_TOOLS},
                        "parts": [
                            {
                                "type": "text",
                                "text": (
                                    "The provider rejected native structured output. Complete the research task above "
                                    "and return only one JSON object with reasoning, github_url, and project_page_url."
                                ),
                            }
                        ],
                    },
                )
    except TimeoutError:
        failure = f"episode timed out after {episode_timeout:g} seconds"
        with suppress(Exception):
            await raw_json(client, "post", f"/session/{session_id}/abort")
    except Exception as exc:  # noqa: BLE001 - preserve failed provider attempts for diagnosis and resumption
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task
    failure = failure or watchdog_reason
    messages = await raw_json(client, "get", f"/session/{session_id}/message")
    provider_errors = [
        ((message.get("info") or {}).get("error") or {}).get("data", {}).get("message") for message in messages
    ]
    provider_errors = [message for message in provider_errors if message]
    if provider_errors:
        failure = failure or f"provider error: {provider_errors[-1]}"
    events = tool_events(messages)
    findings: URLFindings | None = None
    structured = (response.get("info") or {}).get("structured")
    if structured is not None:
        try:
            findings = URLFindings.model_validate(structured)
        except ValidationError as exc:
            failure = failure or f"invalid structured output: {exc}"
    if findings is None:
        findings = findings_from_messages(messages)
    complete, incomplete_reason = (False, "no valid structured output")
    if findings is not None:
        complete, incomplete_reason = derive_attempt_complete(candidate, findings, events)
    if len(events) >= max_tool_calls:
        complete, incomplete_reason = False, f"tool-call limit reached ({max_tool_calls})"
    scoring = (
        score_findings(row, findings)
        if findings is not None
        else {"fields": {}, "num_scoreable": 0, "all_scoreable_correct": False}
    )
    all_requested_scoreable = row_all_requested_scoreable(row)
    label_ok = all_requested_scoreable or (ACCEPT_PARTIAL and scoring["num_scoreable"] > 0)
    accepted = findings is not None and complete and label_ok and scoring["all_scoreable_correct"]
    return {
        "version": 1,
        "paper_id": candidate.paper_id,
        "arxiv_id": candidate.arxiv_id,
        "model": f"{model[0]}/{model[1]}",
        "session_id": session_id,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "accepted": accepted,
        "attempt_complete": complete,
        "failure_reason": failure or incomplete_reason,
        "findings": findings.model_dump(mode="json") if findings else None,
        "verification": scoring,
        "all_requested_scoreable": all_requested_scoreable,
        "label_quality": {
            "github": row.get("github_label_quality"),
            "project_page": row.get("project_page_label_quality"),
        },
        "tool_events": [event.model_dump(mode="json") for event in events],
        "trajectory": portable_trajectory(messages, findings),
        "opencode_messages": messages,
    }


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
    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(row) for row in dataset]
    global ACCEPT_PARTIAL
    ACCEPT_PARTIAL = args.accept_partial
    if args.accept_partial:
        rows = [row for row in rows if row_any_requested_scoreable(row)]
    elif not args.include_unscoreable:
        rows = [row for row in rows if row_all_requested_scoreable(row)]
    rows = rows[args.start : args.start + args.limit]
    done = accepted_ids(args.output, any_attempt=args.resume_attempted) if args.resume else set()
    rows = [row for row in rows if int(row["paper_id"]) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with (
        opencode_server(args.base_url, args.opencode_command) as base_url,
        AsyncOpencode(base_url=base_url, timeout=httpx.Timeout(args.timeout, connect=10), max_retries=1) as client,
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
                    try:
                        line = json.dumps(record, ensure_ascii=False)
                        line.encode("utf-8")
                    except UnicodeEncodeError:  # lone surrogates from tool output; escaped form is still valid JSON
                        line = json.dumps(record)
                    with args.output.open("a", encoding="utf-8") as handle:
                        # OpenCode needs HF_TOKEN for Inference Providers, so the agent's shell can see it: redact.
                        handle.write(redact_credentials(line) + "\n")
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
    parser.add_argument("--opencode-command", default="opencode")
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
