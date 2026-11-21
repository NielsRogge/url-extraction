"""Running the task in OpenCode: server lifecycle, one episode, and turning its messages into records.

`run_one` is the episode loop shared by teacher-trace generation, evaluation and the difficulty probe; the
GRPO rollout session builds its request with `message_body` and its record with the same parsers, so every
stage sees the same tools, prompt, schema and parsing.
"""

from __future__ import annotations

import asyncio
import io
import json
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from harness.scoring import score_episode
from harness.task import (
    SYSTEM_PROMPT,
    PaperCandidate,
    ToolEvent,
    URLFindings,
    candidate_from_row,
    findings_json_schema,
    user_prompt,
)

# The Python SDK's structured-output payload is accepted by OpenCode 1.2.x. The 1.18.14 binary pinned by the
# OpenEnv GRPO example expects a different OutputFormatJsonSchema body.
OPENCODE_VERSION = "1.2.4"
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
SESSION_PERMISSIONS = [
    {"permission": "*", "pattern": "*", "action": "allow"},
    {"permission": "external_directory", "pattern": "*", "action": "deny"},
    {"permission": "doom_loop", "pattern": "*", "action": "deny"},
]
MAX_PORTABLE_TOOL_OUTPUT_CHARS = 8_000
STRUCTURED_OUTPUT_RETRY = (
    "The provider rejected native structured output. Complete the research task above "
    "and return only one JSON object with reasoning, github_url, and project_page_url."
)


def record_line(record: dict[str, Any]) -> str:
    """One JSONL line for an episode record, written as UTF-8.

    Tool output can contain lone UTF-16 surrogates (e.g. math symbols cut out of paper text), which UTF-8
    cannot encode; those records fall back to ASCII-escaped JSON, which is still valid and round-trips.
    """
    line = json.dumps(record, ensure_ascii=False)
    try:
        line.encode("utf-8")
    except UnicodeEncodeError:
        line = json.dumps(record)
    return line + "\n"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------------------------------
# Parsing OpenCode messages
# --------------------------------------------------------------------------------------------------


def tool_events(messages: list[dict[str, Any]]) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    for message in messages:
        for part in message.get("parts", []):
            if part.get("type") != "tool":
                continue
            state = part.get("state") or {}
            events.append(
                ToolEvent(
                    tool_name=str(part.get("tool", "")),
                    tool_input=state.get("input") or {},
                    success=state.get("status") == "completed",
                    error=state.get("error"),
                    response=state.get("output"),
                )
            )
    return events


def findings_from_text(text: str) -> URLFindings | None:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        with suppress(ValueError, TypeError):
            return URLFindings.model_validate_json(candidate)
    return None


def findings_from_messages(messages: list[dict[str, Any]]) -> URLFindings | None:
    """Parse a JSON-only final answer when provider-native structured output is unavailable."""
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


def parse_findings(response: dict[str, Any], messages: list[dict[str, Any]]) -> tuple[URLFindings | None, str | None]:
    """The episode's answer: OpenCode's validated structured output, else a JSON final message."""
    findings: URLFindings | None = None
    failure: str | None = None
    structured = (response.get("info") or {}).get("structured")
    if structured is not None:
        try:
            findings = URLFindings.model_validate(structured)
        except ValidationError as exc:
            failure = f"invalid structured output: {exc}"
    if findings is None:
        findings = findings_from_messages(messages)
    return findings, failure


def provider_errors(messages: list[dict[str, Any]]) -> list[str]:
    errors = [((message.get("info") or {}).get("error") or {}).get("data", {}).get("message") for message in messages]
    return [str(error) for error in errors if error]


def portable_trajectory(messages: list[dict[str, Any]], final: URLFindings | None) -> list[dict[str, Any]]:
    """Convert OpenCode parts into ordered OpenAI-style chat messages (the SFT format).

    Reasoning parts are kept as `reasoning_content` on the assistant message that follows them (the field chat
    templates such as Qwen3.5's and Inkling 2's render as a thinking block); `train_sft.py` drops it unless the
    student profile distills reasoning.
    """
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
                call = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": str(part.get("tool", "")), "arguments": json_text(state.get("input") or {})},
                }
                result.append(assistant("", tool_calls=[call]))
                output = str(state.get("output") if state.get("status") == "completed" else state.get("error") or "")
                if len(output) > MAX_PORTABLE_TOOL_OUTPUT_CHARS:
                    output = output[:MAX_PORTABLE_TOOL_OUTPUT_CHARS] + "\n[tool output truncated in portable trajectory]"
                result.append({"role": "tool", "tool_call_id": call_id, "content": output})
        if text_parts or reasoning_parts:
            result.append(assistant("\n".join(text_parts)))
    used_structured_output = any(
        call.get("function", {}).get("name", "").lower() == "structuredoutput"
        for message in result
        for call in message.get("tool_calls", [])
    )
    if final is not None and not used_structured_output:
        existing_final = None
        if result and result[-1].get("role") == "assistant" and isinstance(result[-1].get("content"), str):
            existing_final = findings_from_text(result[-1]["content"])
        if existing_final != final:
            result.append({"role": "assistant", "content": final.model_dump_json(exclude_none=False)})
    return result


# --------------------------------------------------------------------------------------------------
# Episode records
# --------------------------------------------------------------------------------------------------


def build_record(
    row: dict[str, Any],
    model: str,
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    response: dict[str, Any],
    failure: str | None,
    elapsed_seconds: float,
    max_tool_calls: int | None,
) -> dict[str, Any]:
    """The episode record every stage writes (and the Pi session exporter reads)."""
    errors = provider_errors(messages)
    if errors:
        failure = failure or f"provider error: {errors[-1]}"
    findings, parse_failure = parse_findings(response, messages)
    failure = failure or parse_failure
    events = tool_events(messages)
    score = score_episode(row, findings, events, max_tool_calls=max_tool_calls)
    return {
        "version": 1,
        "paper_id": int(row["paper_id"]),
        "arxiv_id": str(row["arxiv_id"]),
        "model": model,
        "session_id": session_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "accepted": findings is not None and score["attempt_complete"] and score["verification"]["all_scoreable_correct"],
        "attempt_complete": score["attempt_complete"],
        "failure_reason": failure or score["incomplete_reason"],
        "findings": findings.model_dump(mode="json") if findings else None,
        "verification": score["verification"],
        "label_quality": {name: row.get(f"{name}_label_quality") for name in ("github", "project_page")},
        "tool_events": [event.model_dump(mode="json") for event in events],
        "trajectory": portable_trajectory(messages, findings),
        "opencode_messages": messages,
    }


# --------------------------------------------------------------------------------------------------
# OpenCode server and one episode
# --------------------------------------------------------------------------------------------------


async def raw_json(client: Any, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
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


def install_opencode(destination: Path, version: str = OPENCODE_VERSION) -> Path:
    architecture = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine().lower()
    )
    if architecture is None:
        raise RuntimeError(f"Unsupported architecture: {platform.machine()}")
    url = f"https://github.com/anomalyco/opencode/releases/download/v{version}/opencode-linux-{architecture}.tar.gz"
    with urllib.request.urlopen(url, timeout=180) as response:
        archive = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        binary = tar.extractfile("opencode")
        if binary is None:
            raise RuntimeError("OpenCode archive does not contain the opencode binary")
        destination.write_bytes(binary.read())
    destination.chmod(0o755)
    return destination


def find_or_install_opencode(command: str = "opencode") -> str:
    """An OpenCode executable: `command` if on PATH, else the pinned release in a temporary directory."""
    found = shutil.which(command)
    if found:
        return found
    return str(install_opencode(Path(tempfile.mkdtemp(prefix="opencode-")) / "opencode"))


@asynccontextmanager
async def opencode_server(command: str = "opencode", base_url: str | None = None) -> AsyncIterator[str]:
    """Attach to `base_url`, or run `opencode serve` in an isolated temporary workspace."""
    if base_url:
        yield base_url.rstrip("/")
        return
    executable = shutil.which(command) or command
    port = free_port()
    workspace = tempfile.TemporaryDirectory(prefix="url-agent-")
    process = subprocess.Popen(  # noqa: ASYNC220 - startup is immediate; the process stays asynchronous
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
                with suppress(httpx.HTTPError):
                    if (await health.get(f"{url}/global/health", timeout=1)).is_success:
                        break
                await asyncio.sleep(0.1)
            else:
                raise TimeoutError("OpenCode did not become healthy within 10 seconds")
        yield url
    finally:
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        if process.poll() is None:
            process.kill()
        workspace.cleanup()


def message_body(model: tuple[str, str], candidate: PaperCandidate, *, structured: bool = True) -> dict[str, Any]:
    """The prompt request: system prompt, allowed tools, structured-output schema, user prompt."""
    body: dict[str, Any] = {
        "model": {"providerID": model[0], "modelID": model[1]},
        "system": SYSTEM_PROMPT,
        "tools": {name: name in ALLOWED_TOOLS for name in OPENCODE_TOOLS},
    }
    if structured:
        body["format"] = {"type": "json_schema", "schema": findings_json_schema(), "retryCount": 2}
        body["parts"] = [{"type": "text", "text": user_prompt(candidate)}]
    else:
        body["parts"] = [{"type": "text", "text": STRUCTURED_OUTPUT_RETRY}]
    return body


async def run_one(
    client: Any,
    row: dict[str, Any],
    model: tuple[str, str],
    *,
    episode_timeout: float,
    max_tool_calls: int,
    title: str = "URL research",
) -> dict[str, Any]:
    """Run one paper in OpenCode and return its scored episode record."""
    candidate = candidate_from_row(row)
    started = time.monotonic()
    session = await raw_json(
        client, "post", "/session", {"title": f"{title} {candidate.arxiv_id}", "permission": SESSION_PERMISSIONS}
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
            path = f"/session/{session_id}/message"
            response = await raw_json(client, "post", path, message_body(model, candidate))
            if (response.get("info") or {}).get("error"):
                response = await raw_json(client, "post", path, message_body(model, candidate, structured=False))
    except TimeoutError:
        failure = f"episode timed out after {episode_timeout:g} seconds"
        with suppress(Exception):
            await raw_json(client, "post", f"/session/{session_id}/abort")
    except Exception as exc:  # noqa: BLE001 - keep failed provider attempts for diagnosis and resumption
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task
    messages = await raw_json(client, "get", f"/session/{session_id}/message")
    return build_record(
        row,
        f"{model[0]}/{model[1]}",
        session_id=session_id,
        messages=messages,
        response=response,
        failure=failure or watchdog_reason,
        elapsed_seconds=time.monotonic() - started,
        max_tool_calls=max_tool_calls,
    )
