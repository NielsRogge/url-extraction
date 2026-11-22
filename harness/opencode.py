"""Running the task in OpenCode: server lifecycle, one episode, and turning its messages into records.

`run_one` is the episode loop shared by teacher-trace generation, evaluation and the difficulty probe; the
GRPO rollout session builds its request with `message_body` and its record with the same parsers, so every
stage sees the same tools, prompt, schema and parsing.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from harness.credentials import CREDENTIAL_ENV_VARS, redact_credentials
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

# OpenCode releases the harness supports. Every run pins one (`--opencode-version`, recorded in run manifests and
# in each episode record); a binary of any other version is never used silently.
#   1.2.4: every published result before the OpenCode-v2 experiment. v1 HTTP API (`/session/...`), per-request
#          system prompt, tool toggles and native structured output (`StructuredOutput` tool).
#   2.3.2: OpenCode v2. `/api/...` routes behind a server password, config agents instead of per-request
#          system/tools, no structured output (the answer is a JSON-only final message); see harness/opencode_v2.py.
# The 1.18.14 binary pinned by the OpenEnv GRPO example expects a different OutputFormatJsonSchema body.
OPENCODE_VERSION = "1.2.4"  # the default, so existing commands reproduce published runs
OPENCODE_V2_VERSION = "2.3.2"
SUPPORTED_OPENCODE_VERSIONS = (OPENCODE_VERSION, OPENCODE_V2_VERSION)
# sha256 of the `opencode` binary inside each release's opencode-linux-<arch>.tar.gz (2.3.2: identical to the
# binary of the npm package @opencode/cli-linux-x64@2.3.2, which carries SLSA provenance).
OPENCODE_SHA256 = {
    ("1.2.4", "x64"): "d24af565cfd07504d3d1ba081a702401d0024bf0e0ede3d9d37dc2c004d5505a",
    ("1.2.4", "arm64"): "e3d5f53171cd2e27b7afb1dc4590f456e7273d1a75829051c8cacaf622f1fd30",
    ("2.3.2", "x64"): "e6543f6e5c39c40971ba38afd1819ac852c29ac735b5a89065d3d148d47f27ad",
    ("2.3.2", "arm64"): "fc1ad8f2dffed8494e31265837f78d1e6ce5200f5c1c50e953cd9b63e607efdb",
}
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


# Tool failures that mean the OpenCode server itself is broken (e.g. its workspace directory vanished, so every
# shell spawn fails), not that the model did something wrong. Episodes hitting them say nothing about the model.
HARNESS_FAILURE_MARKERS = ("posix_spawn",)


def harness_failure(record: dict[str, Any]) -> str | None:
    for event in record.get("tool_events") or []:
        text = str(event.get("response") or "") + str(event.get("error") or "")
        for marker in HARNESS_FAILURE_MARKERS:
            if marker in text:
                return text.strip().splitlines()[-1][:200] if text.strip() else marker
    return None


LINE_BREAK_ESCAPES = {"\u2028": "\\u2028", "\u2029": "\\u2029", "\x85": "\\u0085", "\x0b": "\\u000b",
                      "\x0c": "\\u000c", "\x1c": "\\u001c", "\x1d": "\\u001d", "\x1e": "\\u001e", "\r": "\\r"}  # fmt: skip


def record_line(record: dict[str, Any]) -> str:
    """One JSONL line for an episode record, written as UTF-8, with credential-shaped strings redacted.

    Tool output can contain lone UTF-16 surrogates (e.g. math symbols cut out of paper text), which UTF-8
    cannot encode; those records fall back to ASCII-escaped JSON, which is still valid and round-trips.
    """
    line = json.dumps(record, ensure_ascii=False)
    try:
        line.encode("utf-8")
    except UnicodeEncodeError:
        line = json.dumps(record)
    # Unicode line breaks (U+2028, U+0085, ...) are written raw by ensure_ascii=False, and readers that split
    # with str.splitlines() (the Hub's dataset viewer) would cut the record in two.
    for char, escaped in LINE_BREAK_ESCAPES.items():
        line = line.replace(char, escaped)
    return redact_credentials(line) + "\n"


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


def portable_trajectory(
    messages: list[dict[str, Any]], final: URLFindings | None, system_prompt: str = SYSTEM_PROMPT
) -> list[dict[str, Any]]:
    """Convert OpenCode parts into ordered OpenAI-style chat messages (the SFT format).

    Reasoning parts are kept as `reasoning_content` on the assistant message that follows them (the field chat
    templates such as Qwen3.5's and Inkling 2's render as a thinking block); `train_sft.py` drops it unless the
    student profile distills reasoning.
    """
    result: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
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
    opencode_version: str = OPENCODE_VERSION,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """The episode record every stage writes (and the Pi session exporter reads).

    `messages` are OpenCode 1.x messages ({info, parts}); 2.x messages are converted to that shape first
    (`harness.opencode_v2.normalize_messages`), so parsing, scoring and the Pi export are shared.
    """
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
        "opencode_version": opencode_version,
        "session_id": session_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "accepted": findings is not None and score["attempt_complete"] and score["verification"]["all_scoreable_correct"],
        "attempt_complete": score["attempt_complete"],
        "failure_reason": failure or score["incomplete_reason"],
        "findings": findings.model_dump(mode="json") if findings else None,
        "verification": score["verification"],
        "label_quality": {name: row.get(f"{name}_label_quality") for name in ("github", "project_page")},
        "tool_events": [event.model_dump(mode="json") for event in events],
        "trajectory": portable_trajectory(messages, findings, system_prompt),
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


def opencode_api(version: str) -> int:
    """The HTTP API generation of an OpenCode release: 1 (`/session`, 1.x) or 2 (`/api/session`, 2.x)."""
    major = int(str(version).lstrip("v").split(".")[0])
    return 2 if major >= 2 else 1


def check_supported(version: str) -> str:
    version = str(version).lstrip("v")
    if version not in SUPPORTED_OPENCODE_VERSIONS:
        raise ValueError(f"OpenCode {version} is not supported (supported: {', '.join(SUPPORTED_OPENCODE_VERSIONS)})")
    return version


def parse_opencode_version(output: str) -> str | None:
    """`opencode --version` output -> "1.2.4". 1.x prints `1.2.4`, 2.x prints `opencode v2.3.2`."""
    tokens = output.strip().split()
    if not tokens:
        return None
    version = tokens[-1].lstrip("v")
    return version if re.fullmatch(r"\d+\.\d+\.\d+(?:[-+.\w]*)?", version) else None


def binary_version(executable: str | Path) -> str | None:
    """The version an OpenCode executable reports, or None if it cannot be run."""
    try:
        result = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_opencode_version(result.stdout or result.stderr)


def _architecture() -> str:
    architecture = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if architecture is None:
        raise RuntimeError(f"Unsupported architecture: {platform.machine()}")
    return architecture


def install_opencode(destination: Path, version: str = OPENCODE_VERSION) -> Path:
    """Download the pinned OpenCode release binary and check its sha256."""
    version = check_supported(version)
    architecture = _architecture()
    url = f"https://github.com/anomalyco/opencode/releases/download/v{version}/opencode-linux-{architecture}.tar.gz"
    with urllib.request.urlopen(url, timeout=180) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        binary = tar.extractfile("opencode")
        if binary is None:
            raise RuntimeError("OpenCode archive does not contain the opencode binary")
        data = binary.read()
    expected = OPENCODE_SHA256.get((version, architecture))
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"OpenCode {version} ({architecture}) binary does not match its pinned sha256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    destination.chmod(0o755)
    return destination


def find_or_install_opencode(command: str = "opencode", version: str = OPENCODE_VERSION) -> str:
    """An OpenCode executable of exactly `version`: `command` if it reports that version, else the pinned release.

    Any other `opencode` on PATH is ignored (with a note), never used: results depend on the OpenCode version.
    """
    version = check_supported(version)
    found = shutil.which(command)
    if found:
        reported = binary_version(found)
        if reported == version:
            return found
        print(f"ignoring {found} (OpenCode {reported or 'unknown version'}); installing pinned {version}", flush=True)
    return str(install_opencode(Path(tempfile.mkdtemp(prefix="opencode-")) / "opencode", version))


def require_opencode_version(executable: str | Path, version: str) -> str:
    """Fail unless `executable` is OpenCode `version`; returns the version."""
    reported = binary_version(executable)
    if reported != check_supported(version):
        raise RuntimeError(f"{executable} is OpenCode {reported or 'unknown'}, but this run pins {version}")
    return reported


@dataclass(frozen=True)
class OpenCodeServer:
    """A running (or attached) `opencode serve`: its URL, version, v2 password and workspace directory."""

    url: str
    version: str = OPENCODE_VERSION
    password: str | None = None  # v2 only; HTTP Basic auth as user "opencode"
    workspace: str | None = None

    @property
    def api(self) -> int:
        return opencode_api(self.version)

    def __str__(self) -> str:  # never print the password
        return self.url


def server_env(
    version: str, base: dict[str, str] | None = None, *, workspace: str | Path | None = None, strip_credentials: bool = False
) -> dict[str, str]:
    """The environment of an `opencode serve` process (and so of every shell its agent runs).

    v2: no server password (it is generated and read from stdout, so the agent cannot read it from `env`),
    no ~/.claude prompts or skills, the bundled model catalog instead of a live models.dev fetch, the config agent
    of `harness.opencode_v2` merged into OPENCODE_CONFIG_CONTENT, and per-workspace data/config/state directories.
    """
    env = dict(os.environ if base is None else base)
    if strip_credentials:
        env = {key: value for key, value in env.items() if key not in CREDENTIAL_ENV_VARS}
    if opencode_api(version) == 2:
        from harness import opencode_v2

        for key in ("OPENCODE_PASSWORD", "OPENCODE_SERVER_PASSWORD"):
            env.pop(key, None)
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
        env["OPENCODE_DISABLE_MODELS_FETCH"] = "1"
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_v2.server_config(env.get("OPENCODE_CONFIG_CONTENT")))
        if workspace is not None:
            for key, name in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"), ("XDG_STATE_HOME", "state")):
                env.setdefault(key, str(Path(workspace) / f".opencode-{name}"))
    return env


def _drain(stream: Any) -> None:
    with suppress(Exception):
        for _ in stream:
            pass


def start_opencode(
    executable: str,
    version: str,
    workspace: str | Path,
    env: dict[str, str],
    *,
    port: int | None = None,
    startup_timeout: float = 120,
    new_session: bool = False,
) -> tuple[subprocess.Popen, OpenCodeServer]:
    """Start `opencode serve` on 127.0.0.1 and wait until it answers; the caller owns the process."""
    port = port or free_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [executable, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
        cwd=str(workspace),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if opencode_api(version) == 2 else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if opencode_api(version) == 2 else subprocess.DEVNULL,
        text=True,
        start_new_session=new_session,
    )
    deadline = time.monotonic() + startup_timeout
    password: str | None = None
    try:
        if opencode_api(version) == 2:
            # 2.x prints `server password <pw>` once when it generates one; read it, then keep draining stdout.
            for line in process.stdout:  # type: ignore[union-attr]
                if line.startswith("server password "):
                    password = line.split()[-1].strip()
                    break
                if time.monotonic() > deadline:
                    break
            threading.Thread(target=_drain, args=(process.stdout,), daemon=True).start()
            if password is None:
                raise RuntimeError(f"OpenCode exited during startup with status {process.poll()}")
        health = "/api/info" if opencode_api(version) == 2 else "/global/health"
        auth = ("opencode", password) if password else None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"OpenCode exited during startup with status {process.returncode}")
            with suppress(httpx.HTTPError):
                if httpx.get(url + health, auth=auth, timeout=1).is_success:
                    return process, OpenCodeServer(url, version, password, str(workspace))
            time.sleep(0.1)
        raise TimeoutError(f"OpenCode did not become healthy within {startup_timeout:g} seconds")
    except BaseException:
        stop_process(process)
        raise


def stop_process(process: subprocess.Popen, *, group: bool = False) -> None:
    with suppress(ProcessLookupError, PermissionError):
        if group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError, PermissionError):
            if group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


@asynccontextmanager
async def opencode_server(
    command: str = "opencode",
    base_url: str | None = None,
    *,
    strip_credentials: bool = False,
    version: str | None = None,
) -> AsyncIterator[OpenCodeServer]:
    """Attach to `base_url`, or run `opencode serve` in an isolated temporary workspace.

    `version` pins the OpenCode release: the executable must report it (default: whatever it reports, which must
    be a supported version). Attaching to a 2.x server takes its password from OPENCODE_PASSWORD.

    With `strip_credentials`, the server (and so every shell the agent runs) gets no HF/API tokens in its
    environment: an agent can and does run `env`, and its tool output ends up in published records.
    """
    if base_url:
        attached = check_supported(version or OPENCODE_VERSION)
        yield OpenCodeServer(base_url.rstrip("/"), attached, os.environ.get("OPENCODE_PASSWORD"))
        return
    executable = shutil.which(command) or command
    version = require_opencode_version(executable, version) if version else check_supported(binary_version(executable) or "")
    workspace = tempfile.TemporaryDirectory(prefix="url-agent-")
    env = server_env(version, workspace=workspace.name, strip_credentials=strip_credentials)
    process, server = await asyncio.to_thread(start_opencode, executable, version, workspace.name, env, startup_timeout=30)
    try:
        yield server
    finally:
        stop_process(process)
        workspace.cleanup()


class OpenCodeV2Client(httpx.AsyncClient):
    """httpx client for an OpenCode 2.x server (there is no Python SDK for the v2 API)."""

    def __init__(self, server: OpenCodeServer, **kwargs: Any) -> None:
        kwargs.pop("max_retries", None)
        auth = ("opencode", server.password) if server.password else None
        super().__init__(base_url=server.url, auth=auth, **kwargs)
        self.server = server


def opencode_client(server: OpenCodeServer | str, *, timeout: Any, max_retries: int = 1) -> Any:
    """The client `run_one` takes: the `opencode-ai` SDK client for 1.x, `OpenCodeV2Client` for 2.x."""
    if isinstance(server, str):
        server = OpenCodeServer(server)
    if server.api == 2:
        return OpenCodeV2Client(server, timeout=timeout)
    from opencode_ai import AsyncOpencode

    return AsyncOpencode(base_url=server.url, timeout=timeout, max_retries=max_retries)


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
    if isinstance(client, OpenCodeV2Client):
        from harness import opencode_v2

        return await opencode_v2.run_one(
            client, row, model, episode_timeout=episode_timeout, max_tool_calls=max_tool_calls, title=title
        )
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
