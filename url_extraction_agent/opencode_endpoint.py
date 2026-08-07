"""Reusable OpenCode harness for the managed URL-agent endpoint."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from opencode_ai import AsyncOpencode
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError, field_validator

from .schema import AgentAttempt, PaperCandidate, ToolEvent, URLFindings

OPENCODE_VERSION = "1.2.4"
DEFAULT_PROVIDER_ID = "hf-endpoint"
DEFAULT_MODEL_ID = "nielsr/qwen35-4b-url-agent-merged"
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
SYSTEM_PROMPT = """You find the official GitHub repository and official project page for one research paper.

Research rules:
1. Use the hf-cli skill first. Run `hf papers info PAPER_ID --format json`.
2. If a requested URL is still missing, inspect the paper with
   `hf papers read PAPER_ID`. You may use Bash, Read, Grep, Glob, and WebFetch
   to inspect the resulting public material.
   Run the two `hf papers` commands exactly without shell pipes so their real
   success or failure can be observed by the orchestrator.
3. A repository must be the authors' official implementation for this paper.
   Reject forks, mirrors, unrelated repositories, and comparison/baseline code.
4. A project page must be an official paper-specific page, not arXiv,
   Hugging Face, a generic author page, or a GitHub repository.
5. You may fetch project-page content with WebFetch or `curl` only while the
   GitHub URL is still missing. If GitHub is known, do not fetch project-page
   content, even to verify the project page; return a project-page URL obtained
   from trusted metadata without opening it. When GitHub is missing and a
   project page is found, inspect that page for a repository link before
   concluding that GitHub is unavailable.
6. Research only fields marked missing. Return null when no supported URL is
   found. Do not guess.

Your final response is validated against the supplied JSON Schema. Keep
reasoning concise and put only canonical HTTP(S) URLs in URL fields."""

StrictHttpUrl = Annotated[str, StringConstraints(pattern=r"^https?://[^\s]+$")]


class StructuredFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    github_url: StrictHttpUrl | None
    project_page_url: StrictHttpUrl | None

    @field_validator("github_url", "project_page_url", mode="before")
    @classmethod
    def normalize_null_url(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value


def endpoint_config(base_url: str, model_id: str, provider_id: str = DEFAULT_PROVIDER_ID) -> dict[str, Any]:
    return {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": [provider_id],
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Hugging Face Inference Endpoint",
                "options": {
                    "baseURL": base_url.rstrip("/") + "/v1",
                    "apiKey": "{env:HF_TOKEN}",
                    "headers": {"X-Scale-Up-Timeout": "600"},
                },
                "models": {
                    model_id: {
                        "name": "Qwen3.5 4B URL Agent SFT",
                        "limit": {"context": 65_536, "output": 1_024},
                    }
                },
            }
        },
    }


def user_prompt(candidate: PaperCandidate) -> str:
    requested = []
    if candidate.missing_github:
        requested.append("GitHub repository")
    if candidate.missing_project_page:
        requested.append("project page")
    return "\n".join(
        [
            f"Paper ID: {candidate.arxiv_id}",
            f"Title: {candidate.title}",
            f"Abstract: {candidate.abstract}",
            f"Find only: {', '.join(requested)}",
            f"Known GitHub URL: {candidate.known_github_url or 'none'}",
            f"Known project page URL: {candidate.known_project_page_url or 'none'}",
        ]
    )


def _successful_bash(events: list[ToolEvent], fragment: str) -> bool:
    fragment = fragment.lower()
    return any(
        event.success
        and event.tool_name.lower() in {"bash", "shell"}
        and fragment in str(event.tool_input.get("command", "")).lower()
        for event in events
    )


def _masks_hf_exit_status(event: ToolEvent) -> bool:
    command = str(event.tool_input.get("command", "")).strip().lower()
    return (
        event.tool_name.lower() in {"bash", "shell"}
        and command.startswith(("hf papers info ", "hf papers read "))
        and any(operator in command for operator in ("|", ";", "&&"))
    )


def _is_external_page_fetch(event: ToolEvent) -> bool:
    if event.tool_name.lower() in {"webfetch", "web_fetch"}:
        url = str(event.tool_input.get("url", ""))
    elif event.tool_name.lower() in {"bash", "shell"}:
        command = str(event.tool_input.get("command", ""))
        if "curl" not in command.lower():
            return False
        match = re.search(r"https?://[^\s\"']+", command)
        url = match.group(0) if match else ""
    else:
        return False
    host = urlparse(url).netloc.lower().removeprefix("www.")
    research_hosts = {"api.github.com", "github.com", "huggingface.co", "arxiv.org", "google.com", "paperswithcode.com"}
    return bool(host) and host not in research_hosts


def _successful_arxiv_fetch(events: list[ToolEvent], arxiv_id: str, paths: tuple[str, ...]) -> bool:
    return any(
        event.success
        and event.tool_name.lower() in {"webfetch", "web_fetch"}
        and "arxiv.org" in str(event.tool_input.get("url", "")).lower()
        and arxiv_id.lower() in str(event.tool_input.get("url", "")).lower()
        and any(path in str(event.tool_input.get("url", "")).lower() for path in paths)
        for event in events
    )


def _successful_page_fetch(events: list[ToolEvent], url: str) -> bool:
    wanted = url.rstrip("/").lower()
    for event in events:
        if not event.success:
            continue
        if (
            event.tool_name.lower() in {"webfetch", "web_fetch"}
            and str(event.tool_input.get("url", "")).rstrip("/").lower() == wanted
        ):
            return True
        command = str(event.tool_input.get("command", "")).lower()
        if event.tool_name.lower() in {"bash", "shell"} and "curl" in command and wanted in command:
            return True
    return False


def derive_attempt_complete(
    candidate: PaperCandidate, findings: StructuredFindings, events: list[ToolEvent]
) -> tuple[bool, str | None]:
    github_discovered = False
    for event in events:
        if _masks_hf_exit_status(event):
            return False, "an hf papers command masked its exit status"
        if github_discovered and _is_external_page_fetch(event):
            return False, "project-page content was fetched after GitHub discovery"
        if event.success and "github.com/" in str(event.response or "").lower():
            github_discovered = True
    metadata_checked = _successful_bash(events, f"hf papers info {candidate.arxiv_id}") or _successful_arxiv_fetch(
        events, candidate.arxiv_id, ("/abs/",)
    )
    if not metadata_checked:
        return False, "paper metadata was not successfully inspected"
    unresolved = (candidate.missing_github and findings.github_url is None) or (
        candidate.missing_project_page and findings.project_page_url is None
    )
    paper_checked = _successful_bash(events, f"hf papers read {candidate.arxiv_id}") or _successful_arxiv_fetch(
        events, candidate.arxiv_id, ("/html/", "/pdf/")
    )
    if unresolved and not paper_checked:
        return False, "paper content was not successfully inspected"
    if candidate.missing_github and findings.github_url is None:
        project_url = findings.project_page_url or candidate.known_project_page_url
        if project_url and not _successful_page_fetch(events, str(project_url)):
            return False, "project page was not inspected for a GitHub link"
    return True, None


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


def findings_from_text(text: str) -> StructuredFindings | None:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        with suppress(ValueError, TypeError):
            return StructuredFindings.model_validate_json(candidate)
    return None


def findings_from_messages(messages: list[dict[str, Any]]) -> StructuredFindings | None:
    texts = [
        str(part.get("text", ""))
        for message in messages
        if (message.get("info") or {}).get("role") == "assistant"
        for part in message.get("parts", [])
        if part.get("type") == "text" and part.get("text")
    ]
    for text in reversed(texts):
        if findings := findings_from_text(text):
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
async def opencode_server(command: str) -> AsyncIterator[str]:
    port = free_port()
    workspace = tempfile.TemporaryDirectory(prefix="url-agent-")
    process = subprocess.Popen(  # noqa: ASYNC220
        [command, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
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
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        if process.poll() is None:
            process.kill()
        workspace.cleanup()


def install_opencode(destination: Path) -> Path:
    architecture = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine().lower()
    )
    if architecture is None:
        raise RuntimeError(f"Unsupported architecture: {platform.machine()}")
    url = (
        f"https://github.com/anomalyco/opencode/releases/download/v{OPENCODE_VERSION}/"
        f"opencode-linux-{architecture}.tar.gz"
    )
    with urllib.request.urlopen(url, timeout=180) as response:
        archive = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=__import__("io").BytesIO(archive), mode="r:gz") as tar:
        binary = tar.extractfile("opencode")
        if binary is None:
            raise RuntimeError("OpenCode archive does not contain the opencode binary")
        destination.write_bytes(binary.read())
    destination.chmod(0o755)
    return destination


def wait_for_endpoint(endpoint_url: str, token: str, timeout: float) -> None:
    """Wake a scaled-to-zero Endpoint before starting any episode timers."""
    deadline = time.monotonic() + timeout
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Scale-Up-Timeout": str(min(600, max(1, int(timeout)))),
    }
    last_status: int | None = None
    with httpx.Client(timeout=httpx.Timeout(min(timeout, 610), connect=30)) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(endpoint_url.rstrip("/") + "/health", headers=headers)
                last_status = response.status_code
                if response.is_success:
                    return
                if response.status_code in {401, 403}:
                    response.raise_for_status()
            except (httpx.TimeoutException, httpx.NetworkError):
                pass
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(10, remaining))
    raise TimeoutError(f"Endpoint did not become healthy within {timeout:g} seconds (last status: {last_status})")


async def run_one(
    client: AsyncOpencode,
    candidate: PaperCandidate,
    model: tuple[str, str],
    *,
    episode_timeout: float,
    max_tool_calls: int,
) -> dict[str, Any]:
    started = time.monotonic()
    session = await raw_json(
        client,
        "post",
        "/session",
        {
            "title": f"URL extraction {candidate.arxiv_id}",
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
                    "format": {
                        "type": "json_schema",
                        "schema": StructuredFindings.model_json_schema(),
                        "retryCount": 2,
                    },
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
                                    "The provider rejected native structured output. Complete the research task "
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
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task
    failure = failure or watchdog_reason
    messages = await raw_json(client, "get", f"/session/{session_id}/message")
    events = tool_events(messages)
    findings: StructuredFindings | None = None
    structured = (response.get("info") or {}).get("structured")
    if structured is not None:
        try:
            findings = StructuredFindings.model_validate(structured)
        except ValidationError as exc:
            failure = failure or f"invalid structured output: {exc}"
    if findings is None:
        findings = findings_from_messages(messages)
    complete, incomplete_reason = (False, "no valid structured output")
    if findings is not None:
        complete, incomplete_reason = derive_attempt_complete(candidate, findings, events)
    if len(events) >= max_tool_calls:
        complete, incomplete_reason = False, f"tool-call limit reached ({max_tool_calls})"
    return {
        "session_id": session_id,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "attempt_complete": complete,
        "failure_reason": failure or incomplete_reason,
        "findings": findings.model_dump(mode="json") if findings else None,
        "tool_events": [event.model_dump(mode="json") for event in events],
    }


def attempt_from_episode(candidate: PaperCandidate, episode: dict[str, Any]) -> AgentAttempt:
    findings = URLFindings.model_validate(episode["findings"]) if episode.get("findings") else None
    return AgentAttempt(
        paper_id=candidate.paper_id,
        arxiv_id=candidate.arxiv_id,
        findings=findings,
        attempt_complete=bool(episode["attempt_complete"]),
        failure_reason=episode.get("failure_reason"),
        result_subtype="success" if findings is not None else "error",
        tool_events=[ToolEvent.model_validate(event) for event in episode.get("tool_events", [])],
    )


async def run_candidates(
    args: Any,
    candidates: list[PaperCandidate],
    on_result: Callable[[AgentAttempt], Awaitable[None]] | None = None,
) -> list[AgentAttempt]:
    wait_for_endpoint(args.endpoint_url, os.environ["HF_TOKEN"], args.endpoint_startup_timeout)
    config = endpoint_config(args.endpoint_url, args.model_id, args.provider_id)
    config["provider"][args.provider_id]["options"]["apiKey"] = os.environ["HF_TOKEN"]
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    binary = shutil.which(args.opencode_binary)
    if binary is None:
        binary = str(install_opencode(Path(tempfile.gettempdir()) / "url-agent-bin" / "opencode"))
    async with (
        opencode_server(binary) as base_url,
        AsyncOpencode(
            base_url=base_url,
            timeout=httpx.Timeout(args.request_timeout, connect=10),
            max_retries=1,
        ) as client,
    ):
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_candidate(candidate: PaperCandidate) -> AgentAttempt:
            async with semaphore:
                episode = await run_one(
                    client,
                    candidate,
                    (args.provider_id, args.model_id),
                    episode_timeout=args.episode_timeout,
                    max_tool_calls=args.max_tool_calls,
                )
                attempt = attempt_from_episode(candidate, episode)
                if on_result is not None:
                    await on_result(attempt)
                return attempt

        return list(await asyncio.gather(*(run_candidate(candidate) for candidate in candidates)))
