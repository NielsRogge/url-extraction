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
"""Evaluate the Qwen URL-agent LoRA in OpenCode through a local vLLM server.

The script is self-contained for ``hf jobs uv run``. It serves the base model
and LoRA adapter together, runs both on the same fully scoreable held-out
papers, persists each episode as JSONL, and publishes the final results to a
private Hub dataset repository.
"""

from __future__ import annotations

import argparse
import asyncio
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
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse, urlunparse

import httpx
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from opencode_ai import AsyncOpencode
from pydantic import BaseModel, ConfigDict, HttpUrl, StringConstraints, ValidationError, field_validator

DATASET_ID = "nielsr/paper-url-extraction-v1"
ADAPTER_ID = "nielsr/qwen35-4b-url-agent-lora"
BASE_MODEL = "Qwen/Qwen3.5-4B"
# The Python SDK's current structured-output payload is accepted by 1.2.x.
# The older 1.18.14 binary pinned by the OpenEnv GRPO example expects a
# different OutputFormatJsonSchema body and cannot run this SDK evaluator.
OPENCODE_VERSION = "1.2.4"
SCOREABLE_QUALITIES = {"gold", "silver", "verified_null"}
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
MAX_PORTABLE_TOOL_OUTPUT_CHARS = 8_000
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


class URLFindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    # JSON Schema's `format: uri` is only an annotation for several guided
    # decoders. A pattern is enforceable by vLLM, preventing prose such as
    # "None.OnlyGitHub..." from being emitted into URL fields.
    github_url: StrictHttpUrl | None
    project_page_url: StrictHttpUrl | None

    @field_validator("github_url", "project_page_url", mode="before")
    @classmethod
    def normalize_null_url(cls, value: Any) -> Any:
        # Small models occasionally quote JSON null or prefix it with a
        # newline. Only exact null-like tokens are normalized; prose and
        # malformed URLs still fail validation and count as incorrect.
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"", "null", "none"}:
                return None
        return value


class PaperCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: int
    arxiv_id: str
    title: str
    abstract: str
    missing_github: bool
    missing_project_page: bool
    known_github_url: HttpUrl | None = None
    known_project_page_url: HttpUrl | None = None
    priority: int
    priority_source: str


class ToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_input: dict[str, Any]
    success: bool
    error: str | None = None
    response: str | None = None


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


def user_prompt(candidate: PaperCandidate) -> str:
    requested: list[str] = []
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


def canonical_url(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"https://{text}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    host = parsed.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if host == "github.com":
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            path = "/" + "/".join(parts[:2])
    return urlunparse(("https", host, path.lower(), "", "", ""))


def score_findings(row: dict[str, Any], findings: URLFindings) -> dict[str, Any]:
    field_scores: dict[str, bool | None] = {}
    for name in ("github", "project_page"):
        if not bool(row[f"missing_{name}"]) or row.get(f"{name}_label_quality") not in SCOREABLE_QUALITIES:
            field_scores[name] = None
            continue
        predicted = canonical_url(getattr(findings, f"{name}_url"))
        expected = canonical_url(row.get(f"reference_{name}_url"))
        field_scores[name] = predicted == expected
    scoreable = [value for value in field_scores.values() if value is not None]
    return {
        "fields": field_scores,
        "num_scoreable": len(scoreable),
        "all_scoreable_correct": bool(scoreable) and all(scoreable),
    }


def score_missing_findings(row: dict[str, Any]) -> dict[str, Any]:
    """Count every requested, scoreable field as wrong when no answer was produced."""
    field_scores = {
        name: (
            False if bool(row[f"missing_{name}"]) and row.get(f"{name}_label_quality") in SCOREABLE_QUALITIES else None
        )
        for name in ("github", "project_page")
    }
    return {
        "fields": field_scores,
        "num_scoreable": sum(value is not None for value in field_scores.values()),
        "all_scoreable_correct": False,
    }


def _successful_bash(events: list[ToolEvent], fragment: str) -> bool:
    fragment = fragment.lower()
    return any(
        event.success
        and event.tool_name.lower() in {"bash", "shell"}
        and fragment in str(event.tool_input.get("command", "")).lower()
        for event in events
    )


def _masks_hf_exit_status(event: ToolEvent) -> bool:
    if event.tool_name.lower() not in {"bash", "shell"}:
        return False
    command = str(event.tool_input.get("command", "")).strip().lower()
    return command.startswith(("hf papers info ", "hf papers read ")) and any(
        operator in command for operator in ("|", ";", "&&")
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
        if event.tool_name.lower() in {"bash", "shell"}:
            command = str(event.tool_input.get("command", "")).lower()
            if "curl" in command and wanted in command:
                return True
    return False


def derive_attempt_complete(
    candidate: PaperCandidate, findings: URLFindings, events: list[ToolEvent]
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


def portable_trajectory(messages: list[dict[str, Any]], final: URLFindings | None) -> list[dict[str, Any]]:
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
        for part in message.get("parts", []):
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif part.get("type") == "tool":
                if text_parts:
                    result.append({"role": "assistant", "content": "\n".join(text_parts)})
                    text_parts.clear()
                state = part.get("state") or {}
                call_id = str(part.get("callID") or part.get("id"))
                result.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": str(part.get("tool", "")),
                                    "arguments": json.dumps(state.get("input") or {}, separators=(",", ":")),
                                },
                            }
                        ],
                    }
                )
                output = str(state.get("output") if state.get("status") == "completed" else state.get("error") or "")
                if len(output) > MAX_PORTABLE_TOOL_OUTPUT_CHARS:
                    output = output[:MAX_PORTABLE_TOOL_OUTPUT_CHARS] + "\n[tool output truncated]"
                result.append({"role": "tool", "tool_call_id": call_id, "content": output})
        if text_parts:
            result.append({"role": "assistant", "content": "\n".join(text_parts)})
    if final is not None and not any(
        call.get("function", {}).get("name", "").lower() == "structuredoutput"
        for message in result
        for call in message.get("tool_calls", [])
    ):
        final_text = final.model_dump_json(exclude_none=False)
        if not result or result[-1].get("role") != "assistant" or result[-1].get("content", "").strip() != final_text:
            result.append({"role": "assistant", "content": final_text})
    return result


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
async def opencode_server(command: str) -> AsyncIterator[str]:
    port = free_port()
    workspace = tempfile.TemporaryDirectory(prefix="url-eval-")
    process = subprocess.Popen(  # noqa: ASYNC220 - process remains asynchronous after immediate startup
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
            "title": f"URL evaluation {candidate.arxiv_id}",
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
                    "format": {"type": "json_schema", "schema": URLFindings.model_json_schema(), "retryCount": 2},
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
    except Exception as exc:  # noqa: BLE001 - preserve provider failures in per-episode results
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
    scoring = score_findings(row, findings) if findings is not None else score_missing_findings(row)
    return {
        "version": 1,
        "paper_id": candidate.paper_id,
        "arxiv_id": candidate.arxiv_id,
        "model": f"{model[0]}/{model[1]}",
        "session_id": session_id,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "accepted": findings is not None and complete and scoring["all_scoreable_correct"],
        "attempt_complete": complete,
        "failure_reason": failure or incomplete_reason,
        "findings": findings.model_dump(mode="json") if findings else None,
        "verification": scoring,
        "label_quality": {
            "github": row.get("github_label_quality"),
            "project_page": row.get("project_page_label_quality"),
        },
        "tool_events": [event.model_dump(mode="json") for event in events],
        "trajectory": portable_trajectory(messages, findings),
        "opencode_messages": messages,
    }


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
        shutil.which("vllm") or "vllm",
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
        "--enforce-eager",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--lora-modules",
        f"url-agent={adapter_path}",
    ]
    print("Starting vLLM with the base model and URL-agent LoRA", flush=True)
    return subprocess.Popen(command, start_new_session=True)


def prepare_text_adapter(source: str, destination: Path) -> str:
    """Remove inert vision-tower tensors that text-only vLLM correctly rejects."""
    from safetensors.torch import load_file, save_file

    source_path = Path(source)
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
                        args.base_model: {"name": "Qwen3.5-4B base", "limit": model_limit},
                        "url-agent": {"name": "Qwen3.5-4B URL-agent LoRA", "limit": model_limit},
                    },
                }
            },
        }
    )


def fully_scoreable(row: dict[str, Any]) -> bool:
    qualities = [row[f"{name}_label_quality"] for name in ("github", "project_page") if bool(row[f"missing_{name}"])]
    return bool(qualities) and all(quality in SCOREABLE_QUALITIES for quality in qualities)


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with path.open() as handle:
        for line in handle:
            with suppress(ValueError, KeyError, json.JSONDecodeError):
                record = json.loads(line)
                keys.add((str(record["evaluation_model"]), int(record["paper_id"])))
    return keys


async def evaluate(args: argparse.Namespace, output_path: Path) -> None:
    rows = [dict(row) for row in load_dataset(args.dataset, split=args.split) if fully_scoreable(dict(row))]
    if args.paper_ids:
        requested_ids = {int(value) for value in args.paper_ids.split(",")}
        rows = [row for row in rows if int(row["paper_id"]) in requested_ids]
        found_ids = {int(row["paper_id"]) for row in rows}
        if missing_ids := requested_ids - found_ids:
            raise ValueError(f"Requested paper IDs were not fully scoreable in {args.split}: {sorted(missing_ids)}")
    else:
        rows = rows[args.start : args.start + args.limit]
    if not rows:
        raise ValueError("No fully scoreable held-out rows selected")
    done = completed_keys(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = [("base", args.base_model), ("adapter", "url-agent")]

    async with (
        opencode_server(str(args.opencode_binary)) as base_url,
        AsyncOpencode(
            base_url=base_url, timeout=httpx.Timeout(args.request_timeout, connect=10), max_retries=1
        ) as client,
    ):
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_model(label: str, model_id: str) -> None:
            pending = [row for row in rows if (label, int(row["paper_id"])) not in done]

            async def run_row(index: int, row: dict[str, Any]) -> None:
                async with semaphore:
                    print(f"[{label} {index}/{len(pending)}] {row['arxiv_id']}", flush=True)
                    record = await run_one(
                        client,
                        row,
                        ("local-vllm", model_id),
                        episode_timeout=args.episode_timeout,
                        max_tool_calls=args.max_tool_calls,
                    )
                    record["evaluation_model"] = label
                    record["served_model"] = model_id
                    with output_path.open("a") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                    print(
                        f"  complete={record['attempt_complete']} correct={record['verification']['all_scoreable_correct']} "
                        f"tools={len(record['tool_events'])}",
                        flush=True,
                    )

            await asyncio.gather(*(run_row(i, row) for i, row in enumerate(pending, 1)))

        for label, model_id in models:
            await run_model(label, model_id)

        # Explicitly verify that the OpenCode service remained responsive through the final episode.
        await raw_json(client, "get", "/global/health")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}}
    for label in ("base", "adapter"):
        selected = [record for record in records if record.get("evaluation_model") == label]
        field_correct = Counter()
        field_total = Counter()
        failures = Counter()
        tool_calls = 0
        tool_success = 0
        recovered_failures = 0
        total_failures = 0
        for record in selected:
            for field, value in (record.get("verification") or {}).get("fields", {}).items():
                if value is not None:
                    field_total[field] += 1
                    field_correct[field] += int(bool(value))
            events = record.get("tool_events") or []
            tool_calls += len(events)
            tool_success += sum(bool(event.get("success")) for event in events)
            for index, event in enumerate(events):
                if not event.get("success"):
                    total_failures += 1
                    if any(later.get("success") for later in events[index + 1 :]):
                        recovered_failures += 1
            if record.get("failure_reason"):
                failures[str(record["failure_reason"])] += 1
        count = len(selected)
        scoreable = sum(field_total.values())
        summary["models"][label] = {
            "episodes": count,
            "complete_episodes": sum(bool(record.get("attempt_complete")) for record in selected),
            "exact_match_episodes": sum(
                bool((record.get("verification") or {}).get("all_scoreable_correct")) for record in selected
            ),
            "field_accuracy": (sum(field_correct.values()) / scoreable) if scoreable else None,
            "fields": {
                field: {"correct": field_correct[field], "total": field_total[field]} for field in sorted(field_total)
            },
            "average_tool_calls": (tool_calls / count) if count else None,
            "tool_success_rate": (tool_success / tool_calls) if tool_calls else None,
            "failed_tool_recovery_rate": (recovered_failures / total_failures) if total_failures else None,
            "failure_reasons": dict(failures.most_common()),
        }
    base = summary["models"]["base"]
    adapter = summary["models"]["adapter"]
    summary["comparison"] = {
        "field_accuracy_delta": (
            adapter["field_accuracy"] - base["field_accuracy"]
            if adapter["field_accuracy"] is not None and base["field_accuracy"] is not None
            else None
        ),
        "exact_match_delta": (
            adapter["exact_match_episodes"] / adapter["episodes"] - base["exact_match_episodes"] / base["episodes"]
            if adapter["episodes"] and base["episodes"]
            else None
        ),
    }
    return summary


def publish(args: argparse.Namespace, output_path: Path, summary_path: Path) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.output_repo, repo_type="dataset", private=True, exist_ok=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--paper-ids", help="Comma-separated paper IDs; overrides --start and --limit")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER_ID)
    parser.add_argument("--output", type=Path, default=Path("/artifacts/qwen35-4b-url-agent-eval/validation.jsonl"))
    parser.add_argument("--output-repo", default="nielsr/qwen35-4b-url-agent-eval")
    parser.add_argument("--tag", default="", help="Prefix for published file names, e.g. 'sft-v2/'")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--episode-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be passed as a Job secret")
    workdir = Path(tempfile.mkdtemp(prefix="url-agent-eval-"))
    args.opencode_binary = install_opencode(workdir / "bin" / "opencode")
    os.environ["PATH"] = str(args.opencode_binary.parent) + os.pathsep + os.environ["PATH"]
    configure_opencode(args)
    adapter_snapshot = snapshot_download(args.adapter, token=os.environ["HF_TOKEN"])
    adapter_path = prepare_text_adapter(adapter_snapshot, workdir / "text-adapter")
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
