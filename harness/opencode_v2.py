"""The URL-research episode in OpenCode 2.x (`--opencode-version 2.3.2`).

OpenCode 2 replaced the 1.x HTTP API. What the harness does differently (see docs/EXPERIMENT_glm53_qwen35-4b_opencode2.md):

- Routes live under `/api/`, behind HTTP Basic auth (`opencode:<password>`). The server generates the password
  and prints it once; `harness.opencode.start_opencode` reads it, so it never enters the agent's environment.
- A prompt request carries only text. The system prompt and the tool set are a config agent (`url-agent`,
  merged into OPENCODE_CONFIG_CONTENT by `server_config`): its `system` replaces OpenCode's base prompt, and
  permission rules deny every tool except shell, read, grep, glob and webfetch. The same rules are repeated on
  the session, because OpenCode appends its own rules (e.g. `skill:browser` deny) after the agent's, and a tool
  is hidden from the model only when the last rule matching it denies every resource.
- There is no structured output (no `format`, no `StructuredOutput` tool): the agent ends with a JSON-only
  message (the contract and schema are in `harness.task.system_prompt_v2`), parsed and validated by the same
  `URLFindings` model.
- Prompts are asynchronous: post the prompt, then wait for the session to go idle.
- Every episode runs on its own server (`harness.opencode.isolated_episode_server`): OpenCode 2 spills long shell
  output to files in its data directory and the agent reads them back, so episodes sharing a server read (and once
  deleted) each other's files.
- Messages are flat (`{id, type, content: [...]}`, one assistant message per model step). `normalize_messages`
  converts them to the 1.x `{info, parts}` shape, so tool events, scoring, the portable SFT trajectory and the Pi
  session export are shared with 1.x. Tool names are kept as OpenCode 2 reports them (`shell`, not `bash`).
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from typing import Any

from harness.task import candidate_from_row, system_prompt_v2, user_prompt

AGENT_ID = "url-agent"
ALLOWED_TOOLS = ("shell", "read", "grep", "glob", "webfetch")
PERMISSIONS = [
    {"action": "*", "resource": "*", "effect": "deny"},
    *({"action": tool, "resource": "*", "effect": "allow"} for tool in ALLOWED_TOOLS),
    # Same as 1.x's `external_directory: deny`: tools stay inside the episode's workspace (shell can still cd).
    {"action": "external_directory", "resource": "*", "effect": "deny"},
]
PAGE_LIMIT = 200


def server_config(content: str | dict[str, Any] | None) -> dict[str, Any]:
    """OPENCODE_CONFIG_CONTENT for a 2.x server: the caller's config (1.x `provider` entries are migrated by
    OpenCode itself) plus the `url-agent` agent."""
    if isinstance(content, str):
        config = json.loads(content or "{}")
    else:
        config = dict(content or {})
    config.pop("$schema", None)  # the 1.x schema URL; 2.x validates against its own
    for legacy in ("share", "autoupdate", "small_model"):  # 1.x keys 2.x does not need (titles are given)
        config.pop(legacy, None)
    agents = config.setdefault("agents", {})
    agents[AGENT_ID] = {
        "system": system_prompt_v2(),
        "mode": "primary",
        "description": "Finds a paper's official GitHub repository and project page",
        "permissions": PERMISSIONS,
        **agents.get(AGENT_ID, {}),
    }
    return config


# --------------------------------------------------------------------------------------------------
# Messages: OpenCode 2 -> the 1.x shape the parsers, scorer and exporter read
# --------------------------------------------------------------------------------------------------


def _tool_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):  # a call still streaming its arguments
        with suppress(ValueError):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        return {"raw": value} if value else {}
    return {}


def _tool_text(state: dict[str, Any]) -> str:
    items = state.get("content") or []
    texts = [str(item.get("text", "")) for item in items if isinstance(item, dict) and item.get("type") == "text"]
    files = [str(item.get("uri", "")) for item in items if isinstance(item, dict) and item.get("type") == "file"]
    return "\n".join(texts + [f"[file {uri}]" for uri in files])


def _error(error: Any) -> dict[str, Any] | None:
    if not error:
        return None
    if isinstance(error, dict):
        return {"name": str(error.get("type") or "error"), "data": {"message": str(error.get("message") or error)}}
    return {"name": "error", "data": {"message": str(error)}}


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenCode 2 session messages -> 1.x-style `{info, parts}` messages (user and assistant only).

    Tool state: `completed` keeps its text content as `output`; `error` keeps `error.message`; a call that was
    still `streaming`/`running` when the episode ended (interrupted) becomes an `error` with that status.
    The final `idle` message's failure, if any, is attached to the last assistant message as its error.
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        kind = message.get("type")
        created = (message.get("time") or {}).get("created")
        if kind == "user":
            text = message.get("text")
            if text is None:
                text = (message.get("payload") or {}).get("text", "")
            info = {"id": message.get("id"), "role": "user", "time": {"created": created}}
            result.append({"info": info, "parts": [{"type": "text", "text": str(text)}]})
        elif kind == "assistant":
            model = message.get("model") or {}
            times = message.get("time") or {}
            info: dict[str, Any] = {
                "id": message.get("id"),
                "role": "assistant",
                "providerID": model.get("providerID"),
                "modelID": model.get("id"),
                "agent": message.get("agent"),
                "finish": message.get("finish"),
                "cost": message.get("cost") or 0,
                "tokens": message.get("tokens") or {},
                "time": {"created": created, "completed": times.get("completed")},
            }
            if error := _error(message.get("error")):
                info["error"] = error
            parts: list[dict[str, Any]] = []
            for item in message.get("content") or []:
                item_type = item.get("type")
                if item_type in {"text", "reasoning"}:
                    parts.append({"type": item_type, "text": str(item.get("text") or "")})
                elif item_type == "tool":
                    state = item.get("state") or {}
                    status = state.get("status")
                    normalized: dict[str, Any] = {"status": status, "input": _tool_input(state.get("input"))}
                    if status == "completed":
                        normalized["output"] = _tool_text(state)
                    else:
                        error = state.get("error")
                        message_text = error.get("message") if isinstance(error, dict) else error
                        normalized["status"] = "error"
                        normalized["error"] = str(message_text or f"tool call {status or 'did not complete'}")
                    item_time = item.get("time") or {}
                    if isinstance(item_time, dict) and item_time.get("completed"):
                        normalized["time"] = {"end": item_time.get("completed")}
                    call_id = str(item.get("id"))
                    parts.append({"type": "tool", "id": call_id, "callID": call_id, "tool": str(item.get("name", "")),
                                  "state": normalized})  # fmt: skip
            result.append({"info": info, "parts": parts})
        elif kind == "idle" and message.get("outcome") == "failed" and message.get("error"):
            for entry in reversed(result):
                if entry["info"].get("role") == "assistant":
                    entry["info"].setdefault("error", _error(message.get("error")))
                    break
    return result


def tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        item.get("type") == "tool"
        for message in messages
        if message.get("type") == "assistant"
        for item in message.get("content") or []
    )


# --------------------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------------------


async def api(client: Any, method: str, path: str, body: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    response = await client.request(method.upper(), path, json=body, **kwargs)
    response.raise_for_status()
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


async def list_messages(client: Any, session_id: str) -> list[dict[str, Any]]:
    """Every message of a session, oldest first (the endpoint pages at most 200 at a time)."""
    messages: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": PAGE_LIMIT, "order": "asc"}
        if cursor:
            params["cursor"] = cursor
        page = await api(client, "get", f"/api/session/{session_id}/message", params=params)
        data = page.get("data") or []
        messages.extend(data)
        cursor = (page.get("cursor") or {}).get("next")
        if not data or not cursor or len(data) < PAGE_LIMIT:
            return messages


async def create_session(client: Any, model: tuple[str, str], title: str) -> str:
    body: dict[str, Any] = {
        "title": title,
        "agent": AGENT_ID,
        "model": {"providerID": model[0], "id": model[1]},
        "permissions": PERMISSIONS,
    }
    workspace = getattr(getattr(client, "server", None), "workspace", None)
    if workspace:
        body["location"] = {"directory": workspace}
    session = await api(client, "post", "/api/session", body)
    return str(session["data"]["id"])


async def wait_idle(client: Any, session_id: str) -> None:
    """Block until the session's loop has finished the prompt (an `idle` message follows the last user message)."""
    while True:
        with suppress(Exception):
            await api(client, "post", f"/api/experimental/session/{session_id}/wait")
        messages = await list_messages(client, session_id)
        if messages and messages[-1].get("type") == "idle":
            return
        await asyncio.sleep(0.5)


async def interrupt(client: Any, session_id: str) -> None:
    with suppress(Exception):
        await api(client, "post", f"/api/session/{session_id}/interrupt")


async def run_session(
    client: Any,
    model: tuple[str, str],
    prompt: str,
    *,
    title: str,
    episode_timeout: float,
    max_tool_calls: int | None,
) -> tuple[str, list[dict[str, Any]], str | None, bool]:
    """Create a session, send `prompt`, wait (with the tool-call watchdog and the episode timeout).

    Returns (session id, raw OpenCode 2 messages, failure or None, timed out)."""
    session_id = await create_session(client, model, title)
    failure: str | None = None
    timed_out = False
    watchdog_reason: str | None = None

    async def watchdog() -> None:
        nonlocal watchdog_reason
        while True:
            await asyncio.sleep(2)
            with suppress(Exception):
                if max_tool_calls and tool_call_count(await list_messages(client, session_id)) >= max_tool_calls:
                    watchdog_reason = f"tool-call limit reached ({max_tool_calls})"
                    await interrupt(client, session_id)
                    return

    watchdog_task = asyncio.create_task(watchdog())
    try:
        async with asyncio.timeout(episode_timeout):
            await api(client, "post", f"/api/session/{session_id}/prompt", {"text": prompt})
            await wait_idle(client, session_id)
    except TimeoutError:
        timed_out = True
        failure = f"episode timed out after {episode_timeout:g} seconds"
        await interrupt(client, session_id)
    except Exception as exc:  # noqa: BLE001 - keep failed provider attempts for diagnosis and resumption
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await watchdog_task
    if watchdog_reason or timed_out:
        with suppress(Exception):  # let the interrupt settle so the last tool call is recorded as stopped
            async with asyncio.timeout(15):
                await wait_idle(client, session_id)
    messages = await list_messages(client, session_id)
    return session_id, messages, failure or watchdog_reason, timed_out


async def run_one(
    client: Any,
    row: dict[str, Any],
    model: tuple[str, str],
    *,
    episode_timeout: float,
    max_tool_calls: int,
    title: str = "URL research",
) -> dict[str, Any]:
    """Run one paper in OpenCode 2 and return its scored episode record (same format as 1.x)."""
    from harness.opencode import build_record

    from harness.opencode import OpenCodeV2Client, isolated_episode_server

    candidate = candidate_from_row(row)
    started = time.monotonic()
    session_args = dict(
        title=f"{title} {candidate.arxiv_id}", episode_timeout=episode_timeout, max_tool_calls=max_tool_calls
    )
    server = getattr(client, "server", None)
    if server is not None and server.executable:
        # A server started by `opencode_server`: run the episode on its own server (see isolated_episode_server).
        async with (
            isolated_episode_server(server) as episode_server,
            OpenCodeV2Client(episode_server, timeout=client.timeout) as episode_client,
        ):
            session_id, messages, failure, _ = await run_session(
                episode_client, model, user_prompt(candidate), **session_args
            )
    else:
        session_id, messages, failure, _ = await run_session(client, model, user_prompt(candidate), **session_args)
    return build_record(
        row,
        f"{model[0]}/{model[1]}",
        session_id=session_id,
        messages=normalize_messages(messages),
        response={},
        failure=failure,
        elapsed_seconds=time.monotonic() - started,
        max_tool_calls=max_tool_calls,
        opencode_version=client.server.version,
        system_prompt=system_prompt_v2(),
    )
