#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub>=1.13.0"]
# ///
"""Export OpenCode episode records as Pi session files for the Hub's agent-traces viewer.

The Hub renders agent traces (https://huggingface.co/docs/hub/en/agent-traces) when a dataset holds one
session per `.jsonl` file in a supported harness format. Our episode records (from
`generate_opencode_traces.py`, `evaluate_vllm.py`, ...) keep many episodes per JSONL line-file, so the
viewer cannot open them. This module converts each record's raw `opencode_messages` into Pi's session
format (version 3, the format the Hub docs recommend for custom harnesses, with a `harness` field on
the header):

    {"type": "session", "version": 3, "id": ..., "harness": "opencode", "cwd": ..., ...episode metadata}
    {"type": "model_change", ...}
    {"type": "message", "message": {"role": "custom", "customType": "system_prompt", ...}}
    {"type": "message", "message": {"role": "user", "content": [{"type": "text", ...}]}}
    {"type": "message", "message": {"role": "assistant", "content": [thinking | text | toolCall ...]}}
    {"type": "message", "message": {"role": "toolResult", "toolCallId": ..., "content": [...]}}
    ...
    {"type": "message", "message": {"role": "custom", "customType": "verification", ...}}

Reasoning parts are kept (as `thinking` blocks), so the viewer shows the model's thinking, unlike the
portable training trajectory. Hugging Face tokens are redacted.

Sessions go under `sessions/<prefix>/` in the dataset repo, and the dataset card gets a `sessions`
config (the default, so Data Studio opens the trace viewer) plus an `episodes` config for the raw
records, and the `agent-traces` tag.

Backfill an existing repo:

    uv run trace_export.py --repo nielsr/paper-url-opencode-traces-kimi
    uv run trace_export.py --repo nielsr/qwen35-9b-url-agent-eval --files sft/test.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HARNESS = "opencode"
MAX_TOOL_OUTPUT_CHARS = 60_000
TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
SESSIONS_DIR = "sessions"


def _redact(text: str) -> str:
    return TOKEN_RE.sub("hf_***REDACTED***", text)


def _iso(ms: float | None) -> str:
    if not ms:
        return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:80] or "model"


def session_filename(record: dict[str, Any]) -> str:
    model = _slug(str(record.get("evaluation_model") or record.get("model") or "model").split("/")[-1])
    arxiv = _slug(str(record.get("arxiv_id") or record.get("paper_id") or "paper"))
    attempt = record.get("attempt")
    suffix = f"_a{attempt}" if attempt is not None else ""
    return f"{arxiv}_{model}{suffix}.jsonl"


def _system_prompt(record: dict[str, Any]) -> str | None:
    for message in record.get("trajectory") or []:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    return None


def _usage(info: dict[str, Any]) -> dict[str, Any]:
    tokens = info.get("tokens") or {}
    cache = tokens.get("cache") or {}
    cost = float(info.get("cost") or 0)
    return {
        "input": tokens.get("input", 0),
        "output": tokens.get("output", 0),
        "cacheRead": cache.get("read", 0),
        "cacheWrite": cache.get("write", 0),
        "totalTokens": tokens.get("total", 0),
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": cost},
    }


def _tool_output(state: dict[str, Any]) -> tuple[str, bool]:
    if state.get("status") == "completed":
        text, error = state.get("output"), False
    else:
        text, error = state.get("error") or f"tool {state.get('status', 'did not complete')}", True
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = text[:MAX_TOOL_OUTPUT_CHARS] + f"\n[truncated {len(text) - MAX_TOOL_OUTPUT_CHARS} chars for the viewer]"
    return _redact(text), error


def record_to_pi_session(record: dict[str, Any]) -> list[dict[str, Any]]:
    """One episode record -> Pi session lines (header first)."""
    messages = record.get("opencode_messages") or []
    first_ms = next(((m.get("info") or {}).get("time", {}).get("created") for m in messages), None)
    session_id = str(record.get("session_id") or hashlib.sha1(json.dumps(record.get("trajectory"), default=str).encode()).hexdigest()[:16])
    cwd = next(((m.get("info") or {}).get("path", {}).get("cwd") for m in messages if (m.get("info") or {}).get("path")), "")
    model = str(record.get("model") or "")
    verification = record.get("verification") or {}
    name_model = record.get("evaluation_model") or model.split("/")[-1]
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "harness": HARNESS,
        "name": f"{record.get('arxiv_id', '')} · {name_model} · "
        + ("correct" if verification.get("all_scoreable_correct") else "incorrect"),
        "timestamp": _iso(first_ms),
        "cwd": cwd,
        # Extra fields are ignored by the viewer and keep the episode self-describing.
        "paper_id": record.get("paper_id"),
        "arxiv_id": record.get("arxiv_id"),
        "model": model,
        "evaluation_model": record.get("evaluation_model"),
        "difficulty_tier": record.get("difficulty_tier"),
        "accepted": record.get("accepted"),
        "attempt": record.get("attempt"),
    }
    lines: list[dict[str, Any]] = [header]
    parent: str | None = None
    counter = 0
    last_ms: list[int | float | None] = [first_ms]

    def add(entry: dict[str, Any], ms: float | None) -> None:
        nonlocal parent, counter
        counter += 1
        entry_id = f"{counter:08x}"
        ms = ms or last_ms[0]
        last_ms[0] = ms
        lines.append({"id": entry_id, "parentId": parent, "timestamp": _iso(ms), **entry})
        parent = entry_id

    provider, model_id = (model.split("/", 1) + [""])[:2] if "/" in model else ("", model)
    first_assistant = next((m.get("info") for m in messages if (m.get("info") or {}).get("role") == "assistant"), None)
    if first_assistant:
        provider = first_assistant.get("providerID") or provider
        model_id = first_assistant.get("modelID") or model_id
    add({"type": "model_change", "provider": provider, "modelId": model_id}, first_ms)

    system = _system_prompt(record)
    if system:
        add(
            {"type": "message", "message": {"role": "custom", "customType": "system_prompt", "content": _redact(system), "display": True, "timestamp": first_ms}},
            first_ms,
        )

    if not messages:  # older records: fall back to the portable chat trajectory
        for message in record.get("trajectory") or []:
            role = message.get("role")
            if role == "user":
                add({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": _redact(str(message.get("content") or ""))}]}}, None)
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if message.get("content"):
                    content.append({"type": "text", "text": _redact(str(message["content"]))})
                for call in message.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    try:
                        arguments = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {"raw": fn.get("arguments")}
                    content.append({"type": "toolCall", "id": call.get("id"), "name": fn.get("name"), "arguments": arguments})
                add({"type": "message", "message": {"role": "assistant", "content": content, "provider": provider, "model": model_id}}, None)
            elif role == "tool":
                add(
                    {"type": "message", "message": {"role": "toolResult", "toolCallId": message.get("tool_call_id"), "toolName": "", "content": [{"type": "text", "text": _redact(str(message.get("content") or ""))}], "isError": False}},
                    None,
                )

    for message in messages:
        info = message.get("info") or {}
        role = info.get("role")
        created = (info.get("time") or {}).get("created")
        parts = message.get("parts") or []
        if role == "user":
            text = "\n".join(str(p.get("text", "")) for p in parts if p.get("type") == "text")
            add({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": _redact(text)}], "timestamp": created}}, created)
            continue
        if role != "assistant":
            continue
        content = []
        results: list[tuple[dict[str, Any], int | float | None]] = []
        for part in parts:
            kind = part.get("type")
            if kind == "reasoning" and part.get("text"):
                content.append({"type": "thinking", "thinking": _redact(str(part["text"])), "thinkingSignature": "reasoning"})
            elif kind == "text" and part.get("text"):
                content.append({"type": "text", "text": _redact(str(part["text"]))})
            elif kind == "tool":
                state = part.get("state") or {}
                call_id = str(part.get("callID") or part.get("id"))
                name = str(part.get("tool", ""))
                arguments = state.get("input") or {}
                if isinstance(arguments, dict):
                    arguments = json.loads(_redact(json.dumps(arguments, ensure_ascii=False)))
                content.append({"type": "toolCall", "id": call_id, "name": name, "arguments": arguments})
                output, is_error = _tool_output(state)
                end = (state.get("time") or {}).get("end") or created
                results.append(
                    (
                        {"role": "toolResult", "toolCallId": call_id, "toolName": name, "content": [{"type": "text", "text": output}], "isError": is_error, "timestamp": end},
                        end,
                    )
                )
        assistant = {
            "role": "assistant",
            "content": content,
            "api": "openai-completions",
            "provider": info.get("providerID") or provider,
            "model": info.get("modelID") or model_id,
            "usage": _usage(info),
            "stopReason": "toolUse" if results else ("error" if info.get("error") else "stop"),
            "timestamp": created,
        }
        if info.get("error"):
            assistant["errorMessage"] = _redact(json.dumps(info["error"], ensure_ascii=False)[:2000])
        add({"type": "message", "message": assistant}, created)
        for result, ms in results:
            add({"type": "message", "message": result}, ms)

    findings = record.get("findings") or {}
    fields = verification.get("fields") or {}
    summary = [
        f"github_url: {findings.get('github_url')}  ->  {fields.get('github', 'not scored')}",
        f"project_page_url: {findings.get('project_page_url')}  ->  {fields.get('project_page', 'not scored')}",
        f"all scoreable fields correct: {verification.get('all_scoreable_correct')}",
    ]
    if record.get("failure_reason"):
        summary.append(f"failure: {record['failure_reason']}")
    add(
        {"type": "message", "message": {"role": "custom", "customType": "verification", "content": "\n".join(summary), "display": True, "details": {"findings": findings, "verification": verification}}},
        None,
    )
    return lines


def write_sessions(records: Iterable[dict[str, Any]], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in records:
        if not (record.get("opencode_messages") or record.get("trajectory")):
            continue
        path = out_dir / session_filename(record)
        with path.open("w", encoding="utf-8") as handle:
            for line in record_to_pi_session(record):
                try:
                    handle.write(json.dumps(line, ensure_ascii=False) + "\n")
                except UnicodeEncodeError:
                    handle.write(json.dumps(line) + "\n")
        count += 1
    return count


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ensure_card(api: Any, repo_id: str, episode_files: list[str]) -> None:
    """Make `sessions` the default config (trace viewer), keep raw records as `episodes`, tag agent-traces."""
    from huggingface_hub import DatasetCard
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        card = DatasetCard.load(repo_id, repo_type="dataset", token=api.token)
    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        card = DatasetCard(f"---\ntags: []\n---\n\n# {repo_id.split('/')[-1]}\n\nOpenCode URL-extraction agent episodes. "
                           "See https://github.com/NielsRogge/url-extraction.\n")
    data = card.data.to_dict()
    tags = list(dict.fromkeys([*(data.get("tags") or []), "agent-traces", "opencode"]))
    configs = [c for c in (data.get("configs") or []) if c.get("config_name") not in {"sessions", "episodes"}]
    for config in configs:
        config.pop("default", None)
    configs.insert(0, {"config_name": "sessions", "default": True, "data_files": [{"split": "train", "path": f"{SESSIONS_DIR}/**/*.jsonl"}]})
    existing = {f for c in (data.get("configs") or []) if c.get("config_name") == "episodes" for d in c.get("data_files") or [] for f in ([d["path"]] if isinstance(d.get("path"), str) else d.get("path") or [])}
    files = sorted(existing | set(episode_files))
    if files:
        configs.append({"config_name": "episodes", "data_files": [{"split": "train", "path": files}]})
    data.update(tags=tags, configs=configs)
    from huggingface_hub import DatasetCardData

    card.data = DatasetCardData(**data)
    if "agent-traces viewer" not in card.text:
        card.text += (
            "\n\n## Trace viewer\n\nEach episode is also stored as a Pi-format session under `sessions/` "
            "(one file per episode, default `sessions` config), so Data Studio opens it in the "
            "[agent-traces viewer](https://huggingface.co/docs/hub/en/agent-traces) with the model's "
            "reasoning, tool calls and results, and the final verification. The raw records are in the "
            "`episodes` config.\n"
        )
    card.push_to_hub(repo_id, repo_type="dataset", token=api.token, commit_message="Card: agent-traces viewer configs")


def publish_sessions(api: Any, repo_id: str, records_path: Path, path_in_repo: str) -> int:
    """Convert a records file and upload its sessions next to it; update the card. Returns sessions written."""
    prefix = path_in_repo.removesuffix(".jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        count = write_sessions(read_records(records_path), Path(tmp))
        if count:
            api.upload_folder(
                folder_path=tmp,
                path_in_repo=f"{SESSIONS_DIR}/{prefix}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Agent-trace sessions for {path_in_repo}",
            )
    ensure_card(api, repo_id, [path_in_repo])
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Dataset repo with episode-record JSONL files")
    parser.add_argument("--files", nargs="*", help="Record files in the repo (default: every *.jsonl outside sessions/)")
    parser.add_argument("--local-only", type=Path, help="Write sessions to this directory instead of uploading")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = args.files or [
        f for f in api.list_repo_files(args.repo, repo_type="dataset")
        if f.endswith(".jsonl") and not f.startswith(f"{SESSIONS_DIR}/") and "difficulty/" not in f and "mined/" not in f
    ]
    for name in files:
        local = Path(hf_hub_download(args.repo, name, repo_type="dataset"))
        first = next(read_records(local), {})
        if not (first.get("opencode_messages") or first.get("trajectory")):
            print(f"skip {name}: not episode records", flush=True)
            continue
        if args.local_only:
            count = write_sessions(read_records(local), args.local_only / name.removesuffix(".jsonl"))
        else:
            count = publish_sessions(api, args.repo, local, name)
        print(f"{name}: {count} sessions", flush=True)


if __name__ == "__main__":
    main()
