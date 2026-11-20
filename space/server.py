"""Web server for the race: static page + a server-sent-events stream of race snapshots."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from opencode_ai import AsyncOpencode

import core
from core import (
    EPISODE_TIMEOUT,
    MAX_TOOL_CALLS,
    STUDENT_LABEL,
    TEACHER_MODEL,
    AgentRun,
    PaperCandidate,
    canonical_url,
    opencode_config,
    paper_metadata,
    parse_arxiv_id,
    reference_for,
    run_agent,
    start_opencode,
    student_model_id,
)

STATIC = Path(__file__).parent / "static"
MAX_RACES = int(os.environ.get("MAX_CONCURRENT_RACES", "2"))
_slots = asyncio.Semaphore(MAX_RACES)

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# --------------------------------------------------------------------------------------------------
# Snapshots: everything the page draws, as plain JSON
# --------------------------------------------------------------------------------------------------


def classify(part: dict[str, Any]) -> tuple[str, str, str]:
    """(verb shown in the row, argument shown after it, what the agent is doing, for the caption)."""
    state = part.get("state") or {}
    data = state.get("input") or {}
    name = part.get("tool", "")
    if name == "bash":
        command = str(data.get("command") or "").strip()
        if command.startswith("hf papers info"):
            return "hf", command[3:], "checking the paper's Hub metadata"
        if command.startswith("hf papers read"):
            return "hf", command[3:], "reading the paper"
        if command.startswith("hf "):
            return "hf", command[3:], "querying the Hub"
        if command.startswith("curl"):
            return "curl", command[5:], "downloading a page"
        return "shell", command, "running a shell command"
    if name == "webfetch":
        return "fetch", str(data.get("url") or ""), "opening a web page"
    if name == "read":
        return "read", str(data.get("filePath") or ""), "reading a file"
    if name == "grep":
        return "grep", str(data.get("pattern") or ""), "searching text"
    if name == "glob":
        return "glob", str(data.get("pattern") or ""), "listing files"
    if name == "StructuredOutput":
        urls = [data.get("github_url"), data.get("project_page_url")]
        urls = [str(u).strip() for u in urls if u and str(u).strip().lower() not in {"none", "null", ""}]
        return "answer", " · ".join(urls) or "no URLs", "submitting its answer"
    return name or "tool", json.dumps(data)[:120], f"using {name}"


def verdict(predicted: str | None, expected: str | None) -> str:
    if expected == "unknown":
        return "unlabelled"
    return "correct" if canonical_url(predicted) == canonical_url(expected) else "wrong"


def friendly_error(error: str | None) -> str | None:
    if not error:
        return None
    lowered = error.lower()
    if "context length" in lowered:
        return "ran out of context"
    if "timed out" in lowered:
        return "ran out of time"
    if "tool-call budget" in lowered:
        return f"hit the {MAX_TOOL_CALLS}-call budget"
    if "not configured or not reachable" in lowered:
        return "model unavailable"
    return "stopped with an error"


def run_snapshot(run: AgentRun, reference: dict[str, Any]) -> dict[str, Any]:
    rows = []
    parts = run.parts()
    tools = [p for p in parts if p.get("type") == "tool"]
    last_tool = tools[-1] if tools else None
    thoughts = 0
    for part in parts:
        kind = part.get("type")
        text = " ".join(str(part.get("text") or "").split())
        if kind in {"reasoning", "text"} and text:
            # Visible reasoning between tool calls (a model trained without it simply has none).
            thoughts += 1
            rows.append({"verb": "think", "arg": text[:400], "status": "completed"})
            continue
        if kind != "tool":
            continue
        verb, arg, _ = classify(part)
        state = part.get("state") or {}
        status = state.get("status", "running")
        if part.get("tool") == "StructuredOutput" and (
            part is not last_tool or (run.status == "done" and run.findings is None)
        ):
            status = "rejected"
        output = str(state.get("output") or state.get("error") or "")
        rows.append({"verb": verb, "arg": arg, "status": status, "out": output[:300], "out_len": len(output)})
    answer = None
    correct = scored = 0
    if run.findings is not None:
        answer = {}
        for name in ("github", "project_page"):
            url = getattr(run.findings, f"{name}_url")
            expected = reference.get(name)
            v = verdict(url, expected)
            answer[name] = {"url": url, "verdict": v, "expected": None if v != "wrong" else expected}
            if v != "unlabelled":
                scored += 1
                correct += v == "correct"
    tokens_in, tokens_out = run.tokens()
    doing = classify(tools[-1])[2] if tools and run.status == "running" else None
    return {
        "side": run.side,
        "label": run.label,
        "status": run.status,
        "elapsed": round(run.elapsed, 1),
        "tool_calls": len(tools),
        "tokens": tokens_in + run.cached + tokens_out,
        "cost": round(run.cost(), 5),
        "rows": rows,
        "answer": answer,
        "correct": correct,
        "scored": scored,
        "error": friendly_error(run.error) if run.status == "done" else None,
        "doing": doing,
        "thoughts": thoughts,
    }


def caption(student: dict[str, Any], teacher: dict[str, Any]) -> str:
    def name(s: dict[str, Any]) -> str:
        return "the GRPO student" if s["side"] == "student" else f"{TEACHER_MODEL.split('/')[-1]}"

    def outcome(s: dict[str, Any]) -> str:
        if s["answer"] is None:
            return f"{name(s)} {s['error'] or 'returned no answer'} after {s['tool_calls']} tool calls"
        score = f"{s['correct']}/{s['scored']} URLs right" if s["scored"] else "an answer"
        return f"{name(s)} got {score} in {s['tool_calls']} tool calls ({s['elapsed']:.1f} s, ${s['cost']:.4f})"

    done = [s for s in (student, teacher) if s["status"] == "done"]
    if len(done) == 2:
        return f"Same paper, same harness: {outcome(student)}; {outcome(teacher)}."
    parts = []
    for s in (student, teacher):
        if s["status"] == "done":
            parts.append(outcome(s))
        elif s["status"] == "waking":
            parts.append(f"{name(s)} is waiting for its GPU to wake up")
        elif s["doing"]:
            parts.append(f"{name(s)} is {s['doing']}")
        else:
            parts.append(f"{name(s)} is starting")
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


# --------------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


HF_CLI = {"ok": None, "detail": ""}


@app.on_event("startup")
async def hf_cli_self_test() -> None:
    """The agents' first step is `hf papers info`; make a broken CLI visible instead of silently degrading runs."""
    proc = await asyncio.create_subprocess_exec(
        "hf", "papers", "info", "1907.11692", "--format", "json",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )  # fmt: skip
    out, _ = await proc.communicate()
    text = out.decode(errors="replace")
    HF_CLI["ok"] = proc.returncode == 0 and '"1907.11692"' in text
    HF_CLI["detail"] = "" if HF_CLI["ok"] else text[-500:]
    print(f"[server] hf CLI self-test: {'ok' if HF_CLI['ok'] else 'FAILED: ' + text[-300:]}", flush=True)


@app.get("/api/status")
async def status() -> JSONResponse:
    ready = False
    with suppress(Exception):
        async with httpx.AsyncClient(timeout=3) as client:
            ready = (await client.get(f"{core.STUDENT_URL}/health")).status_code == 200
    return JSONResponse(
        {"student_ready": ready, "student": STUDENT_LABEL, "teacher": TEACHER_MODEL, "hf_cli_ok": HF_CLI["ok"]}
    )


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def race_events(arxiv_id: str):
    try:
        meta = await asyncio.to_thread(paper_metadata, arxiv_id)
    except Exception as exc:  # noqa: BLE001
        yield sse("failed", {"message": str(exc)[:300]})
        return
    reference, source = reference_for(arxiv_id, meta)
    paper = {"id": arxiv_id, "title": meta["title"], "source": source}
    candidate = PaperCandidate(
        paper_id=0,
        arxiv_id=arxiv_id,
        title=meta["title"],
        abstract=meta["abstract"],
        missing_github=True,
        missing_project_page=True,
        priority=0,
        priority_source="space",
    )
    student = AgentRun("student", STUDENT_LABEL, "student", "")
    teacher = AgentRun("teacher", TEACHER_MODEL, "teacher", TEACHER_MODEL)
    started = time.monotonic()

    def frame() -> str:
        s, t = run_snapshot(student, reference), run_snapshot(teacher, reference)
        winner = None
        finished = [x for x in (s, t) if x["status"] == "done" and x["answer"] is not None]
        if finished:
            winner = min(finished, key=lambda x: x["elapsed"])["side"]
        payload = {
            "t": round(time.monotonic() - started, 2),
            "paper": paper,
            "student": s,
            "teacher": t,
            "winner": winner,
            "caption": caption(s, t),
            "budget": MAX_TOOL_CALLS,
        }
        return sse("frame", payload)

    if _slots.locked():
        yield sse("queued", {"message": "Both race slots are busy; waiting for one to free up…"})
    async with _slots:
        student.status = "waking"
        yield frame()
        student.model = await asyncio.to_thread(student_model_id) or ""
        student.status = "waiting"
        yield frame()
        workdir = Path(tempfile.mkdtemp(prefix="race-"))
        process, base_url = await start_opencode(workdir, opencode_config(student.model or None))
        try:
            async with AsyncOpencode(base_url=base_url, timeout=httpx.Timeout(EPISODE_TIMEOUT + 60, connect=10)) as client:
                tasks = [asyncio.create_task(run_agent(client, teacher, candidate))]
                if student.model:
                    tasks.append(asyncio.create_task(run_agent(client, student, candidate)))
                else:
                    student.status, student.error = "done", "Student model is not configured or not reachable."
                while not all(task.done() for task in tasks):
                    yield frame()
                    await asyncio.sleep(0.5)
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, 15)
            shutil.rmtree(workdir, ignore_errors=True)
        yield frame()
        yield sse("done", {})


@app.get("/api/race")
async def race(paper: str = Query(..., max_length=200)) -> StreamingResponse:
    arxiv_id = parse_arxiv_id(paper)
    if not arxiv_id:
        return JSONResponse({"error": "Enter an arXiv ID like 2604.12575 or an arXiv / HF papers URL."}, status_code=400)
    return StreamingResponse(
        race_events(arxiv_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
