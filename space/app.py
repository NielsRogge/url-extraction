"""Side-by-side race: a fine-tuned 4B URL agent vs. its frontier teacher, both inside real OpenCode.

Enter an arXiv paper. Two OpenCode sessions start at the same time with the same system prompt,
tools and structured-output schema used for training and evaluation. Each column streams that
agent's reasoning and tool calls live, with elapsed time, tool count, tokens and cost, and the
final URLs are checked against a reference.

Security: the agents run shell commands and read untrusted web pages, so they never see a
credential. Model calls go to a local auth proxy in this process, which adds the token (and the
optional bill-to header) on the way out; OpenCode only knows `http://127.0.0.1` and a dummy key.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import uvicorn
from opencode_ai import AsyncOpencode
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from evaluate_vllm import (
    ALLOWED_TOOLS,
    OPENCODE_TOOLS,
    SCOREABLE_QUALITIES,
    SYSTEM_PROMPT,
    PaperCandidate,
    URLFindings,
    canonical_url,
    findings_from_messages,
    install_opencode,
    raw_json,
    user_prompt,
)

# --------------------------------------------------------------------------------------------------
# Configuration (Space secrets/variables). Credentials are removed from os.environ immediately so
# no child process (OpenCode, its shell tool) can inherit them.
# --------------------------------------------------------------------------------------------------

HF_TOKEN = os.environ.pop("HF_TOKEN", None) or os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
for _key in [k for k in os.environ if "TOKEN" in k.upper() or "SECRET" in k.upper()]:
    os.environ.pop(_key, None)
BILL_TO = os.environ.get("HF_BILL_TO", "").strip() or None
STUDENT_URL = os.environ.get("STUDENT_ENDPOINT_URL", "").rstrip("/")
STUDENT_LABEL = os.environ.get("STUDENT_LABEL", "Qwen3.5-4B URL agent (SFT, 1× L4)")
STUDENT_GPU_PER_HOUR = float(os.environ.get("STUDENT_GPU_PER_HOUR", "0.80"))
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "zai-org/GLM-5.2")
TEACHER_ROUTER = "https://router.huggingface.co/v1"
LABEL_DATASET = os.environ.get("LABEL_DATASET", "nielsr/paper-url-extraction-v1")

EPISODE_TIMEOUT = 180.0
MAX_TOOL_CALLS = 15
PROXY_PORT = 7861
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

OPENCODE_BIN = shutil.which("opencode") or str(install_opencode(Path(tempfile.gettempdir()) / "oc-bin" / "opencode"))


def teacher_prices() -> tuple[float, float]:
    """USD per token (input, output) from models.dev, the table OpenCode itself uses."""
    with suppress(Exception):
        data = httpx.get("https://models.dev/api.json", timeout=10).json()
        cost = data["huggingface"]["models"][TEACHER_MODEL]["cost"]
        return float(cost["input"]) / 1e6, float(cost["output"]) / 1e6
    return 1.4e-6, 4.4e-6


TEACHER_IN, TEACHER_OUT = teacher_prices()


# --------------------------------------------------------------------------------------------------
# Local auth proxy: OpenCode -> 127.0.0.1 -> (router | student endpoint) with credentials attached.
# --------------------------------------------------------------------------------------------------

_proxy_client = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30))


async def _forward(request: Request) -> Response:
    target, path = request.path_params["target"], request.path_params["path"]
    if target == "teacher":
        url = f"{TEACHER_ROUTER}/{path}"
        headers = {"X-HF-Bill-To": BILL_TO} if BILL_TO else {}
    elif target == "student" and STUDENT_URL:
        url = f"{STUDENT_URL}/v1/{path}"
        headers = {"X-Scale-Up-Timeout": "600"}
    else:
        return Response("unknown target", status_code=404)
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    headers["Content-Type"] = request.headers.get("content-type", "application/json")
    body = await request.body()
    try:
        upstream = await _proxy_client.send(
            _proxy_client.build_request(request.method, url, headers=headers, content=body), stream=True
        )
    except httpx.HTTPError as exc:
        # A scaling-from-zero endpoint drops connections while it boots; report it like a gateway would.
        return Response(f"upstream unavailable: {type(exc).__name__}", status_code=502)

    async def stream():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    passthrough = {k: v for k, v in upstream.headers.items() if k.lower() in {"content-type", "cache-control"}}
    return StreamingResponse(stream(), status_code=upstream.status_code, headers=passthrough)


def _start_proxy() -> None:
    app = Starlette(routes=[Route("/{target}/v1/{path:path}", _forward, methods=["GET", "POST"])])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PROXY_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        with suppress(OSError), socket.create_connection(("127.0.0.1", PROXY_PORT), timeout=0.2):
            return
        time.sleep(0.1)


_start_proxy()


def student_model_id(deadline_s: float = 600) -> str | None:
    """Served model id; waits for a scaled-to-zero endpoint to wake up (cold start)."""
    if not STUDENT_URL:
        return None
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        with suppress(Exception):
            response = httpx.get(f"http://127.0.0.1:{PROXY_PORT}/student/v1/models", timeout=60)
            if response.status_code == 200:
                return response.json()["data"][0]["id"]
        time.sleep(5)
    return None


# --------------------------------------------------------------------------------------------------
# Papers and reference labels
# --------------------------------------------------------------------------------------------------

_labels: dict[str, dict[str, Any]] = {}


def _load_labels() -> None:
    with suppress(Exception):
        from datasets import load_dataset

        for split in ("train", "validation", "test"):
            for row in load_dataset(LABEL_DATASET, split=split):
                _labels[str(row["arxiv_id"])] = dict(row)


threading.Thread(target=_load_labels, daemon=True).start()


def parse_arxiv_id(text: str) -> str | None:
    match = ARXIV_RE.search(text or "")
    return match.group(1) if match else None


def paper_metadata(arxiv_id: str) -> dict[str, Any]:
    response = httpx.get(f"https://huggingface.co/api/papers/{arxiv_id}", timeout=20)
    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title") or "",
            "abstract": data.get("summary") or "",
            "github": data.get("githubRepo"),
            "project_page": data.get("projectPage"),
        }
    feed = httpx.get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}", timeout=20).text
    title = re.search(r"<entry>.*?<title>(.*?)</title>", feed, re.S)
    summary = re.search(r"<summary>(.*?)</summary>", feed, re.S)
    if not title:
        raise ValueError(f"Could not find arXiv paper {arxiv_id}")
    return {
        "title": " ".join(title.group(1).split()),
        "abstract": " ".join(summary.group(1).split()) if summary else "",
        "github": None,
        "project_page": None,
    }


def reference_for(arxiv_id: str, meta: dict[str, Any]) -> tuple[dict[str, str | None], str]:
    row = _labels.get(arxiv_id)
    if row:
        reference, verified = {}, False
        for name in ("github", "project_page"):
            if row.get(f"{name}_label_quality") in SCOREABLE_QUALITIES:
                reference[name] = row.get(f"reference_{name}_url")
                verified = True
            else:
                reference[name] = "unknown"
        if verified:
            return reference, f"verified labels from {LABEL_DATASET}"
    return {"github": meta.get("github"), "project_page": meta.get("project_page")}, "current HF paper metadata"


# --------------------------------------------------------------------------------------------------
# One OpenCode server per race, one session per agent
# --------------------------------------------------------------------------------------------------


class AgentRun:
    def __init__(self, side: str, label: str, provider: str, model: str) -> None:
        self.side, self.label, self.provider, self.model = side, label, provider, model
        self.messages: list[dict[str, Any]] = []
        self.findings: URLFindings | None = None
        self.status = "waiting"
        self.error: str | None = None
        self.started = self.finished = None
        self.cached = 0

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.monotonic()) - self.started

    def parts(self) -> list[dict[str, Any]]:
        return [part for message in self.messages for part in message.get("parts", [])]

    def tool_parts(self) -> list[dict[str, Any]]:
        return [part for part in self.parts() if part.get("type") == "tool"]

    def tokens(self) -> tuple[int, int]:
        """(uncached input, output incl. reasoning); cache reads are tracked separately, as OpenCode prices them."""
        tokens_in = tokens_out = 0
        self.cached = 0
        for message in self.messages:
            info = message.get("info") or {}
            if info.get("role") != "assistant":
                continue
            tokens = info.get("tokens") or {}
            tokens_in += int(tokens.get("input") or 0)
            self.cached += int((tokens.get("cache") or {}).get("read") or 0)
            tokens_out += int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0)
        return tokens_in, tokens_out

    def cost(self) -> float:
        tokens_in, tokens_out = self.tokens()
        if self.side == "teacher":
            return tokens_in * TEACHER_IN + tokens_out * TEACHER_OUT
        return self.elapsed / 3600 * STUDENT_GPU_PER_HOUR


def opencode_config(student_model: str | None) -> dict[str, Any]:
    providers: dict[str, Any] = {
        "teacher": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Teacher",
            "options": {"baseURL": f"http://127.0.0.1:{PROXY_PORT}/teacher/v1", "apiKey": "local-proxy"},
            "models": {TEACHER_MODEL: {"name": TEACHER_MODEL, "limit": {"context": 131072, "output": 8192}}},
        }
    }
    if student_model:
        providers["student"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Student",
            "options": {"baseURL": f"http://127.0.0.1:{PROXY_PORT}/student/v1", "apiKey": "local-proxy"},
            "models": {student_model: {"name": "student", "limit": {"context": 65536, "output": 1024}}},
        }
    return {
        "$schema": "https://opencode.ai/config.json",
        "share": "disabled",
        "autoupdate": False,
        "enabled_providers": list(providers),
        "provider": providers,
        "small_model": f"teacher/{TEACHER_MODEL}",
    }


async def start_opencode(workdir: Path, config: dict[str, Any]) -> tuple[subprocess.Popen, str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "XDG_DATA_HOME": str(workdir / "data"),
        "XDG_STATE_HOME": str(workdir / "state"),
        "XDG_CONFIG_HOME": str(workdir / "config"),
    }
    (workdir / "workspace").mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [OPENCODE_BIN, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
        cwd=workdir / "workspace",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient() as client:
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError("OpenCode exited during startup")
            with suppress(httpx.HTTPError):
                if (await client.get(f"{url}/global/health", timeout=1)).is_success:
                    return process, url
            await asyncio.sleep(0.2)
    raise TimeoutError("OpenCode did not start")


async def run_agent(client: AsyncOpencode, run: AgentRun, candidate: PaperCandidate) -> None:
    session = await raw_json(
        client,
        "post",
        "/session",
        {
            "title": f"{run.side} {candidate.arxiv_id}",
            "permission": [
                {"permission": "*", "pattern": "*", "action": "allow"},
                {"permission": "external_directory", "pattern": "*", "action": "deny"},
                {"permission": "doom_loop", "pattern": "*", "action": "deny"},
            ],
        },
    )
    session_id = session["id"]
    run.status, run.started = "running", time.monotonic()

    async def poll() -> None:
        while True:
            with suppress(Exception):
                run.messages = await raw_json(client, "get", f"/session/{session_id}/message")
                if len(run.tool_parts()) >= MAX_TOOL_CALLS:
                    run.error = f"stopped at the {MAX_TOOL_CALLS}-tool-call budget"
                    await raw_json(client, "post", f"/session/{session_id}/abort")
                    return
            await asyncio.sleep(0.5)

    poller = asyncio.create_task(poll())
    response: dict[str, Any] = {}
    try:
        async with asyncio.timeout(EPISODE_TIMEOUT):
            response = await raw_json(
                client,
                "post",
                f"/session/{session_id}/message",
                {
                    "model": {"providerID": run.provider, "modelID": run.model},
                    "system": SYSTEM_PROMPT,
                    "tools": {name: name in ALLOWED_TOOLS for name in OPENCODE_TOOLS},
                    "format": {"type": "json_schema", "schema": URLFindings.model_json_schema(), "retryCount": 2},
                    "parts": [{"type": "text", "text": user_prompt(candidate)}],
                },
            )
    except TimeoutError:
        run.error = f"timed out after {EPISODE_TIMEOUT:g} s"
        with suppress(Exception):
            await raw_json(client, "post", f"/session/{session_id}/abort")
    except Exception as exc:  # noqa: BLE001 - shown in the column
        run.error = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller
    with suppress(Exception):
        run.messages = await raw_json(client, "get", f"/session/{session_id}/message")
    structured = (response.get("info") or {}).get("structured")
    if structured is not None:
        with suppress(Exception):
            run.findings = URLFindings.model_validate(structured)
    if run.findings is None:
        run.findings = findings_from_messages(run.messages)
    errors = [((m.get("info") or {}).get("error") or {}) for m in run.messages]
    errors = [str((e.get("data") or {}).get("message") or e)[:200] for e in errors if e]
    if run.findings is None and errors and not run.error:
        run.error = errors[-1]
    run.status, run.finished = "done", time.monotonic()


# --------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------

TOOL_ICONS = {"bash": "⌨️", "webfetch": "🌐", "read": "📄", "grep": "🔎", "glob": "🗂️", "StructuredOutput": "✅"}


def esc(value: Any, limit: int | None = None) -> str:
    text = str(value if value is not None else "")
    if limit and len(text) > limit:
        text = text[:limit] + f"\n… ({len(text) - limit:,} more characters)"
    return html.escape(text)


def tool_title(part: dict[str, Any]) -> str:
    state = part.get("state") or {}
    data = state.get("input") or {}
    name = part.get("tool", "")
    if name == "bash":
        return data.get("command") or ""
    if name == "webfetch":
        return data.get("url") or ""
    if name == "StructuredOutput":
        return "final answer"
    return json.dumps(data)[:200]


def render_part(part: dict[str, Any], index: int) -> str:
    if part.get("type") == "reasoning" and (part.get("text") or "").strip():
        return f'<div class="thought">💭 {esc(part["text"].strip(), 400)}</div>'
    if part.get("type") == "text" and (part.get("text") or "").strip():
        return f'<div class="thought">💬 {esc(part["text"].strip(), 400)}</div>'
    if part.get("type") != "tool":
        return ""
    state = part.get("state") or {}
    status = state.get("status", "pending")
    timing = state.get("time") or {}
    duration = ""
    if timing.get("start") and timing.get("end"):
        duration = f"{(timing['end'] - timing['start']) / 1000:.1f}s"
    badge = {"completed": "ok", "error": "err", "running": "run", "pending": "run"}.get(status, "run")
    mark = {"ok": "✓", "err": "✗", "run": "…"}[badge]
    name = part.get("tool", "")
    if name == "StructuredOutput":
        body = esc(json.dumps(state.get("input") or {}, indent=2), 1500)
    else:
        body = esc(state.get("output") or state.get("error") or "", 1500)
    return (
        f'<details class="call {badge}"><summary><span class="idx">{index}</span>'
        f'<span class="icon">{TOOL_ICONS.get(name, "🔧")}</span><span class="tool">{esc(name)}</span>'
        f'<code>{esc(tool_title(part), 160)}</code><span class="dur">{duration}</span>'
        f'<span class="mark">{mark}</span></summary><pre>{body}</pre></details>'
    )


def verdict(predicted: str | None, expected: str | None) -> str:
    if expected == "unknown":
        return '<span class="na">no verified label</span>'
    good = canonical_url(predicted) == canonical_url(expected)
    return '<span class="good">✓ matches</span>' if good else f'<span class="bad">✗ expected {esc(expected or "none")}</span>'


def render_column(run: AgentRun | None, reference: dict[str, Any] | None, winner: str | None) -> str:
    if run is None:
        return '<div class="col empty">Enter a paper and press <b>Race</b>.</div>'
    tokens_in, tokens_out = run.tokens()
    tool_count = len(run.tool_parts())
    cost = run.cost()
    cost_note = "router price" if run.side == "teacher" else "GPU time"
    crown = " 🏁 finished first" if winner == run.side else ""
    header = (
        f'<div class="head {run.side}"><div class="name">{esc(run.label)}{crown}</div>'
        f'<div class="stats"><span>⏱ {run.elapsed:.1f}s</span><span>🔧 {tool_count} tool calls</span>'
        f'<span>🧮 {tokens_in:,} in (+{run.cached:,} cached) / {tokens_out:,} out</span><span>💲{cost:.4f} ({cost_note})</span></div></div>'
    )
    body = []
    index = 0
    for part in run.parts():
        if part.get("type") == "tool":
            index += 1
        body.append(render_part(part, index))
    if run.status == "waking":
        body.append('<div class="thought pulse">Waking the student endpoint (scale-to-zero cold start, up to a few minutes)…</div>')
    elif run.status == "waiting":
        body.append('<div class="thought">Starting… (a scaled-to-zero endpoint can take a few minutes to wake up)</div>')
    elif run.status == "running":
        body.append('<div class="thought pulse">working…</div>')
    result = ""
    if run.status == "done":
        if run.findings is not None:
            rows = []
            for name, label in (("github", "GitHub"), ("project_page", "Project page")):
                value = getattr(run.findings, f"{name}_url")
                link = f'<a href="{esc(value)}" target="_blank">{esc(value)}</a>' if value else "<i>none</i>"
                check = verdict(value, (reference or {}).get(name)) if reference is not None else ""
                rows.append(f"<tr><th>{label}</th><td>{link}</td><td>{check}</td></tr>")
            result = f'<div class="result"><table>{"".join(rows)}</table><p>{esc(run.findings.reasoning, 600)}</p></div>'
        if run.error:
            result += f'<div class="error">{esc(run.error)}</div>'
        if run.findings is None and not run.error:
            result += '<div class="error">No structured answer.</div>'
    return f'<div class="col">{header}<div class="timeline">{"".join(body)}</div>{result}</div>'


CSS = """
.col{border:1px solid var(--border-color-primary);border-radius:12px;padding:12px;min-height:420px;background:var(--background-fill-primary)}
.col.empty{display:flex;align-items:center;justify-content:center;color:var(--body-text-color-subdued)}
.head{border-radius:8px;padding:10px 12px;margin-bottom:10px;color:#fff}
.head.student{background:linear-gradient(90deg,#ff9d00,#ff6a00)}.head.teacher{background:linear-gradient(90deg,#4f46e5,#7c3aed)}
.head .name{font-weight:700;font-size:1.05em}.stats{display:flex;flex-wrap:wrap;gap:12px;font-size:.88em;margin-top:4px;opacity:.95}
.timeline{display:flex;flex-direction:column;gap:6px}
.call{border:1px solid var(--border-color-primary);border-left-width:4px;border-radius:8px;padding:2px 8px;background:var(--background-fill-secondary)}
.call.ok{border-left-color:#16a34a}.call.err{border-left-color:#dc2626}.call.run{border-left-color:#f59e0b}
.call summary{display:flex;align-items:center;gap:8px;cursor:pointer;padding:4px 0;list-style:none}
.call summary code{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.85em}
.idx{font-size:.75em;color:var(--body-text-color-subdued);width:1.4em;text-align:right}.tool{font-weight:600;font-size:.88em}
.dur{font-size:.8em;color:var(--body-text-color-subdued)}.mark{font-weight:700}
.call pre{max-height:260px;overflow:auto;font-size:.78em;white-space:pre-wrap;word-break:break-word}
.thought{font-size:.85em;color:var(--body-text-color-subdued);font-style:italic;padding:2px 6px}
.pulse{animation:pulse 1.2s infinite}@keyframes pulse{50%{opacity:.35}}
.result{margin-top:12px;border-radius:8px;padding:10px;background:var(--background-fill-secondary)}
.result table{width:100%;font-size:.9em}.result th{text-align:left;width:7.5em}.result p{font-size:.85em;margin:.5em 0 0}
.good{color:#16a34a;font-weight:600}.bad{color:#dc2626;font-weight:600}.na{color:var(--body-text-color-subdued)}
.error{margin-top:8px;color:#dc2626;font-size:.85em}
.paper{font-size:.92em}.paper b{font-size:1.05em}
"""


# --------------------------------------------------------------------------------------------------
# Gradio
# --------------------------------------------------------------------------------------------------


async def race(paper: str):
    arxiv_id = parse_arxiv_id(paper)
    if not arxiv_id:
        raise gr.Error("Enter an arXiv ID (e.g. 2604.12575) or an arxiv.org / huggingface.co/papers URL.")
    try:
        meta = await asyncio.to_thread(paper_metadata, arxiv_id)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(str(exc)) from exc
    reference, source = reference_for(arxiv_id, meta)
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
    info = (
        f'<div class="paper"><b>{esc(meta["title"])}</b> · '
        f'<a href="https://huggingface.co/papers/{arxiv_id}" target="_blank">{arxiv_id}</a><br>'
        f"Both agents get the same prompt, tools and output schema. Checked against {esc(source)}.</div>"
    )
    student = AgentRun("student", STUDENT_LABEL, "student", "")
    teacher = AgentRun("teacher", f"{TEACHER_MODEL} (frontier teacher, Inference Providers)", "teacher", TEACHER_MODEL)
    yield info, render_column(student, reference, None), render_column(teacher, reference, None)

    student.status = "waking"
    yield info, render_column(student, reference, None), render_column(teacher, reference, None)
    student.model = await asyncio.to_thread(student_model_id) or ""
    student.status = "waiting"
    workdir = Path(tempfile.mkdtemp(prefix="race-"))
    process, base_url = await start_opencode(workdir, opencode_config(student.model or None))
    try:
        async with AsyncOpencode(base_url=base_url, timeout=httpx.Timeout(EPISODE_TIMEOUT + 60, connect=10)) as client:
            tasks = [asyncio.create_task(run_agent(client, teacher, candidate))]
            if student.model:
                tasks.append(asyncio.create_task(run_agent(client, student, candidate)))
            else:
                student.status, student.error = "done", "Student endpoint is not configured or not reachable."
            while not all(task.done() for task in tasks):
                done = [r for r in (student, teacher) if r.status == "done" and r.findings]
                winner = min(done, key=lambda r: r.elapsed).side if done else None
                yield info, render_column(student, reference, winner), render_column(teacher, reference, winner)
                await asyncio.sleep(0.6)
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, 15)
        shutil.rmtree(workdir, ignore_errors=True)
    finished = [r for r in (student, teacher) if r.findings is not None and r.finished]
    winner = min(finished, key=lambda r: r.elapsed).side if finished else None
    yield info, render_column(student, reference, winner), render_column(teacher, reference, winner)


EXAMPLES = ["2604.12575", "2309.09958", "2604.11004", "2601.05851"]

with gr.Blocks(title="Small agent vs. frontier teacher", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🏎️ Paper URL agent: fine-tuned 4B vs. frontier teacher\n"
        "Both agents run in the unmodified **OpenCode** harness with the same tools, prompt and JSON schema. "
        "The small model was distilled from the teacher's OpenCode traces (and optionally refined with GRPO through "
        "OpenEnv). Enter an arXiv paper to watch them find its official **GitHub repo** and **project page**."
    )
    with gr.Row():
        paper = gr.Textbox(label="arXiv ID or URL", placeholder="2604.12575", scale=4)
        button = gr.Button("Race", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=paper)
    paper_info = gr.HTML()
    with gr.Row(equal_height=False):
        left = gr.HTML(render_column(None, None, None))
        right = gr.HTML(render_column(None, None, None))
    gr.Markdown(
        f"Each agent may use at most {MAX_TOOL_CALLS} tool calls and {EPISODE_TIMEOUT:g} s. Teacher cost uses router "
        "prices from models.dev; student cost is the L4 endpoint's GPU time for this request. "
        "Code: [paper-url agent pipeline](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-eval-v2)."
    )
    button.click(race, paper, [paper_info, left, right], concurrency_limit=2)
    paper.submit(race, paper, [paper_info, left, right], concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
