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
STUDENT_LABEL = os.environ.get("STUDENT_LABEL", "Qwen3.5-4B URL agent (SFT, 1× A10G)")
STUDENT_GPU_PER_HOUR = float(os.environ.get("STUDENT_GPU_PER_HOUR", "1.00"))
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
        return [
            part
            for message in self.messages
            if (message.get("info") or {}).get("role") == "assistant"
            for part in message.get("parts", [])
        ]

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

SIDES = {
    "student": {"role": "Student", "detail": "Fine-tuned on the teacher's traces · single-GPU endpoint"},
    "teacher": {"role": "Teacher", "detail": "Frontier model · Hugging Face Inference Providers"},
}


def esc(value: Any, limit: int | None = None) -> str:
    text = str(value if value is not None else "")
    if limit and len(text) > limit:
        text = text[:limit].rstrip() + f" … (+{len(text) - limit:,} characters)"
    return html.escape(text)


def step_label(part: dict[str, Any]) -> tuple[str, str]:
    """(plain-language label, the exact command/target) for one tool call."""
    state = part.get("state") or {}
    data = state.get("input") or {}
    name = part.get("tool", "")
    if name == "bash":
        command = str(data.get("command") or "")
        if "hf papers info" in command:
            return "Looked up the paper's Hub metadata", command
        if "hf papers read" in command:
            return "Read the paper", command
        if "api.github.com" in command or "github.com" in command:
            return "Checked GitHub", command
        if "curl" in command or "wget" in command:
            return "Downloaded a web page", command
        return "Ran a shell command", command
    if name == "webfetch":
        return "Opened a web page", str(data.get("url") or "")
    if name == "StructuredOutput":
        return "Submitted the final answer", ""
    if name == "grep":
        return "Searched text", str(data.get("pattern") or "")
    if name == "read":
        return "Read a file", str(data.get("filePath") or data.get("path") or "")
    if name == "glob":
        return "Listed files", str(data.get("pattern") or "")
    return f"Used {name}", json.dumps(data)[:200]


def render_step(part: dict[str, Any], index: int, rejected: bool = False) -> str:
    kind = part.get("type")
    text = (part.get("text") or "").strip()
    if kind in {"reasoning", "text"} and text:
        if len(text) <= 220:
            return f'<div class="thought"><span class="thought-tag">Thinking</span>{esc(text)}</div>'
        return (
            f'<details class="thought"><summary><span class="thought-tag">Thinking</span>{esc(text[:200])}…'
            f'<span class="more">more</span></summary><div class="thought-full">{esc(text, 4000)}</div></details>'
        )
    if kind != "tool":
        return ""
    state = part.get("state") or {}
    status = state.get("status", "pending")
    timing = state.get("time") or {}
    duration = f"{(timing['end'] - timing['start']) / 1000:.1f} s" if timing.get("start") and timing.get("end") else ""
    if status == "completed":
        badge = '<span class="badge ok">✓ Done</span>'
    elif status == "error":
        badge = '<span class="badge err">✗ Failed</span>'
    else:
        badge = '<span class="badge run">Running…</span>'
    label, target = step_label(part)
    name = part.get("tool", "")
    if rejected:
        label = "Final answer rejected, retried"
        badge = '<span class="badge err">✗ Not accepted</span>'
    if name == "StructuredOutput":
        output = json.dumps(state.get("input") or {}, indent=2)
    else:
        output = str(state.get("output") or state.get("error") or "")
    target_html = f'<code class="target">{esc(target, 400)}</code>' if target else ""
    output_html = (
        f'<details class="output"><summary>Show output</summary><pre>{esc(output, 3000)}</pre></details>'
        if output.strip()
        else ""
    )
    return (
        f'<div class="step {status}"><div class="num">{index}</div><div class="step-body">'
        f'<div class="step-head"><span class="step-label">{esc(label)}</span>'
        f'<span class="step-meta">{duration}{badge}</span></div>{target_html}{output_html}</div></div>'
    )


def field_check(predicted: str | None, expected: str | None) -> tuple[str, bool | None]:
    if expected == "unknown":
        return '<span class="pill na">— No verified label</span>', None
    if canonical_url(predicted) == canonical_url(expected):
        return '<span class="pill ok">✓ Correct</span>', True
    shown = esc(expected) if expected else "none"
    return f'<span class="pill err">✗ Wrong</span><div class="expected">Expected: {shown}</div>', False


def run_score(run: AgentRun, reference: dict[str, Any] | None) -> tuple[int, int]:
    """(correct, scored) over the two URL fields."""
    if run.findings is None or reference is None:
        return 0, sum(1 for v in (reference or {}).values() if v != "unknown")
    correct = scored = 0
    for name in ("github", "project_page"):
        _, ok = field_check(getattr(run.findings, f"{name}_url"), reference.get(name))
        if ok is not None:
            scored += 1
            correct += int(ok)
    return correct, scored


def status_chip(run: AgentRun, winner: str | None) -> str:
    if run.status == "waking":
        return '<span class="chip wait">Waking endpoint…</span>'
    if run.status == "waiting":
        return '<span class="chip wait">Starting…</span>'
    if run.status == "running":
        return '<span class="chip run"><span class="dot-live"></span>Running</span>'
    if run.findings is None:
        return '<span class="chip err">✗ No answer</span>'
    if winner == run.side:
        return '<span class="chip ok">🏁 Finished first</span>'
    return '<span class="chip done">✓ Finished</span>'


def friendly_error(error: str) -> str:
    lowered = error.lower()
    if "maximum context length" in lowered or "context length" in lowered:
        message = "Ran out of context: the conversation outgrew the model's 64K-token window."
    elif "timed out" in lowered:
        message = "Ran out of time before submitting an answer."
    elif "tool-call budget" in lowered:
        message = "Hit the tool-call budget before submitting an answer."
    elif "not configured or not reachable" in lowered:
        message = "The student endpoint is unavailable right now."
    else:
        message = "The run stopped with an error."
    return (
        f'<div class="error">⚠ {message}<details><summary>Details</summary><code>{esc(error, 1000)}</code>'
        "</details></div>"
    )


def stat(label: str, value: str) -> str:
    return f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'


def render_column(run: AgentRun | None, reference: dict[str, Any] | None, winner: str | None) -> str:
    if run is None:
        return '<div class="col empty">Pick an example or enter an arXiv paper, then press <b>Race</b>.</div>'
    side = SIDES[run.side]
    tokens_in, tokens_out = run.tokens()
    tools = len(run.tool_parts())
    header = (
        f'<div class="col-head"><div class="who"><span class="swatch {run.side}"></span>'
        f'<div><div class="role">{side["role"]}</div><div class="model">{esc(run.label)}</div>'
        f'<div class="detail">{side["detail"]}</div></div></div>{status_chip(run, winner)}</div>'
    )
    stats = (
        '<div class="stats">'
        + stat("Time", f"{run.elapsed:.1f} s")
        + stat("Tool calls", str(tools))
        + stat("Tokens", f"{(tokens_in + run.cached + tokens_out) / 1000:.1f}K")
        + stat("Cost", f"${run.cost():.4f}")
        + "</div>"
    )
    parts = run.parts()
    tool_positions = [i for i, part in enumerate(parts) if part.get("type") == "tool"]
    last_tool = tool_positions[-1] if tool_positions else -1
    steps, index = [], 0
    for position, part in enumerate(parts):
        rejected = False
        if part.get("type") == "tool":
            index += 1
            if part.get("tool") == "StructuredOutput":
                # OpenCode keeps the session going when a submitted answer fails validation.
                retried = position != last_tool
                never_accepted = run.status == "done" and run.findings is None
                rejected = retried or never_accepted
        steps.append(render_step(part, index, rejected))
    if run.status == "waking":
        steps.append('<div class="placeholder">Waking the student endpoint. After 15 idle minutes it scales to zero, '
                     "so the first race can wait a few minutes.</div>")
    elif run.status == "waiting":
        steps.append('<div class="placeholder">Starting OpenCode…</div>')
    elif run.status == "running":
        steps.append('<div class="placeholder pulse">Working…</div>')
    answer = ""
    if run.status == "done":
        if run.findings is not None:
            rows = []
            for name, label in (("github", "GitHub repository"), ("project_page", "Project page")):
                value = getattr(run.findings, f"{name}_url")
                link = f'<a href="{esc(value)}" target="_blank" rel="noopener">{esc(value)}</a>' if value else '<span class="none">None found</span>'
                check, _ = field_check(value, (reference or {}).get(name)) if reference is not None else ("", None)
                rows.append(f'<div class="ans-row"><div class="ans-label">{label}</div><div class="ans-url">{link}</div><div class="ans-check">{check}</div></div>')
            answer = (
                f'<div class="answer"><div class="answer-title">Answer</div>{"".join(rows)}'
                f'<details class="why"><summary>Why these URLs</summary><p>{esc(run.findings.reasoning, 1500)}</p></details></div>'
            )
        if run.error:
            answer += friendly_error(run.error)
        if run.findings is None and not run.error:
            answer += '<div class="error">⚠ The agent did not return a structured answer.</div>'
    timeline = f'<div class="timeline">{"".join(steps)}</div>' if steps else ""
    return f'<div class="col {run.side}">{header}{stats}{answer}{timeline}</div>'


def render_scoreboard(runs: list[AgentRun] | None, reference: dict[str, Any] | None, winner: str | None) -> str:
    if not runs:
        return ""
    rows = []
    for run in runs:
        correct, scored = run_score(run, reference)
        if run.status != "done":
            result = '<span class="muted">…</span>'
        elif run.findings is None:
            result = '<span class="pill err">✗ No answer</span>'
        elif scored:
            cls = "ok" if correct == scored else ("warn" if correct else "err")
            icon = "✓" if correct == scored else ("⚠" if correct else "✗")
            result = f'<span class="pill {cls}">{icon} {correct}/{scored} correct</span>'
        else:
            result = '<span class="pill na">— No labels</span>'
        flag = " 🏁" if winner == run.side else ""
        rows.append(
            f'<tr><td><span class="swatch {run.side}"></span><b>{SIDES[run.side]["role"]}</b>{flag}'
            f'<div class="sub">{esc(run.label)}</div></td><td class="num-cell">{run.elapsed:.1f} s</td>'
            f'<td class="num-cell">{len(run.tool_parts())}</td><td class="num-cell">${run.cost():.4f}</td><td>{result}</td></tr>'
        )
    summary = ""
    student, teacher = runs
    if student.status == teacher.status == "done" and student.findings and teacher.findings:
        bits = []
        if student.cost() > 0 and teacher.cost() > 0:
            ratio = teacher.cost() / student.cost()
            bits.append(f"the student cost {ratio:.0f}× less" if ratio >= 1.5 else "both cost about the same")
        s_ok, s_n = run_score(student, reference)
        t_ok, t_n = run_score(teacher, reference)
        if s_n and t_n:
            bits.append("both got every verified URL right" if s_ok == s_n and t_ok == t_n else f"student {s_ok}/{s_n}, teacher {t_ok}/{t_n} correct")
        fastest = "Student" if winner == "student" else "Teacher"
        summary = f'<div class="summary">{fastest} finished first; {", ".join(bits)}.</div>' if bits else ""
    return (
        '<div class="board"><table><thead><tr><th>Agent</th><th class="num-cell">Time</th><th class="num-cell">Tool calls</th>'
        f'<th class="num-cell">Cost</th><th>Result</th></tr></thead><tbody>{"".join(rows)}</tbody></table>{summary}</div>'
    )


def render_paper(meta: dict[str, Any], arxiv_id: str, source: str) -> str:
    abstract = meta.get("abstract") or ""
    return (
        f'<div class="paper"><div class="paper-title">{esc(meta["title"])}</div>'
        f'<div class="paper-links"><a href="https://huggingface.co/papers/{arxiv_id}" target="_blank" rel="noopener">hf.co/papers/{arxiv_id}</a>'
        f' · <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank" rel="noopener">arXiv</a>'
        f' · answers checked against {esc(source)}</div>'
        f'<details class="abstract"><summary>Abstract</summary><p>{esc(abstract, 3000)}</p></details></div>'
    )


CSS = """
:root{--ink:#1f1f1d;--ink-2:#5f5e5a;--line:rgba(11,11,11,.12);--card:#ffffff;--soft:#f4f4f1;
--student:#eb6834;--teacher:#2a78d6;--good:#0ca30c;--good-ink:#006300;--bad:#c62f2f;--warn:#8a5d00;--link:#2a62c4}
.dark{--ink:#ecebe6;--ink-2:#a9a8a2;--line:rgba(255,255,255,.14);--card:#1f1f1d;--soft:#2a2a28;
--good-ink:#3fc43f;--bad:#f06a6a;--warn:#fab219;--link:#7fb0ff}
.paper,.board,.col,.step-body,.answer,.stat,.target,.output pre,.num,.chip,.pill{color:var(--ink)}
.paper a,.answer a,.board a{color:var(--link)}
.gradio-container{max-width:1280px!important}
#intro p{font-size:1rem;line-height:1.55;color:var(--ink-2);max-width:900px}
.paper{border:1px solid var(--line);border-radius:12px;padding:14px 16px;background:var(--card)}
.paper-title{font-size:1.15rem;font-weight:650;line-height:1.35}
.paper-links{font-size:.9rem;color:var(--ink-2);margin-top:4px}
.abstract summary,.why summary,.output summary{cursor:pointer;color:var(--ink-2);font-size:.88rem}
.abstract p{font-size:.92rem;line-height:1.55;margin:.5em 0 0}
.board{border:1px solid var(--line);border-radius:12px;padding:6px 14px 10px;background:var(--card)}
.board table{width:100%;border-collapse:collapse;font-size:.95rem;border:none!important;margin:0!important}
.board tr{border:none!important}
.board th,.board td{border-left:none!important;border-right:none!important;border-top:none!important;background:transparent!important}
.board th{font-size:.78rem;font-weight:600;color:var(--ink-2);text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)!important}
.board td{padding:10px 6px;border-bottom:1px solid var(--line)!important;vertical-align:middle}
.board tr:last-child td{border-bottom:none!important}
.board .num-cell{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.board .sub{font-size:.8rem;color:var(--ink-2);margin-left:18px}
.summary{margin-top:8px;font-size:.95rem;font-weight:500}
.swatch{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;vertical-align:baseline}
.swatch.student{background:var(--student)}.swatch.teacher{background:var(--teacher)}
.col{border:1px solid var(--line);border-radius:12px;padding:14px;background:var(--card);border-top-width:4px}
.col.student{border-top-color:var(--student)}.col.teacher{border-top-color:var(--teacher)}
.col.empty{min-height:160px;display:flex;align-items:center;justify-content:center;color:var(--ink-2);border-top-width:1px}
.col-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.who{display:flex;align-items:flex-start}.who .swatch{margin-top:6px}
.role{font-size:.78rem;font-weight:600;color:var(--ink-2);text-transform:uppercase;letter-spacing:.04em}
.model{font-size:1.05rem;font-weight:650;line-height:1.3}
.detail{font-size:.82rem;color:var(--ink-2)}
.chip{font-size:.8rem;font-weight:600;padding:3px 10px;border-radius:999px;white-space:nowrap;border:1px solid var(--line);background:var(--soft)}
.chip.ok{color:var(--good-ink)}.chip.err{color:var(--bad)}
.dot-live{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--student);margin-right:6px;animation:pulse 1.2s infinite}
.col.teacher .dot-live{background:var(--teacher)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}
.stat{background:var(--soft);border-radius:8px;padding:8px 10px}
.stat-label{font-size:.75rem;color:var(--ink-2)}
.stat-value{font-size:1.2rem;font-weight:600;line-height:1.3}
.answer{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:12px}
.answer-title{font-size:.78rem;font-weight:600;color:var(--ink-2);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.ans-row{display:grid;grid-template-columns:9.5em 1fr auto;gap:10px;align-items:start;padding:6px 0;border-bottom:1px solid var(--line)}
.ans-row:last-of-type{border-bottom:none}
.ans-label{font-size:.88rem;color:var(--ink-2)}
.ans-url{font-size:.92rem;word-break:break-all}
.ans-check{text-align:right}
.none{color:var(--ink-2);font-style:italic}
.pill{display:inline-block;font-size:.8rem;font-weight:600;padding:2px 9px;border-radius:999px;white-space:nowrap;border:1px solid var(--line)}
.pill.ok{color:var(--good-ink)}.pill.err{color:var(--bad)}.pill.warn{color:var(--warn)}.pill.na{color:var(--ink-2)}
.expected{font-size:.75rem;color:var(--ink-2);margin-top:3px;word-break:break-all;max-width:16em}
.why p{font-size:.88rem;line-height:1.5;margin:.4em 0 0}
.timeline{display:flex;flex-direction:column;gap:8px}
.step{display:flex;gap:10px;align-items:flex-start}
.num{flex:0 0 24px;height:24px;border-radius:50%;background:var(--soft);border:1px solid var(--line);font-size:.78rem;font-weight:600;display:flex;align-items:center;justify-content:center;margin-top:2px}
.step.completed .num{border-color:var(--good)}.step.error .num{border-color:var(--bad)}
.step-body{flex:1;min-width:0;border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--card)}
.step-head{display:flex;justify-content:space-between;gap:8px;align-items:center}
.step-label{font-weight:600;font-size:.93rem}
.step-meta{display:flex;gap:8px;align-items:center;font-size:.8rem;color:var(--ink-2);white-space:nowrap;font-variant-numeric:tabular-nums}
.badge{font-size:.75rem;font-weight:600}.badge.ok{color:var(--good-ink)}.badge.err{color:var(--bad)}.badge.run{color:var(--ink-2)}
.target{display:block;margin-top:5px;font-size:.82rem;white-space:pre-wrap;word-break:break-all;background:var(--soft);padding:5px 7px;border-radius:6px;max-height:4.6em;overflow:hidden}
.output{margin-top:5px}
.output pre{max-height:280px;overflow:auto;font-size:.78rem;white-space:pre-wrap;word-break:break-word;background:var(--soft);padding:8px;border-radius:6px;margin:6px 0 0}
.thought{font-size:.88rem;line-height:1.5;color:var(--ink-2);padding:2px 2px 2px 34px}
.thought summary{list-style:none;cursor:pointer}
.thought-tag{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-right:6px;border:1px solid var(--line);border-radius:4px;padding:0 4px}
.more{margin-left:6px;text-decoration:underline;font-size:.8rem}
.thought-full{margin-top:4px;white-space:pre-wrap}
.placeholder{font-size:.9rem;color:var(--ink-2);padding:8px 2px 2px 34px}
.pulse{animation:pulse 1.2s infinite}@keyframes pulse{50%{opacity:.35}}
.error{margin:4px 0 12px;color:var(--bad);font-size:.9rem}
.error summary{cursor:pointer;color:var(--ink-2);font-size:.8rem}.error code{display:block;color:var(--ink-2);font-size:.78rem;white-space:pre-wrap;margin-top:4px}
.muted{color:var(--ink-2)}
@media (max-width:760px){.stats{grid-template-columns:repeat(2,1fr)}.ans-row{grid-template-columns:1fr}.ans-check{text-align:left}}
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
    info = render_paper(meta, arxiv_id, source)
    student = AgentRun("student", STUDENT_LABEL, "student", "")
    teacher = AgentRun("teacher", TEACHER_MODEL, "teacher", TEACHER_MODEL)
    runs = [student, teacher]

    def frame(winner: str | None = None):
        return (
            info,
            render_scoreboard(runs, reference, winner),
            render_column(student, reference, winner),
            render_column(teacher, reference, winner),
        )

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
                student.status, student.error = "done", "Student endpoint is not configured or not reachable."
            while not all(task.done() for task in tasks):
                done = [r for r in runs if r.status == "done" and r.findings]
                yield frame(min(done, key=lambda r: r.elapsed).side if done else None)
                await asyncio.sleep(0.6)
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, 15)
        shutil.rmtree(workdir, ignore_errors=True)
    finished = [r for r in runs if r.findings is not None and r.finished]
    yield frame(min(finished, key=lambda r: r.elapsed).side if finished else None)


EXAMPLES = ["2604.12575", "2309.09958", "2604.11004", "2601.05851"]

with gr.Blocks(title="Small agent vs. frontier teacher", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Paper URL agent: fine-tuned 4B vs. frontier teacher")
    gr.Markdown(
        "Two agents race to find a paper's official **GitHub repository** and **project page**. Both run in the "
        "unmodified OpenCode harness with the same tools, prompt and answer format. The student is a 4B model "
        "distilled from the teacher's OpenCode traces.",
        elem_id="intro",
    )
    with gr.Row(equal_height=True):
        paper = gr.Textbox(label="arXiv ID or URL", placeholder="e.g. 2604.12575", scale=5)
        button = gr.Button("Race", variant="primary", scale=1, min_width=120)
    gr.Examples(EXAMPLES, inputs=paper, label="Examples")
    paper_info = gr.HTML()
    scoreboard = gr.HTML()
    with gr.Row(equal_height=False):
        left = gr.HTML(render_column(None, None, None))
        right = gr.HTML(render_column(None, None, None))
    gr.Markdown(
        f"Each agent gets at most {MAX_TOOL_CALLS} tool calls and {EPISODE_TIMEOUT:g} seconds. Teacher cost uses "
        "Inference Providers prices from models.dev; student cost is the endpoint's GPU time for this request. "
        "Click a step to see its output."
    )
    outputs = [paper_info, scoreboard, left, right]
    button.click(race, paper, outputs, concurrency_limit=2, show_progress="hidden")
    paper.submit(race, paper, outputs, concurrency_limit=2, show_progress="hidden")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
