"""Race core: two OpenCode agents (fine-tuned student vs. frontier teacher) on the same paper.

Shared by the web server. Each race starts one OpenCode server with two sessions that use the same
system prompt, tools and structured-output schema as training and evaluation.

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
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from opencode_ai import AsyncOpencode
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

# The Space bundles harness/ next to this file (scripts/serve_space.py); in a checkout it is one level up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from harness.opencode import ALLOWED_TOOLS, OPENCODE_TOOLS, findings_from_messages, install_opencode, raw_json  # noqa: E402
from harness.scoring import SCOREABLE_QUALITIES, canonical_url  # noqa: E402
from harness.task import SYSTEM_PROMPT, PaperCandidate, URLFindings, user_prompt  # noqa: E402

# --------------------------------------------------------------------------------------------------
# Configuration (Space secrets/variables). Credentials are removed from os.environ immediately so
# no child process (OpenCode, its shell tool) can inherit them.
# --------------------------------------------------------------------------------------------------

def _read_token() -> str | None:
    """The launcher hands the token over in a one-time file and strips it from every environment."""
    path = Path(os.environ.pop("HF_TOKEN_FILE", "/tmp/.hf_token"))
    token = None
    if path.exists():
        token = path.read_text().strip() or None
        path.unlink(missing_ok=True)
    return token or os.environ.pop("HF_TOKEN", None) or os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)


HF_TOKEN = _read_token()
for _key in [k for k in os.environ if "TOKEN" in k.upper() or "SECRET" in k.upper()]:
    os.environ.pop(_key, None)
BILL_TO = os.environ.get("HF_BILL_TO", "").strip() or None
# The student runs on this Space's own GPU (vLLM started by launch.py).
STUDENT_URL = os.environ.get("STUDENT_URL", "http://127.0.0.1:8000").rstrip("/")
STUDENT_LABEL = os.environ.get("STUDENT_LABEL", "Qwen3.5-4B · SFT + GRPO")
STUDENT_GPU_PER_HOUR = float(os.environ.get("STUDENT_GPU_PER_HOUR", "0.80"))
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "zai-org/GLM-5.3")
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
    if structured is None:
        # Fall back to the last StructuredOutput call OpenCode accepted.
        for part in reversed(run.parts()):
            if part.get("type") == "tool" and part.get("tool") == "StructuredOutput":
                structured = (part.get("state") or {}).get("input")
                break
    if isinstance(structured, dict):
        with suppress(Exception):
            run.findings = URLFindings.model_validate(structured)
        if run.findings is None:
            # OpenCode accepted this answer; read its URLs leniently (missing reasoning, extra keys,
            # URL strings with stray text) so the page shows what the agent actually answered.
            def url(value: Any) -> str | None:
                match = re.search(r"https?://\S+", str(value or ""))
                return match.group(0).rstrip(").,;") if match else None

            with suppress(Exception):
                run.findings = URLFindings(
                    reasoning=str(structured.get("reasoning") or ""),
                    github_url=url(structured.get("github_url")),
                    project_page_url=url(structured.get("project_page_url")),
                )
    if run.findings is None:
        run.findings = findings_from_messages(run.messages)
    errors = [((m.get("info") or {}).get("error") or {}) for m in run.messages]
    errors = [str((e.get("data") or {}).get("message") or e)[:200] for e in errors if e]
    if run.findings is None and errors and not run.error:
        run.error = errors[-1]
    run.status, run.finished = "done", time.monotonic()


