#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "trl[kernels]==1.18.1",
#   "openenv==0.14.0",
#   "transformers==5.23.0",
#   "torch==2.15.0",
#   "peft==0.21.3",
#   "flash-linear-attention==0.6.0",
#   "datasets>=4.0.0",
#   "trackio",
#   "aiohttp>=3.13.3",
#   "requests",
#   "httpx>=0.27.0",
#   "huggingface-hub>=1.13.0",
#   "opencode-ai==0.1.0a36",
#   "pydantic>=2.10.0,<2.13",
#   "safetensors>=0.4.0",
# ]
# ///
"""Harness-level GRPO for the paper-URL agent: real OpenCode, OpenEnv capture proxy, TRL LoRA adapters.

This is the "SFT -> RL" stage. The policy starts from the merged SFT checkpoint and trains a fresh
LoRA adapter, exactly as in the harness-adapter recipe:

    OpenCode (one `opencode serve` per rollout, unmodified)
      -> OpenEnv capture proxy (API key = capture session id; records engine token ids + logprobs)
      -> vLLM on GPU 0, serving the base weights plus every published adapter version by name
    TRL `HarnessAsyncGRPOTrainer` on GPU 1
      <- `TrainingTrace` (validated tokens, logprobs, loss masks) + deterministic URL verifier reward
      -> after each step, loads the new adapter into vLLM by name (no full-weight reload)

vLLM runs from its own isolated environment (`uvx --from vllm==VLLM_VERSION`) because current vLLM
and OpenEnv pin incompatible `mcp` majors; the trainer talks to it only over HTTP.

The URL task needs no sandbox: OpenCode only reads public web pages and runs `hf papers`, so each
rollout runs locally in its own temporary workspace with its own OpenCode database.

Reward: fraction of verified URL fields that are correct (0 without a structured answer), plus a
small tool-efficiency bonus `0.1 * (1 - tool_calls / 15)` only when every verified field is correct.

Run on a two-GPU node with `--launch` (starts vLLM on GPU 0, the trainer on GPU 1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_vllm import (  # noqa: E402 - shared harness definition with the evaluator
    ALLOWED_TOOLS,
    OPENCODE_TOOLS,
    SCOREABLE_QUALITIES,
    SYSTEM_PROMPT,
    URLFindings,
    candidate_from_row,
    findings_from_messages,
    install_opencode,
    raw_json,
    score_findings,
    tool_events,
    user_prompt,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("url-grpo")

DATASET_ID = "nielsr/paper-url-extraction-v1"
TEXT_LORA_TARGET_MODULES = [
    "down_proj",
    "gate_proj",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "k_proj",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
]
# Only the language model: the vision tower is frozen and vLLM serves the text model alone.
LORA_TARGET_REGEX = r".*language_model.*\.(" + "|".join(TEXT_LORA_TARGET_MODULES) + r")$"

VLLM_VERSION = "0.33.0"
W_TOOL_EFFICIENCY = 0.1
TOOL_BUDGET = 15.0


# --------------------------------------------------------------------------------------------------
# Task data and reward
# --------------------------------------------------------------------------------------------------


def any_requested_scoreable(row: dict[str, Any]) -> bool:
    return any(
        bool(row[f"missing_{name}"]) and row.get(f"{name}_label_quality") in SCOREABLE_QUALITIES
        for name in ("github", "project_page")
    )


def load_rows(dataset: str, split: str, exclude_ids: set[str], limit: int | None, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = [dict(row) for row in load_dataset(dataset, split=split)]
    rows = [row for row in rows if any_requested_scoreable(row) and str(row["arxiv_id"]) not in exclude_ids]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def correctness(row: dict[str, Any], findings: URLFindings | None) -> float:
    """Fraction of verified requested fields that match; unknown labels are never scored."""
    if findings is None:
        return 0.0
    scoring = score_findings(row, findings)
    fields = {name: ok for name, ok in scoring["fields"].items() if ok is not None}
    if not fields:
        return 0.0
    return sum(bool(ok) for ok in fields.values()) / len(fields)


# Rule-based shortcut guard, in the spirit of safeagents: the labels live in a public Hub dataset, so a
# policy rewarded for correct URLs could learn to look them up instead of researching. Any authored tool
# input matching these patterns scores 0. Matched against what the agent writes, never tool outputs.
SHORTCUT_PATTERNS = (
    re.compile(r"paper-url-extraction|pwc-url-extraction|\bnielsr/|community-science-llm-traces", re.IGNORECASE),
    re.compile(r"\bhf\s+download\b|load_dataset|snapshot_download|hf_hub_download"),
    re.compile(r"\bHF_TOKEN\b|\bhf\s+auth\b|huggingface-cli\s+login|/\.cache/huggingface/token"),
)


def shortcut_violation(events: list[Any]) -> str | None:
    for event in events:
        authored = json.dumps(getattr(event, "tool_input", {}) or {})
        for pattern in SHORTCUT_PATTERNS:
            if pattern.search(authored):
                return f"{event.tool_name}: {pattern.pattern[:40]}"
    return None


def url_reward(outcome) -> float | None:
    """Module-level so TRL can pickle it into the rollout process."""
    if outcome.env_reward is None:
        return None
    reward = float(outcome.env_reward)
    if reward >= 1.0 and outcome.tool_call_count is not None:
        reward += W_TOOL_EFFICIENCY * max(0.0, min(1.0, 1.0 - outcome.tool_call_count / TOOL_BUDGET))
    return reward


# --------------------------------------------------------------------------------------------------
# OpenEnv session: one OpenCode rollout through the capture proxy
# --------------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class UnconstrainedToolChoice:
    """ASGI middleware: rewrite `tool_choice: "required"` to `"auto"` before the capture proxy sees it.

    OpenCode forces a tool call on every turn when it requests structured output. A forced choice masks
    logits, so the engine's logprobs would not be the policy's and OpenEnv rightly refuses to train on
    them. Letting the policy choose keeps sampling unconstrained; a turn that answers in prose instead
    of calling `StructuredOutput` fails OpenCode's structured-output check and earns no reward.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST" or not scope.get("path", "").endswith(
            "/chat/completions"
        ):
            await self.app(scope, receive, send)
            return
        chunks = []
        more = True
        while more:
            message = await receive()
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        with suppress(ValueError, TypeError):
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("tool_choice") not in (None, "auto", "none"):
                payload["tool_choice"] = "auto"
                body = json.dumps(payload).encode()
        headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        scope = {**scope, "headers": headers}
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class OpenCodeCaptureSession:
    """A loop-owning OpenEnv session (`TrainableSession`): OpenCode drives, the proxy records."""

    def __init__(self, factory: OpenCodeCaptureFactory, row: dict[str, Any], episode_id: str) -> None:
        from openenv.core.harness import ToolResult  # noqa: F401 - validates the OpenEnv install early

        self.factory = factory
        self.row = row
        self.candidate = candidate_from_row(row)
        self.episode_id = episode_id
        self.workdir = Path(tempfile.mkdtemp(prefix="url-grpo-", dir=factory.scratch_dir))
        self.findings: URLFindings | None = None
        self.messages: list[dict[str, Any]] = []
        self.failure: str | None = None
        self.capture_id: str | None = None
        self.process: subprocess.Popen | None = None

    # ResourceSession surface ------------------------------------------------------------------
    def initial_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": user_prompt(self.candidate)}]

    def list_tools(self) -> list:
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]):
        from openenv.core.harness import ToolResult

        return ToolResult(error="OpenCode owns its own tool loop")

    def verify(self, transcript: list[dict[str, Any]], final_state: Any | None = None):
        from openenv.core.harness import VerifyResult

        if self.failure and self.failure.startswith("infrastructure:"):
            return VerifyResult(env_reward=None, done=True)
        violation = shortcut_violation(tool_events(self.messages))
        if violation:
            logger.warning("shortcut guard: %s on %s scores 0", violation, self.candidate.arxiv_id)
            return VerifyResult(env_reward=0.0, done=True)
        return VerifyResult(env_reward=correctness(self.row, self.findings), done=True)

    def close(self) -> None:
        if self.process is not None:
            # OpenCode spawns workers (LSP, shell tools); kill its whole process group, not just the parent.
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        if self.capture_id is not None:
            with suppress(Exception):
                self.factory.capture.registry.delete(self.capture_id)
        shutil.rmtree(self.workdir, ignore_errors=True)

    # Loop-owning extensions ---------------------------------------------------------------------
    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        try:
            timed_out = asyncio.run(self._run(timeout_s or self.factory.episode_timeout))
        except Exception as exc:  # noqa: BLE001 - infrastructure failures make the rollout unscorable
            self.failure = f"infrastructure: {type(exc).__name__}: {exc}"
            logger.warning("rollout %s failed: %s", self.candidate.arxiv_id, self.failure)
            return 1
        # Validate the capture now: a failed rollout (exit code 1) is dropped as unscorable, while an
        # invalid capture from a "successful" one would stop the whole trainer.
        try:
            self.fetch_training_trace()
        except Exception as exc:  # noqa: BLE001
            session = self.factory.capture.registry.get(self.capture_id)
            findings = []
            if session is not None:
                from openenv.core.harness.capture.export import export_session

                level = session.capture_level or self.factory.capture.capture_level
                findings = export_session(session, capture_level=level)["validation"]
            self.failure = f"infrastructure: untrainable capture: {exc}"
            logger.warning(
                "rollout %s dropped: %s; findings=%s; opencode_errors=%s",
                self.candidate.arxiv_id,
                exc,
                findings[:6],
                self._provider_errors()[:2],
            )
            return 1
        if timed_out:
            raise TimeoutError(self.failure or "episode timed out")
        return 0

    def _provider_errors(self) -> list[str]:
        errors = []
        for message in self.messages:
            error = (message.get("info") or {}).get("error")
            if error:
                errors.append(str((error.get("data") or {}).get("message") or error)[:300])
        return errors

    def fetch_training_trace(self):
        from openenv.core.harness.capture.contract import to_training_trace
        from openenv.core.harness.capture.export import export_session

        session = self.factory.capture.registry.get(self.capture_id)
        if session is None:
            raise ValueError("capture session disappeared")
        level = session.capture_level or self.factory.capture.capture_level
        document = export_session(session, capture_level=level)
        return to_training_trace(session.graph, document)

    # Internals ----------------------------------------------------------------------------------
    def _mint_capture_session(self) -> str:
        payload = {
            "llm_url": self.factory.llm_url,
            "model": self.factory.model,
            "max_model_calls": self.factory.max_model_calls,
            "purpose": self.factory.purpose,
            "metadata": {"arxiv_id": self.candidate.arxiv_id, "episode_id": self.episode_id},
        }
        if self.factory.sampling:
            payload["sampling"] = self.factory.sampling
        if self.factory.upstream_api_key_env:
            payload["api_key"] = os.environ[self.factory.upstream_api_key_env]
        response = httpx.post(f"{self.factory.capture_url}/sessions", json=payload, timeout=120)
        response.raise_for_status()
        body = response.json()
        if self.factory.purpose == "train" and body.get("rollout_type") != "train":
            raise RuntimeError(f"capture session is not trainable: {body}")
        return str(body["session_id"])

    def _start_opencode(self, session_id: str) -> str:
        port = free_port()
        limit = {"context": self.factory.max_model_len, "output": self.factory.max_output_tokens}
        config = {
            "$schema": "https://opencode.ai/config.json",
            "share": "disabled",
            "autoupdate": False,
            "provider": {
                "capture": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "OpenEnv capture proxy",
                    "options": {"baseURL": f"{self.factory.capture_url}/v1", "apiKey": session_id},
                    "models": {"policy": {"name": "policy", "limit": limit}},
                }
            },
            "small_model": "capture/policy",
        }
        env = {
            **os.environ,
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            # Separate database and state per rollout; the provider-package cache stays shared.
            "XDG_DATA_HOME": str(self.workdir / "data"),
            "XDG_STATE_HOME": str(self.workdir / "state"),
            "XDG_CONFIG_HOME": str(self.workdir / "config"),
        }
        workspace = self.workdir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        # Many OpenCode servers cold-starting at once crash or stall (shared provider-package cache);
        # start them one at a time. Startup takes ~1-2 s, the episode itself runs concurrently.
        with self.factory.startup_lock:
            return self._spawn_opencode(port, env, workspace)

    def _spawn_opencode(self, port: int, env: dict[str, str], workspace: Path) -> str:
        self.process = subprocess.Popen(
            [self.factory.opencode_binary, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
            cwd=workspace,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"OpenCode exited during startup ({self.process.returncode})")
            with suppress(httpx.HTTPError):
                if httpx.get(f"{url}/global/health", timeout=1).is_success:
                    return url
            time.sleep(0.2)
        raise TimeoutError("OpenCode did not become healthy within 120 seconds")

    async def _run(self, episode_timeout: float) -> bool:
        from opencode_ai import AsyncOpencode

        self.capture_id = self._mint_capture_session()
        base_url = self._start_opencode(self.capture_id)
        timed_out = False
        async with AsyncOpencode(base_url=base_url, timeout=httpx.Timeout(episode_timeout + 60, connect=10)) as client:
            session = await raw_json(
                client,
                "post",
                "/session",
                {
                    "title": f"URL RL {self.candidate.arxiv_id}",
                    "permission": [
                        {"permission": "*", "pattern": "*", "action": "allow"},
                        {"permission": "external_directory", "pattern": "*", "action": "deny"},
                        {"permission": "doom_loop", "pattern": "*", "action": "deny"},
                    ],
                },
            )
            session_id = session["id"]
            response: dict[str, Any] = {}
            try:
                async with asyncio.timeout(episode_timeout):
                    response = await raw_json(
                        client,
                        "post",
                        f"/session/{session_id}/message",
                        {
                            "model": {"providerID": "capture", "modelID": "policy"},
                            "system": SYSTEM_PROMPT,
                            "tools": {name: name in ALLOWED_TOOLS for name in OPENCODE_TOOLS},
                            "format": {
                                "type": "json_schema",
                                "schema": URLFindings.model_json_schema(),
                                "retryCount": 2,
                            },
                            "parts": [{"type": "text", "text": user_prompt(self.candidate)}],
                        },
                    )
            except TimeoutError:
                timed_out = True
                self.failure = f"episode timed out after {episode_timeout:g} seconds"
                with suppress(Exception):
                    await raw_json(client, "post", f"/session/{session_id}/abort")
            self.messages = await raw_json(client, "get", f"/session/{session_id}/message")
        structured = (response.get("info") or {}).get("structured")
        if structured is not None:
            with suppress(Exception):
                self.findings = URLFindings.model_validate(structured)
        if self.findings is None:
            self.findings = findings_from_messages(self.messages)
        self.tool_calls = len(tool_events(self.messages))
        return timed_out


class OpenCodeCaptureFactory:
    """`ResourceSessionFactory` for TRL. Picklable: the capture proxy starts lazily in the rollout process."""

    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        llm_url: str,
        model: str,
        opencode_binary: str,
        sampling: dict[str, Any] | None = None,
        episode_timeout: float = 180.0,
        max_model_calls: int = 16,
        max_model_len: int = 65536,
        max_output_tokens: int = 1024,
        scratch_dir: str | None = None,
        purpose: str = "train",
        upstream_api_key_env: str | None = None,
    ) -> None:
        self.purpose = purpose
        # Name of an environment variable, never the credential itself: the factory is pickled.
        self.upstream_api_key_env = upstream_api_key_env
        self.rows = {user_prompt(candidate_from_row(row)): row for row in rows}
        self.llm_url = llm_url.rstrip("/")
        self.model = model
        self.opencode_binary = opencode_binary
        self.sampling = sampling
        self.episode_timeout = episode_timeout
        self.max_model_calls = max_model_calls
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.scratch_dir = scratch_dir
        self._capture = None
        self._capture_url: str | None = None
        self._lock = threading.Lock()
        self.startup_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_capture"] = None
        state["_capture_url"] = None
        state["_lock"] = None
        state["startup_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self.startup_lock = threading.Lock()

    def _ensure_capture(self) -> None:
        with self._lock:
            if self._capture is None:
                self._start_capture()

    def _start_capture(self) -> None:
        from openenv.core.harness.capture.runner import CaptureServer

        port = free_port()
        capture = CaptureServer(
            llm_url=self.llm_url,
            model=self.model,
            port=port,
            max_output_tokens=self.max_output_tokens,
            max_model_calls=self.max_model_calls,
        )
        capture.app.add_middleware(UnconstrainedToolChoice)
        capture.start()
        self._capture = capture
        self._capture_url = f"http://127.0.0.1:{port}"
        logger.warning("capture proxy listening on %s -> %s", self._capture_url, self.llm_url)

    @property
    def capture(self):
        self._ensure_capture()
        return self._capture

    @property
    def capture_url(self) -> str:
        self._ensure_capture()
        return str(self._capture_url)

    def prompt_rows(self) -> list[dict[str, Any]]:
        return [
            {"prompt": [{"role": "user", "content": prompt}], "arxiv_id": str(row["arxiv_id"])}
            for prompt, row in self.rows.items()
        ]

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> OpenCodeCaptureSession:
        if isinstance(task, list) and task:
            prompt = str(task[-1].get("content") or "")
        elif isinstance(task, dict):
            prompt = str(task.get("content") or task.get("prompt") or "")
        else:
            prompt = str(task)
        row = self.rows.get(prompt)
        if row is None:
            raise KeyError("prompt does not match any training paper")
        self._ensure_capture()
        return OpenCodeCaptureSession(self, row, episode_id or f"ep-{random.getrandbits(48):x}")


def make_factory(*, sampling: dict[str, Any] | None = None, **kwargs: Any) -> OpenCodeCaptureFactory:
    return OpenCodeCaptureFactory(sampling=sampling, **kwargs)


# --------------------------------------------------------------------------------------------------
# Training and two-GPU launch
# --------------------------------------------------------------------------------------------------


def wait_for_vllm(url: str, process: subprocess.Popen, timeout_s: float = 2400) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {process.returncode}")
        with suppress(Exception):
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy")


def start_vllm(args: argparse.Namespace) -> subprocess.Popen:
    command = [
        shutil.which("uvx") or "uvx",
        "--python",
        "3.12",
        "--from",
        f"vllm=={VLLM_VERSION}",
        "vllm",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.vllm_port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--return-tokens-as-token-ids",
        "--logprobs-mode",
        "processed_logprobs",
        "--weight-transfer-config",
        '{"backend":"nccl"}',
        "--enable-lora",
        "--max-lora-rank",
        str(args.lora_r),
        "--max-loras",
        str(args.max_staleness + 2),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.vllm_device,
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
    }
    print("[launch] " + " ".join(command), flush=True)
    return subprocess.Popen(command, env=env, start_new_session=True)


def train(args: argparse.Namespace) -> None:
    from functools import partial

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import AsyncGRPOConfig
    from trl.experimental.async_grpo.openenv_harness import HarnessAsyncGRPOTrainer

    exclude: set[str] = set()
    if args.exclude_ids_file:
        exclude = {line.strip() for line in Path(args.exclude_ids_file).read_text().splitlines() if line.strip()}
    rows = load_rows(args.dataset, args.split, exclude, args.n_prompts, args.seed)
    vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    factory = partial(
        make_factory,
        rows=rows,
        llm_url=vllm_url,
        model=args.model,
        opencode_binary=str(args.opencode_binary),
        episode_timeout=args.episode_timeout,
        max_model_calls=args.max_model_calls,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_completion_length,
        scratch_dir=args.scratch_dir,
    )
    dataset = Dataset.from_list(factory().prompt_rows())
    print(f"[train] {len(dataset)} papers, model {args.model}, output {args.output_dir}", flush=True)

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        score_centering=False,
        max_concurrent_rollouts=args.max_concurrent_rollouts,
        max_staleness=args.max_staleness,
        vllm_server_base_url=vllm_url,
        rollout_worker_timeout=args.episode_timeout + 300,
        optim="adamw_torch",
        dtype="bfloat16",
        bf16=True,
        activation_checkpointing=True,
        activation_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        project="url-agent-grpo",
        trackio_space_id=args.trackio_space_id,
        run_name=args.run_name,
        log_completions=False,
        logging_steps=1,
        seed=args.seed,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules=LORA_TARGET_REGEX,
    )
    trainer = HarnessAsyncGRPOTrainer(
        model=args.model,
        harness_session_factory=factory,
        train_dataset=dataset,
        args=config,
        processing_class=AutoTokenizer.from_pretrained(args.model),
        rollout_reward_fn=url_reward,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.hub_model_id:
        trainer.model.push_to_hub(args.hub_model_id, private=True)
        print(f"[train] pushed adapter to {args.hub_model_id}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch", action="store_true", help="Start vLLM on --vllm-device, then train")
    parser.add_argument("--model", default="nielsr/qwen35-4b-url-agent-merged")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--exclude-ids-file", help="arXiv IDs never to train on (one per line)")
    parser.add_argument("--n-prompts", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=16)
    parser.add_argument("--max-staleness", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-model-calls", type=int, default=16)
    parser.add_argument("--episode-timeout", type=float, default=180.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-device", default="0")
    parser.add_argument("--train-device", default="1")
    parser.add_argument("--opencode-binary", type=Path, default=None)
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--output-dir", default="runs/url-grpo")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--report-to", default="trackio")
    parser.add_argument("--trackio-space-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.opencode_binary is None:
        found = shutil.which("opencode")
        args.opencode_binary = Path(found) if found else install_opencode(Path(tempfile.mkdtemp()) / "opencode")
    if not args.launch:
        train(args)
        return
    vllm = start_vllm(args)
    try:
        wait_for_vllm(f"http://127.0.0.1:{args.vllm_port}", vllm)
        print("[launch] vLLM ready", flush=True)
        child_args = [arg for arg in sys.argv[1:] if arg != "--launch"]
        child_args += ["--opencode-binary", str(args.opencode_binary)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.train_device}
        code = subprocess.call([sys.executable, __file__, *child_args], env=env)
        if code:
            raise SystemExit(code)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(vllm.pid), 15)


if __name__ == "__main__":
    main()
