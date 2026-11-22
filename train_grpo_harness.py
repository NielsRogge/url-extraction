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

Reward (`harness.scoring`, the same `score_episode` the evaluator records): fraction of verified URL fields
that are correct (0 without a structured answer), with `--protocol-weight` of that credit withheld when the
episode is incomplete (research protocol skipped or `--max-tool-calls` reached), plus a small tool-efficiency
bonus `0.1 * (1 - tool_calls / 15)` only on full credit (every verified field correct and, with a protocol
weight, a complete episode).

Run on a two-GPU node with `--launch` (starts vLLM on GPU 0, the trainer on GPU 1).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.opencode import (  # noqa: E402 - the shared harness (same episode code as the evaluator)
    SESSION_PERMISSIONS,
    find_or_install_opencode,
    message_body,
    parse_findings,
    raw_json,
    tool_events,
)
from harness.scoring import (  # noqa: E402
    TOOL_BUDGET,  # noqa: F401 - re-exported
    W_TOOL_EFFICIENCY,  # noqa: F401 - re-exported
    env_reward,
    partly_scoreable,
    score_episode,
    shortcut_violation,
    tool_efficiency_bonus,
)
from harness.task import URLFindings, candidate_from_row, user_prompt  # noqa: E402

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


# --------------------------------------------------------------------------------------------------
# Task data and reward
# --------------------------------------------------------------------------------------------------


def any_requested_scoreable(row: dict[str, Any]) -> bool:
    return partly_scoreable(row)


def load_rows(
    dataset: str,
    split: str,
    exclude_ids: set[str],
    limit: int | None,
    seed: int,
    include_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = [dict(row) for row in load_dataset(dataset, split=split)]
    rows = [row for row in rows if any_requested_scoreable(row) and str(row["arxiv_id"]) not in exclude_ids]
    if include_ids is not None:
        rows = [row for row in rows if str(row["arxiv_id"]) in include_ids]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def weight_by_tier(rows: list[dict[str, Any]], exclude: str, weights: str, seed: int) -> list[dict[str, Any]]:
    """Drop excluded difficulty tiers and repeat harder tiers (e.g. "3:2,5:3") so RL sees them more often."""
    drop = {int(t) for t in exclude.split(",") if t.strip()}
    repeat = {int(k): int(v) for k, v in (item.split(":") for item in weights.split(",") if item.strip())}
    out = []
    for row in rows:
        tier = row.get("difficulty_tier")
        if tier is not None and int(tier) in drop:
            continue
        out.extend([row] * repeat.get(int(tier) if tier is not None else -1, 1))
    random.Random(seed).shuffle(out)
    tiers = collections.Counter(r.get("difficulty_tier") for r in out)
    print(f"[train] prompt pool by difficulty tier (after weighting): {dict(sorted(tiers.items(), key=lambda kv: str(kv[0])))}", flush=True)
    return out


def url_reward(outcome) -> float | None:
    """Module-level so TRL can pickle it into the rollout process: env reward plus the tool-efficiency bonus."""
    if outcome.env_reward is None:
        return None
    return tool_efficiency_bonus(float(outcome.env_reward), outcome.tool_call_count)


def episode_reward(
    row: dict[str, Any],
    findings: URLFindings | None,
    messages: list[dict[str, Any]],
    *,
    protocol_weight: float,
    max_tool_calls: int | None,
) -> float:
    """Environment reward of one finished rollout, from the same `score_episode` the evaluator records."""
    events = tool_events(messages)
    violation = shortcut_violation(events)
    if violation:
        logger.warning("shortcut guard: %s on %s scores 0", violation, row.get("arxiv_id"))
    score = score_episode(row, findings, events, max_tool_calls=max_tool_calls)
    return env_reward(score, events, protocol_weight=protocol_weight)


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
        # Correct URLs found without following the research protocol earn only part of the credit, so the
        # tool-efficiency bonus cannot be bought by skipping the paper or metadata checks.
        score = episode_reward(
            self.row,
            self.findings,
            self.messages,
            protocol_weight=self.factory.protocol_weight,
            max_tool_calls=self.factory.max_tool_calls,
        )
        return VerifyResult(env_reward=score, done=True)

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
                {"title": f"URL RL {self.candidate.arxiv_id}", "permission": SESSION_PERMISSIONS},
            )
            session_id = session["id"]
            response: dict[str, Any] = {}
            try:
                async with asyncio.timeout(episode_timeout):
                    response = await raw_json(
                        client,
                        "post",
                        f"/session/{session_id}/message",
                        message_body(("capture", "policy"), self.candidate),
                    )
            except TimeoutError:
                timed_out = True
                self.failure = f"episode timed out after {episode_timeout:g} seconds"
                with suppress(Exception):
                    await raw_json(client, "post", f"/session/{session_id}/abort")
            self.messages = await raw_json(client, "get", f"/session/{session_id}/message")
        self.findings, _ = parse_findings(response, self.messages)
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
        protocol_weight: float = 0.0,
        max_tool_calls: int | None = 15,
    ) -> None:
        self.purpose = purpose
        self.max_tool_calls = max_tool_calls
        self.protocol_weight = protocol_weight
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
    launcher = (
        [args.vllm_binary]
        if args.vllm_binary
        else [shutil.which("uvx") or "uvx", "--python", "3.12", "--from", f"vllm=={VLLM_VERSION}", "vllm"]
    )
    command = [
        *launcher,
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
        args.tool_call_parser,
        "--return-tokens-as-token-ids",
        "--logprobs-mode",
        "processed_logprobs",
        "--enable-lora",
        "--max-lora-rank",
        str(args.lora_r),
        "--max-loras",
        str(args.max_staleness + 2),
    ]
    if args.trainer == "trl":
        # TRL's merged-weight fallback path; adapter-only trainers never need it.
        command += ["--weight-transfer-config", '{"backend":"nccl"}']
    if args.reasoning_parser:
        command += ["--reasoning-parser", args.reasoning_parser]
    if args.dspark:
        # The checkpoint's bundled DSpark drafter (vLLM recipe); probabilistic drafting keeps the target distribution.
        command += [
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}',
        ]
    if args.vllm_extra_args:
        import shlex

        command += shlex.split(args.vllm_extra_args)
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
    include = None
    if args.include_ids_file:
        include = {line.strip() for line in Path(args.include_ids_file).read_text().splitlines() if line.strip()}
    rows = load_rows(args.dataset, args.split, exclude, args.n_prompts, args.seed, include)
    rows = weight_by_tier(rows, args.exclude_tiers, args.tier_weights, args.seed)
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
        protocol_weight=args.protocol_weight,
        max_tool_calls=args.max_tool_calls,
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
        # Push every saved checkpoint as it is written (latest under `last-checkpoint/`, resumable), so a
        # crash, a full disk or the end of the job never loses the run.
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
        hub_private_repo=not args.public,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules=args.lora_target_modules or LORA_TARGET_REGEX,
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
    if hasattr(trainer, "_lora_dir"):
        # TRL (1.18) saves every adapter version it hot-loads into vLLM under <output_dir>/.vllm_lora, and the
        # checkpoint pushes then upload them to the Hub repo. Keep them outside the output directory instead
        # (vLLM runs on this machine and reads them by absolute path).
        trainer._lora_dir = str(Path(args.scratch_dir or tempfile.gettempdir()).resolve() / "url-grpo-vllm-lora")
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.hub_model_id:
        trainer.model.push_to_hub(args.hub_model_id, private=not args.public)
        print(f"[train] pushed adapter to {args.hub_model_id}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch", action="store_true", help="Start vLLM on --vllm-device, then train")
    parser.add_argument("--model", default="nielsr/qwen35-4b-url-agent-merged")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--exclude-ids-file", help="arXiv IDs never to train on (one per line)")
    parser.add_argument("--include-ids-file", help="Only train on these arXiv IDs (one per line)")
    parser.add_argument(
        "--protocol-weight",
        type=float,
        default=0.0,
        help="Fraction of correctness credit withheld when the research protocol was not followed",
    )
    parser.add_argument("--exclude-tiers", default="", help="Difficulty tiers to drop, e.g. '7' (labels found nowhere)")
    parser.add_argument("--tier-weights", default="", help="Repeat factors per tier, e.g. '3:2,4:2,5:3,6:2'")
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
    parser.add_argument("--public", action="store_true", help="Publish the adapter and checkpoints publicly")
    parser.add_argument("--report-to", default="trackio")
    parser.add_argument("--trackio-space-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=15,
        help="Rollouts with this many tool calls are incomplete, as in evaluation (protocol penalty applies)",
    )
    # Student-specific settings; `pipeline.py` fills them from the student profile.
    parser.add_argument(
        "--trainer", choices=["trl", "simple"], default="trl", help="GRPO backend ('simple': models TRL cannot pack)"
    )
    parser.add_argument("--vllm-binary", help="Serve with this vLLM executable (e.g. a model-specific image) instead of uvx")
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument("--reasoning-parser", default="")
    parser.add_argument("--vllm-extra-args", default="", help="Extra arguments appended to `vllm serve` (shell-quoted)")
    parser.add_argument("--prompts-per-step", type=int, default=4, help="simple trainer: papers per optimizer step")
    parser.add_argument("--sync-every", type=int, default=1, help="simple trainer: steps between adapter syncs")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.28)
    parser.add_argument("--max-train-tokens", type=int, default=49152, help="simple trainer: longest trained sequence")
    parser.add_argument("--attention-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--offload-activations", action=argparse.BooleanOptionalAction, default=True,
        help="simple trainer: keep activations saved for backward in host memory",
    )
    parser.add_argument("--dspark", action="store_true", help="Serve with the checkpoint's DSpark speculative drafter")
    parser.add_argument("--log-rollouts", action="store_true", help="simple trainer: one diagnostic line per rollout")
    parser.add_argument("--card-base-model", help="Hub repo id to name as base_model in adapter cards (when --model is a local path)")
    parser.add_argument("--lora-target-modules", default="", help="LoRA target-module regex (default: Qwen3.5 text)")
    return parser.parse_args()


# --------------------------------------------------------------------------------------------------
# Simple synchronous GRPO (models TRL's tree-packed async trainer cannot run, e.g. Inkling 2)
# --------------------------------------------------------------------------------------------------
#
# TRL's AsyncGRPOTrainer folds rollouts into a prefix trie with FlashAttention-3 tree attention. It rejects
# models with attention sinks or with convolutional/recurrent layers other than gated DeltaNet, which rules out
# Inkling 2 (per-token sinks, causal short convolutions). This loop keeps everything else of the recipe:
# the same OpenCode rollouts through the OpenEnv capture proxy, the same reward, LoRA adapters hot-loaded into
# vLLM by name. Each step samples `--prompts-per-step` papers x `--num-generations` rollouts from the current
# policy, normalizes rewards within each paper, and takes one optimizer step on the clipped importance-weighted
# policy gradient (token-level normalization, DAPO-style asymmetric clipping) using the engine's own logprobs as
# the behaviour policy. Groups whose rollouts all score the same carry no signal and are skipped.


def rollout_chains(trace: Any, max_tokens: int) -> tuple[list[tuple[list[int], list[int], list[float]]], int]:
    """Fold a rollout's engine calls into as few forward passes as possible.

    A call whose prompt extends the previous call's prompt + completion continues that sequence; otherwise a new
    sequence starts (e.g. when the harness drops earlier reasoning from the history). Returns
    `(input_ids, trained_positions, behaviour_logprobs)` per sequence, plus the number of trained tokens dropped
    because their sequence exceeds `max_tokens`.
    """
    chains: list[tuple[list[int], list[int], list[float]]] = []
    ids: list[int] = []
    positions: list[int] = []
    old: list[float] = []
    dropped = 0

    def flush() -> None:
        nonlocal dropped
        if positions:
            if len(ids) <= max_tokens:
                chains.append((list(ids), list(positions), list(old)))
            else:
                dropped += len(positions)

    for turn in trace.turns:
        prompt, completion = list(turn.prompt_token_ids), list(turn.completion_token_ids)
        if not (ids and prompt[: len(ids)] == ids):
            flush()
            positions, old = [], []
        ids = prompt + completion
        mask = turn.loss_mask[len(prompt) :]
        for offset, (keep, logprob) in enumerate(zip(mask, turn.per_token_logps)):
            if keep:
                positions.append(len(prompt) + offset)
                old.append(float(logprob))
    flush()
    return chains, dropped


def train_simple(args: argparse.Namespace) -> None:
    import math
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from huggingface_hub import HfApi
    from openenv.core.harness.capture.upstream import training_sampling
    from peft import LoraConfig, get_peft_model

    from inkling_adapter import convert_adapter
    from train_sft import INKLING_LORA_TARGET_MODULES, is_inkling, load_inkling_model

    if not is_inkling(args.model):
        raise ValueError("--trainer simple is wired for Inkling 2 (inkling2_loop) models")
    exclude: set[str] = set()
    if args.exclude_ids_file:
        exclude = {line.strip() for line in Path(args.exclude_ids_file).read_text().splitlines() if line.strip()}
    include = None
    if args.include_ids_file:
        include = {line.strip() for line in Path(args.include_ids_file).read_text().splitlines() if line.strip()}
    rows = load_rows(args.dataset, args.split, exclude, args.n_prompts, args.seed, include)
    rows = weight_by_tier(rows, args.exclude_tiers, args.tier_weights, args.seed)
    vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    sampling = training_sampling(
        {"temperature": args.temperature, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
    )
    factory = make_factory(
        sampling=sampling,
        rows=rows,
        llm_url=vllm_url,
        model=args.model,
        opencode_binary=str(args.opencode_binary),
        episode_timeout=args.episode_timeout,
        max_model_calls=args.max_model_calls,
        max_model_len=args.max_model_len,
        max_output_tokens=args.max_completion_length,
        scratch_dir=args.scratch_dir,
        protocol_weight=args.protocol_weight,
    )
    print(f"[simple] {len(rows)} papers, model {args.model}, output {args.output_dir}", flush=True)

    device = torch.device("cuda")
    model = load_inkling_model(args.model, args.attention_chunk_size).to(device)
    model = get_peft_model(
        model,
        LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, target_modules=INKLING_LORA_TARGET_MODULES),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi() if args.hub_model_id else None
    if api:
        api.create_repo(args.hub_model_id, private=not args.public, exist_ok=True)

    def publish_adapter(step: int) -> str:
        """Save the PEFT adapter, convert it to vLLM's names and hot-load it as `policy-v{step}`."""
        raw = output / f"checkpoint-{step}"
        model.save_pretrained(raw)
        served = convert_adapter(raw, output / "served" / f"policy-v{step}")
        name = f"policy-v{step}"
        response = httpx.post(
            f"{vllm_url}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(served)}, timeout=600
        )
        response.raise_for_status()
        stale = f"policy-v{step - (args.max_staleness + 1) * args.sync_every}"
        with suppress(Exception):
            httpx.post(f"{vllm_url}/v1/unload_lora_adapter", json={"lora_name": stale}, timeout=120)
        return name

    def describe(session: Any, started: float, **extra: Any) -> str:
        """One-line rollout diagnostics: outcome, and per model call its duration, tokens and finish reason."""
        calls = []
        for message in session.messages:
            info = message.get("info") or {}
            if info.get("role") != "assistant":
                continue
            times = info.get("time") or {}
            tokens = info.get("tokens") or {}
            parts = [part.get("type") for part in message.get("parts", []) if part.get("type") not in ("step-start", "step-finish")]
            calls.append({
                "s": round(((times.get("completed") or times.get("created") or 0) - (times.get("created") or 0)) / 1000, 1),
                "out": tokens.get("output"), "reasoning": tokens.get("reasoning"), "finish": info.get("finish"),
                "parts": parts, "error": str(((info.get("error") or {}).get("data") or {}).get("message") or "")[:160] or None,
            })
        return json.dumps({
            "arxiv": session.candidate.arxiv_id, "secs": round(time.monotonic() - started, 1), "failure": session.failure,
            "findings": session.findings.model_dump(mode="json") if session.findings else None, "calls": calls, **extra,
        })[:4000]

    def run_rollout(row: dict[str, Any]) -> dict[str, Any] | None:
        session = factory.create([{"role": "user", "content": user_prompt(candidate_from_row(row))}])
        started = time.monotonic()
        try:
            try:
                if session.wait_for_completion():
                    print("[rollout] dropped " + describe(session, started), flush=True)
                    return None
            except TimeoutError:
                pass  # a timed-out episode with a valid capture still scores (usually 0): long loops get no credit
            trace = session.fetch_training_trace()
            verdict = session.verify([])
            if verdict.env_reward is None:
                return None
            reward = float(verdict.env_reward)
            tool_calls = int(getattr(session, "tool_calls", 0) or 0)
            if reward >= 1.0:
                reward += W_TOOL_EFFICIENCY * max(0.0, min(1.0, 1.0 - tool_calls / TOOL_BUDGET))
            if args.log_rollouts:
                print("[rollout] " + describe(session, started, reward=round(reward, 3), turns=len(trace.turns)), flush=True)
            return {
                "reward": reward, "trace": trace, "tool_calls": tool_calls, "failure": session.failure,
                "seconds": time.monotonic() - started,
            }
        except Exception as exc:  # noqa: BLE001 - unscorable rollouts are dropped, never fatal
            logger.warning("rollout %s dropped: %s", row.get("arxiv_id"), exc)
            with suppress(Exception):
                print("[rollout] error " + describe(session, started, error=str(exc)[:300]), flush=True)
            return None
        finally:
            session.close()

    def logprobs_of(ids: list[int], positions: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([ids], device=device)
        keep = torch.tensor([p - 1 for p in positions], device=device)
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=keep).logits[0].float()
        if args.temperature != 1.0:
            logits = logits / args.temperature
        targets = torch.tensor([ids[p] for p in positions], device=device)
        return torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None])[:, 0]

    card_base = args.card_base_model or (args.model if not Path(args.model).exists() else None)

    def write_card(folder: Path, step: int | None) -> None:
        """PEFT's card names the local base path, which the Hub rejects as `base_model`; name the Hub repo instead."""
        lines = ["---", "library_name: peft", "tags:", "- grpo", "- inkling2", "- lora"]
        if card_base:
            lines.insert(1, f"base_model: {card_base}")
        title = f"Inkling-2-Mini URL agent GRPO adapter{f' (step {step})' if step else ''}"
        (folder / "README.md").write_text("\n".join(lines + ["---", "", f"# {title}", "",
            f"Attention-only LoRA trained with GRPO (`train_grpo_harness.py --trainer simple`) on top of `{card_base}`.", ""]))

    def push(folder: Path, path_in_repo: str, required: str) -> None:
        """Upload and verify; a failed push is reported but never ends the run."""
        try:
            api.upload_folder(repo_id=args.hub_model_id, folder_path=str(folder), path_in_repo=path_in_repo,
                              allow_patterns=["adapter_*", "README.md", "metrics.jsonl"])
            landed = {f.rfilename for f in api.model_info(args.hub_model_id).siblings}
            if required not in landed:
                raise RuntimeError(f"{required} did not land")
            print(f"[simple] pushed {path_in_repo or 'final adapter'} to {args.hub_model_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[simple] WARNING: push of {path_in_repo or 'final adapter'} failed: {exc}", flush=True)

    metrics_path = output / "metrics.jsonl"
    cursor = 0
    random.Random(args.seed).shuffle(rows)
    policy = args.model
    with ThreadPoolExecutor(max_workers=args.max_concurrent_rollouts) as pool:
        for step in range(1, args.max_steps + 1):
            started = time.monotonic()
            papers = [rows[(cursor + i) % len(rows)] for i in range(args.prompts_per_step)]
            cursor += args.prompts_per_step
            factory.model = policy
            futures = [(i, pool.submit(run_rollout, paper)) for i, paper in enumerate(papers) for _ in range(args.num_generations)]
            groups: dict[int, list[dict[str, Any]]] = {}
            for index, future in futures:
                result = future.result()
                if result is not None:
                    groups.setdefault(index, []).append(result)
            rollout_seconds = time.monotonic() - started
            rewards = [r["reward"] for group in groups.values() for r in group]
            samples: list[tuple[float, list[tuple[list[int], list[int], list[float]]]]] = []
            dropped = 0
            for group in groups.values():
                if len(group) < 2:
                    continue
                values = [r["reward"] for r in group]
                mean = sum(values) / len(values)
                std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
                if std < 1e-6:
                    continue
                for result in group:
                    chains, lost = rollout_chains(result["trace"], args.max_train_tokens)
                    dropped += lost
                    samples.append(((result["reward"] - mean) / (std + 1e-4), chains))
            total_tokens = sum(len(positions) for _, chains in samples for _, positions, _ in chains)
            max_seq_tokens = max((len(ids) for _, chains in samples for ids, _, _ in chains), default=0)
            stats = {"ratio_sum": 0.0, "ratio_max": 0.0, "ratio_min": math.inf, "clipped": 0, "loss": 0.0}
            model.train()
            if total_tokens:
                optimizer.zero_grad(set_to_none=True)
                for advantage, chains in samples:
                    for ids, positions, old in chains:
                        # The looped model saves 48 layer applications (with their block-residual states) per
                        # sequence for backward; keeping them in host memory bounds GPU memory to the weights
                        # plus one layer's recomputation.
                        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.offload_activations else nullcontext()
                        with offload:
                            new = logprobs_of(ids, positions)
                        old_t = torch.tensor(old, device=device)
                        ratio = torch.exp(new - old_t)
                        clipped = ratio.clamp(1 - args.epsilon, 1 + args.epsilon_high)
                        objective = torch.minimum(ratio * advantage, clipped * advantage)
                        loss = -objective.sum() / total_tokens
                        loss.backward()
                        with torch.no_grad():
                            stats["loss"] += float(loss)
                            stats["ratio_sum"] += float(ratio.sum())
                            stats["ratio_max"] = max(stats["ratio_max"], float(ratio.max()))
                            stats["ratio_min"] = min(stats["ratio_min"], float(ratio.min()))
                            stats["clipped"] += int((ratio != clipped).sum())
                        del new, ratio, clipped, objective, loss
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                optimizer.step()
                scheduler.step()
            else:
                grad_norm = 0.0
            if total_tokens and step % args.sync_every == 0:
                policy = publish_adapter(step)
            record = {
                "step": step,
                "reward_mean": sum(rewards) / len(rewards) if rewards else None,
                "rollouts": len(rewards),
                "trained_tokens": total_tokens,
                "dropped_tokens": dropped,
                "max_seq_tokens": max_seq_tokens,
                "importance_ratio_mean": stats["ratio_sum"] / total_tokens if total_tokens else None,
                "importance_ratio_min": stats["ratio_min"] if total_tokens else None,
                "importance_ratio_max": stats["ratio_max"] if total_tokens else None,
                "clip_fraction": stats["clipped"] / total_tokens if total_tokens else None,
                "loss": stats["loss"],
                "grad_norm": grad_norm,
                "policy": policy,
                "rollout_seconds": round(rollout_seconds, 1),
                "step_seconds": round(time.monotonic() - started, 1),
            }
            # Collapse indicators from the harness-adapters recipe: turns and episode length per rollout.
            finished = [r for group in groups.values() for r in group]
            record["tool_calls_mean"] = round(sum(r["tool_calls"] for r in finished) / len(finished), 2) if finished else None
            record["episode_seconds_mean"] = round(sum(r["seconds"] for r in finished) / len(finished), 1) if finished else None
            record["groups_with_signal"] = sum(
                1 for group in groups.values() if len(group) > 1 and max(r["reward"] for r in group) - min(r["reward"] for r in group) > 1e-6
            )
            print("[simple] " + json.dumps(record), flush=True)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if api and step % args.save_steps == 0:
                raw = output / f"checkpoint-{step}"
                if not raw.exists():
                    model.save_pretrained(raw)
                write_card(raw, step)
                shutil.copy(metrics_path, raw / "metrics.jsonl")
                push(raw, f"checkpoint-{step}", f"checkpoint-{step}/adapter_model.safetensors")
    model.save_pretrained(output)
    write_card(output, None)
    if api:
        push(output, "", "adapter_model.safetensors")


def main() -> None:
    args = parse_args()
    if args.opencode_binary is None:
        args.opencode_binary = Path(find_or_install_opencode())
    if not args.launch:
        if args.trainer == "simple":
            simple = globals().get("train_simple")
            if simple is None:
                raise SystemExit("--trainer simple needs the synchronous GRPO trainer (Inkling support, PR #9)")
            simple(args)
        else:
            train(args)
        return
    vllm = start_vllm(args)
    try:
        wait_for_vllm(f"http://127.0.0.1:{args.vllm_port}", vllm)
        print("[launch] vLLM ready", flush=True)
        child_args = [arg for arg in sys.argv[1:] if arg != "--launch"]
        child_args += ["--opencode-binary", str(args.opencode_binary)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.train_device}
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        code = subprocess.call([sys.executable, __file__, *child_args], env=env)
        if code:
            raise SystemExit(code)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(vllm.pid), 15)


if __name__ == "__main__":
    main()
