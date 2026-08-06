# /// script
# dependencies = [
#   "bitsandbytes>=0.49.0",
#   "datasets",
#   "kernels>=0.15.2,<0.16.0",
#   "trackio",
#   "vllm>=0.22.0",
#   "trl @ git+https://github.com/huggingface/trl.git@d242e584b6d45701b178d3d7c16394dab3a9798c",
#   "openenv @ git+https://github.com/huggingface/OpenEnv.git@401e8099644a81841a0668cc7362e9a6620155c3",
#   "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git@401e8099644a81841a0668cc7362e9a6620155c3#subdirectory=envs/opencode_env",
# ]
# ///
"""Train Qwen inside the real OpenCode harness for paper URL extraction.

Use ``--launch`` on a two-GPU node. GPU 0 serves vLLM and GPU 1 trains. The
implementation follows TRL's upstream ``examples/scripts/openenv/opencode.py``
but replaces DeepCoder with a deterministic paper-URL task and verifier.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from datasets import Dataset, load_dataset
from opencode_env import harness as oc_harness
from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.opencode_runtime import proxy_dir, proxy_source_path, proxy_trace_path
from opencode_env.sandbox.base import ExecResult, SandboxHandle
from opencode_env.task import OpenCodeTask
from openenv.core.harness import ResourceSession, ResourceSessionFactory, VerifyResult
from transformers import AutoTokenizer
from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import (
    HarnessRolloutOutcome,
    HarnessRolloutWorker,
    TraceEntry,
    has_tool_call,
)

DATASET_ID = "nielsr/paper-url-extraction-v1"
SCORABLE_QUALITIES = {"gold", "silver", "verified_null"}
OPENCODE_VERSION = "1.18.14"


class LocalBgJob:
    def __init__(self, popen: subprocess.Popen):
        self._process = popen

    @property
    def pid(self) -> int:
        return self._process.pid

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(str(exc)) from exc

    def kill(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            group = os.getpgid(self._process.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(group, signal.SIGTERM)
            self._process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass


class NoOpWeightTransfer:
    """One-step Qwen3.5 smoke shim; vLLM cannot reload its text-model weight names yet."""

    def init_weight_transfer(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def send_weights(self, iterator) -> None:
        for _ in iterator:
            pass

    def resume(self) -> None:
        pass

    def destroy(self) -> None:
        pass


class LocalSandboxHandle:
    """A per-rollout directory that remaps OpenEnv's /home/user paths."""

    def __init__(self, root: str, *, alias: str = "/home/user", envs: dict | None = None, cleanup: bool = False):
        self._root = root
        self._alias = alias
        self._cleanup = cleanup
        self._env = {**os.environ, "HOME": root, **(envs or {})}
        self._jobs: list[LocalBgJob] = []

    @property
    def sandbox_id(self) -> str:
        return self._root

    def _remap(self, value: str | None) -> str | None:
        return value if value is None else value.replace(self._alias, self._root)

    def exec(self, cmd: str, *, envs=None, cwd=None, timeout: float | None = 60) -> ExecResult:
        try:
            result = subprocess.run(
                ["bash", "-lc", self._remap(cmd)],
                cwd=self._remap(cwd) or self._root,
                env={**self._env, **(envs or {})},
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            return ExecResult(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired as exc:
            return ExecResult(exit_code=124, stdout=exc.stdout or "", stderr=f"timeout after {timeout}s")

    def start_bg(self, cmd: str, *, envs=None, cwd=None) -> LocalBgJob:
        expose_output = "interception.py" in cmd
        process = subprocess.Popen(
            ["bash", "-lc", self._remap(cmd)],
            cwd=self._remap(cwd) or self._root,
            env={**self._env, **(envs or {})},
            stdout=None if expose_output else subprocess.DEVNULL,
            stderr=None if expose_output else subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        job = LocalBgJob(process)
        self._jobs.append(job)
        return job

    def write_text(self, path: str, content: str) -> None:
        destination = Path(self._remap(path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)

    def read_text(self, path: str) -> str:
        return Path(self._remap(path)).read_text()

    def exists(self, path: str) -> bool:
        return Path(self._remap(path)).exists()

    def kill(self) -> None:
        for job in self._jobs:
            job.kill()
        self._jobs.clear()
        if self._cleanup:
            shutil.rmtree(self._root, ignore_errors=True)


class LocalSubprocessSandboxBackend:
    def __init__(self, root: str, *, alias: str = "/home/user"):
        self._root = root
        self._alias = alias
        self._template = os.path.join(root, "_template")

    def warmup(self) -> None:
        marker = os.path.join(self._template, ".opencode", "bin", "opencode")
        if os.path.exists(marker):
            return
        architecture = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
            platform.machine().lower()
        )
        if architecture is None:
            raise RuntimeError(f"unsupported OpenCode architecture: {platform.machine()}")
        archive_url = (
            f"https://github.com/anomalyco/opencode/releases/download/v{OPENCODE_VERSION}/"
            f"opencode-linux-{architecture}.tar.gz"
        )
        with urllib.request.urlopen(archive_url, timeout=180) as response:
            archive = response.read()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            binary = tar.extractfile("opencode")
            if binary is None:
                raise RuntimeError("OpenCode archive does not contain the opencode binary")
            payload = binary.read()
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        Path(marker).write_bytes(payload)
        os.chmod(marker, 0o755)

    def create(self, *, timeout_s: int = 900, envs=None, metadata=None) -> LocalSandboxHandle:
        name = (metadata or {}).get("episode_id") or uuid.uuid4().hex
        sandbox_dir = os.path.join(self._root, name)
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        os.makedirs(sandbox_dir, exist_ok=True)
        if os.path.isdir(self._template):
            subprocess.run(["cp", "-al", f"{self._template}/.", f"{sandbox_dir}/"], check=True)
        for subdir in ("workdir", "task", "logs/agent", "logs/verifier", ".config/opencode"):
            path = os.path.join(sandbox_dir, subdir)
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
        return LocalSandboxHandle(sandbox_dir, alias=self._alias, envs=envs, cleanup=True)


def instruction_id(instruction: str) -> str:
    return hashlib.sha1(instruction.encode()).hexdigest()


def normalize_url(value: Any, *, github: bool = False) -> str | None:
    if value is None:
        return None
    value = str(value).strip().strip("<>\"'")
    if not value or value.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    if "://" not in value:
        value = "https://" + value
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    host = (parts.hostname or "").lower().removeprefix("www.")
    if not host:
        return None
    path = parts.path.rstrip("/")
    if github:
        if host != "github.com":
            return None
        path = path.removesuffix(".git").lower()
        pieces = [piece for piece in path.split("/") if piece]
        if len(pieces) < 2:
            return None
        path = "/" + "/".join(pieces[:2])
    return urlunsplit(("https", host, path, "", ""))


def paper_instruction(row: dict) -> str:
    known = {
        "github_url": row.get("known_github_url"),
        "project_page_url": row.get("known_project_page_url"),
    }
    return (
        "Research the official GitHub repository and official project page for the paper below. Use the available "
        "webfetch and bash tools to inspect primary sources such as the arXiv page/PDF, author pages, GitHub, and "
        "the Hugging Face paper page. Do not guess from the title alone. Then write `/home/user/workdir/answer.json` "
        'with exactly this JSON shape: {"github_url": string|null, "project_page_url": string|null}. Use null '
        "only when research finds no URL. Repository URLs must be the repository root, without /tree or /issues.\n\n"
        f"arXiv ID: {row['arxiv_id']}\nTitle: {row['title']}\nAbstract: {row['abstract']}\n"
        f"Already known URLs (do not rediscover these): {json.dumps(known)}"
    )


def score_answer(answer: dict, target: dict) -> float:
    scores: list[float] = []
    for key, quality_key, github in (
        ("github_url", "github_label_quality", True),
        ("project_page_url", "project_page_label_quality", False),
    ):
        if target[quality_key] not in SCORABLE_QUALITIES:
            continue
        predicted = normalize_url(answer.get(key), github=github)
        expected = normalize_url(target[f"reference_{key}"], github=github)
        scores.append(float(predicted == expected))
    return sum(scores) / len(scores) if scores else 0.0


class PaperUrlVerifier:
    def __init__(self, targets: dict[str, dict]):
        self._targets = targets

    def __call__(self, sandbox: SandboxHandle, task: OpenCodeTask) -> VerifyResult:
        target = self._targets.get(instruction_id(task.instruction))
        if target is None or not sandbox.exists("/home/user/workdir/answer.json"):
            return VerifyResult(env_reward=0.0, done=True)
        try:
            answer = json.loads(sandbox.read_text("/home/user/workdir/answer.json"))
            reward = score_answer(answer, target) if isinstance(answer, dict) else 0.0
        except (OSError, ValueError, TypeError):
            reward = 0.0
        return VerifyResult(env_reward=reward, done=True)


def build_dataset(n_prompts: int, seed: int) -> tuple[list[dict], dict[str, dict]]:
    rows = list(load_dataset(DATASET_ID, split="train"))
    rows = [
        row
        for row in rows
        if row["github_label_quality"] in SCORABLE_QUALITIES or row["project_page_label_quality"] in SCORABLE_QUALITIES
    ]
    random.Random(seed).shuffle(rows)
    examples: list[dict] = []
    targets: dict[str, dict] = {}
    for row in rows[:n_prompts]:
        instruction = paper_instruction(row)
        targets[instruction_id(instruction)] = dict(row)
        examples.append({"prompt": [{"role": "user", "content": instruction}]})
    return examples, targets


class OpencodeTaskFactory(ResourceSessionFactory):
    def __init__(self, inner: OpenCodeSessionFactory):
        self._inner = inner

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> ResourceSession:
        instruction = task[-1]["content"] if isinstance(task, list) and task else str(task)
        return self._inner.create(instruction, seed=seed, episode_id=episode_id)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FreePortOpenCodeSessionFactory(OpenCodeSessionFactory):
    """Give each concurrent rollout its own transparent-proxy port."""

    def _start_proxy(self, sandbox):
        port = free_port()
        trace_path = proxy_trace_path(self._config)
        source_path = proxy_source_path(self._config)
        if not sandbox.exists(source_path):
            self._exec_with_retry(
                sandbox,
                "pip install --quiet 'fastapi>=0.104' 'uvicorn[standard]>=0.24' 'httpx>=0.27'",
                timeout=180,
                attempts=3,
                backoff_s=2.0,
                label="proxy deps install",
            )
            sandbox.write_text(source_path, oc_harness._PROXY_SOURCE_PATH.read_text())
            sandbox.write_text(f"{proxy_dir(self._config)}/__init__.py", "")
        args = [
            "python",
            "interception.py",
            "--upstream-url",
            self._config.base_url,
            "--trace",
            trace_path,
            "--port",
            str(port),
            "--top-logprobs",
            str(self._config.proxy_top_logprobs),
        ]
        if self._config.proxy_max_tokens_cap is not None:
            args += ["--max-tokens-cap", str(self._config.proxy_max_tokens_cap)]
        if self._config.proxy_disable_thinking:
            args.append("--disable-thinking")
        if self._config.model:
            args += ["--model-override", self._config.model]
        command = f"cd {shlex.quote(proxy_dir(self._config))} && " + " ".join(shlex.quote(arg) for arg in args)
        proxy = sandbox.start_bg(command, envs={"OPENCODE_UPSTREAM_API_KEY": self._config.api_key})
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.5)
        else:
            proxy.kill()
            raise RuntimeError(f"proxy did not start on port {port}")
        return proxy, f"http://127.0.0.1:{port}/v1", trace_path


def build_factory(
    sandbox_root: str,
    vllm_url: str,
    model: str,
    targets: dict,
    max_tokens: int,
) -> OpencodeTaskFactory:
    config = OpenCodeConfig(
        provider="openai_compatible",
        base_url=f"{vllm_url}/v1",
        api_key="EMPTY",
        model=model,
        sandbox_home="/home/user",
        agent_timeout_s=240.0,
        disabled_tools=["question", "task"],
        run_format="json",
        proxy_max_tokens_cap=max_tokens,
        proxy_disable_thinking=True,
    )
    backend = LocalSubprocessSandboxBackend(sandbox_root)
    backend.warmup()
    return OpencodeTaskFactory(
        FreePortOpenCodeSessionFactory(
            config=config,
            sandbox_backend=backend,
            mode="transparent_proxy",
            verifier=PaperUrlVerifier(targets),
        )
    )


def rollout_reward(outcome: HarnessRolloutOutcome) -> float | None:
    if outcome.env_reward is None:
        return None
    used_research_tool = outcome.tool_calls_by_name.get("webfetch", 0) + outcome.tool_calls_by_name.get("bash", 0)
    research_penalty = 0.1 if used_research_tool == 0 else 0.0
    step_penalty = min(0.3, 0.01 * max(0, outcome.tool_call_count - 20))
    timeout_penalty = 0.2 if outcome.timed_out else 0.0
    return float(outcome.env_reward) - research_penalty - step_penalty - timeout_penalty


def agent_turns(trace: list[TraceEntry]) -> list[TraceEntry]:
    def system_prompt(messages):
        return next((message.get("content") for message in messages if message.get("role") == "system"), None)

    primary = None
    for entry in trace:
        request = entry.get("request") or {}
        if request.get("messages") and request.get("tools"):
            primary = system_prompt(request["messages"])
            break
    entries = [
        copy.deepcopy(entry)
        for entry in trace
        if (request := entry.get("request") or {}).get("messages")
        and request.get("tools")
        and system_prompt(request["messages"]) == primary
    ]
    for entry in entries:
        for message in entry["request"]["messages"]:
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or tool_call
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        function["arguments"] = {}
    return entries


def wait_for_server(url: str, process: subprocess.Popen, timeout_s: int = 900) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited early with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy")


def launch_two_gpu(args: argparse.Namespace) -> None:
    if not args.trainer_only:
        server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_SERVER_DEV_MODE": "1"}
        server_command = [
            shutil.which("vllm") or "vllm",
            "serve",
            args.model,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.70",
            "--enforce-eager",
            "--language-model-only",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--logprobs-mode",
            "processed_logprobs",
            "--return-tokens-as-token-ids",
            "--weight-transfer-config",
            '{"backend":"nccl"}',
        ]
        server = subprocess.Popen(server_command, env=server_env)
        try:
            wait_for_server("http://127.0.0.1:8000", server)
            trainer_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
            trainer_command = [arg for arg in sys.argv if arg != "--launch"] + ["--trainer-only"]
            subprocess.run([sys.executable, *trainer_command], env=trainer_env, check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
        return
    train(args)


def train(args: argparse.Namespace) -> None:
    if args.skip_weight_sync and args.max_steps != 1:
        raise ValueError("--skip-weight-sync is only valid for a one-step smoke test")
    sandbox_root = args.sandbox_root or tempfile.mkdtemp(prefix="paper_url_opencode_")
    rows, targets = build_dataset(args.n_prompts, args.seed)
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        save_strategy="no",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        optim="paged_adamw_8bit",
        temperature=args.temperature,
        max_staleness=2,
        vllm_server_base_url=args.vllm_url,
        report_to="none",
        log_completions=True,
        hub_model_id=args.output_repo,
        push_to_hub=bool(args.output_repo),
    )
    worker = HarnessRolloutWorker(
        harness_session_factory=build_factory(
            sandbox_root,
            args.vllm_url,
            args.model,
            targets,
            args.max_completion_length,
        ),
        harness_adapter=None,
        rollout_reward_fn=rollout_reward,
        train_turn_fn=has_tool_call,
        agent_turn_fn=agent_turns,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        fork_threshold_tokens=1024,
        log_completions=True,
        num_completions_to_print=2,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
        weight_transfer=NoOpWeightTransfer() if args.skip_weight_sync else None,
    )
    trainer.train()
    if args.output_repo:
        trainer.push_to_hub()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="start vLLM on GPU 0 and training on GPU 1")
    parser.add_argument("--validate", action="store_true", help="validate dependencies, data filtering, and scoring")
    parser.add_argument("--trainer-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-weight-sync",
        action="store_true",
        help="one-step Qwen3.5 smoke only: bypass unsupported vLLM live weight reload",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n-prompts", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", default="qwen-paper-url-opencode")
    parser.add_argument("--output-repo")
    parser.add_argument("--sandbox-root")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.validate:
        validation_rows, validation_targets = build_dataset(2, parsed.seed)
        assert len(validation_rows) == 2 and len(validation_targets) == 2
        assert normalize_url("https://GitHub.com/QwenLM/Self-Lengthen.git?tab=readme", github=True) == (
            "https://github.com/qwenlm/self-lengthen"
        )
        assert (
            score_answer(
                {"github_url": "https://github.com/QwenLM/Self-Lengthen", "project_page_url": None},
                {
                    "reference_github_url": "https://github.com/qwenlm/self-lengthen",
                    "reference_project_page_url": None,
                    "github_label_quality": "silver",
                    "project_page_label_quality": "unknown",
                },
            )
            == 1.0
        )
        print(json.dumps({"status": "ok", "examples": len(validation_rows), "dataset": DATASET_ID}))
    elif parsed.launch or parsed.trainer_only:
        launch_two_gpu(parsed)
    else:
        raise SystemExit("Use --launch, or start vLLM separately and pass --trainer-only.")
