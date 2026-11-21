#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub>=1.13.0", "pydantic>=2.10.0,<2.13"]
# ///
"""Distil any teacher into any student: traces -> SFT -> GRPO -> eval -> Hub publishing, on HF Jobs.

    uv run pipeline.py --teacher kimi-k3.1-pro --student qwen3.5-9b --dataset nielsr/paper-url-extraction-v1
    uv run pipeline.py --teacher glm-5.2 --student qwen3.5-4b --stages eval --splits test,test_hard --dry-run

Teachers and students are profiles in `profiles/` (see `harness/profiles.py`). Each stage is one or more HF
Jobs in the `huggingface` namespace; with several stages the pipeline waits for each stage's jobs before
starting the next, a single stage is launched detached unless `--wait`. `--dry-run` prints every job (image,
flavor, command) without staging code or launching anything.

Stages:
  traces   teacher trajectories on the train split (CPU jobs, Inference Providers billed to HF_BILL_TO),
           published with agent-trace sessions to <run>-traces
  sft      LoRA SFT of the student on the accepted traces -> <run>-lora
  grpo     merge SFT (-> <run>-sft-merged), GRPO in OpenCode (-> <run>-grpo-lora), merge (-> <run>-grpo-merged)
  eval     base vs SFT, SFT vs SFT+GRPO, and the teacher baseline on every --splits split -> <run>-eval
  publish  visibility check (models/datasets public, code private) and the project collection

Artifacts are named `<namespace>/<run-name>-<suffix>`; override any of them (e.g. to resume a run whose
repos predate this naming) with --traces-repo, --sft-adapter, --sft-merged, --grpo-adapter,
--grpo-merged, --eval-repo.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.profiles import Student, Teacher, load_student, load_teacher  # noqa: E402

STAGES = ("traces", "sft", "merge", "probe", "grpo", "eval", "publish")
DEFAULT_STAGES = ("traces", "sft", "merge", "grpo", "eval", "publish")  # probe is opt-in
PROBE_TAG = "probe-sft"
JOB_NAMESPACE = "huggingface"  # AGENTS.md: bill Jobs and Inference Providers to the huggingface org
COLLECTION = "nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e"
UV_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
HF_CLI = "uvx --from 'huggingface-hub>=1.13' hf"

# Everything a job needs: the stage scripts, the whole harness package, the trace exporter. Student hook
# modules (`requires`) and the GRPO prompt pool are added per run.
STAGED_FILES = (
    "generate_opencode_traces.py",
    "evaluate_vllm.py",
    "train_sft.py",
    "train_grpo_harness.py",
    "merge_adapter.py",
    "difficulty_probe.py",
    "trace_export.py",
    "pipeline.py",
)
STAGED_PACKAGES = ("harness", "profiles")  # profiles: an orchestrator job runs pipeline.py from the snapshot

# Recipe defaults shared by every pair (student profiles append their own arguments after these, and the
# last occurrence of a flag wins).
GRPO_RECIPE = [
    "--protocol-weight", "0.5", "--num-generations", "8", "--max-steps", "100", "--save-steps", "50",
    "--gradient-accumulation-steps", "2", "--report-to", "none",
]  # fmt: skip
EVAL_RECIPE = {"concurrency": 8, "episode_timeout": 180, "max_tool_calls": 15}


@dataclass
class Job:
    stage: str
    name: str
    image: str
    flavor: str
    timeout: str
    command: str
    env: dict[str, str]


@dataclass
class Run:
    teacher: Teacher
    student: Student
    dataset: str
    namespace: str
    name: str
    traces_repo: str
    sft_adapter: str
    sft_merged: str
    grpo_adapter: str
    grpo_merged: str
    eval_repo: str
    code_repo: str

    def artifacts(self) -> list[tuple[str, str, str]]:
        """(repo, type, collection note) for everything this run publishes."""
        pair = f"{self.teacher.name} -> {self.student.name}"
        return [
            (self.traces_repo, "dataset", f"{pair}: teacher traces ({self.teacher.model}) with agent-trace sessions"),
            (self.sft_adapter, "model", f"{pair}: student SFT LoRA adapter"),
            (self.sft_merged, "model", f"{pair}: student SFT merged checkpoint"),
            (self.grpo_adapter, "model", f"{pair}: student GRPO LoRA adapter (from SFT)"),
            (self.grpo_merged, "model", f"{pair}: student SFT + GRPO merged checkpoint"),
            (self.eval_repo, "dataset", f"{pair}: OpenCode evaluation episodes (base, SFT, GRPO, teacher)"),
        ]


def make_run(args: argparse.Namespace) -> Run:
    teacher, student = load_teacher(args.teacher), load_student(args.student)
    name = args.run_name or f"{student.slug}-url-agent-{teacher.slug}"
    prefix = f"{args.namespace}/{name}"
    return Run(
        teacher=teacher,
        student=student,
        dataset=args.dataset,
        namespace=args.namespace,
        name=name,
        traces_repo=args.traces_repo or f"{prefix}-traces",
        sft_adapter=args.sft_adapter or f"{prefix}-lora",
        sft_merged=args.sft_merged or f"{prefix}-sft-merged",
        grpo_adapter=args.grpo_adapter or f"{prefix}-grpo-lora",
        grpo_merged=args.grpo_merged or f"{prefix}-grpo-merged",
        eval_repo=args.eval_repo or f"{prefix}-eval",
        code_repo=args.code_repo or f"{prefix}-code",
    )


# --------------------------------------------------------------------------------------------------
# Command building
# --------------------------------------------------------------------------------------------------


def script_dependencies(script: str) -> list[str]:
    """The PEP 723 `dependencies` of a staged script."""
    text = (ROOT / script).read_text()
    block = re.search(r"^# /// script\n(.*?)^# ///$", text, flags=re.MULTILINE | re.DOTALL)
    if not block:
        return []
    import tomllib

    metadata = tomllib.loads("\n".join(line.removeprefix("# ").removeprefix("#") for line in block.group(1).splitlines()))
    return list(metadata.get("dependencies", []))


def python(script: str, args: list[str], *, deps: list[str] | None = None, extra: list[str] = ()) -> str:
    """Run a staged script: with its own PEP 723 environment (plus `extra`), or in a `deps` environment."""
    tail = " ".join([script, *args])
    if deps is None:
        return f"uv run {''.join(f'--with {shlex.quote(dep)} ' for dep in extra)}{tail}"
    withs = "".join(f"--with {shlex.quote(dep)} " for dep in [*deps, *extra])
    return f"uv run --no-project --python 3.12 --python-preference only-managed {withs}python {tail}"


def q(value: Any) -> str:
    return shlex.quote(str(value))


def prelude(run: Run, rev: str, *, image_vllm: bool = False) -> str:
    lines = [
        "set -euo pipefail",
        "unset UV_OVERRIDE UV_INDEX_STRATEGY",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        'export PATH="$HOME/.local/bin:$PATH"',
        # The agent itself runs `hf papers ...` inside OpenCode.
        "command -v hf >/dev/null || uv tool install -q 'huggingface-hub>=1.13'",
        "nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true",
        "mkdir -p /work && cd /work",
        f"{HF_CLI} download {q(run.code_repo)} --include 'code/{rev}/*' --local-dir /work/stage >/dev/null",
        f"cp -r /work/stage/code/{rev}/. /work/",
    ]
    if image_vllm:
        lines += [
            "vllm --version || true",
            'LMO=--language-model-only; vllm serve --help=all 2>/dev/null | grep -q -- "--language-model-only" '
            "|| LMO=--no-language-model-only",
        ]
    return "\n".join(lines)


def serve_flags(student: Student, *, grpo: bool = False) -> list[str]:
    serve = student.serve
    flags = ["--tool-call-parser", serve.tool_call_parser, "--reasoning-parser", q(serve.reasoning_parser)]
    if serve.vllm == "image":
        flags += ["--vllm-binary", '"$(command -v vllm)"']
    if grpo:
        return flags
    flags += [
        "--enforce-eager" if serve.enforce_eager else "--no-enforce-eager",
        "$LMO" if serve.vllm == "image" and serve.language_model_only else
        ("--language-model-only" if serve.language_model_only else "--no-language-model-only"),
        "--max-model-len", str(serve.max_model_len), "--max-output-tokens", str(serve.max_output_tokens),
        "--gpu-memory-utilization", str(serve.gpu_memory_utilization), "--max-lora-rank", str(student.lora.rank),
    ]  # fmt: skip
    if serve.vllm_extra_args:
        flags.append(f"--vllm-extra-args={q(serve.vllm_extra_args)}")
    if student.lora.adapter_converter:
        flags += ["--adapter-converter", student.lora.adapter_converter]
    return flags


def evaluator(student: Student, args: list[str]) -> str:
    if student.serve.vllm == "image":  # keep the image's vLLM; the evaluator's other deps in their own env
        deps = [dep for dep in script_dependencies("evaluate_vllm.py") if not dep.startswith("vllm")]
        return python("evaluate_vllm.py", args, deps=deps + student.serve.deps)
    return python("evaluate_vllm.py", args)


def merger(student: Student, args: list[str]) -> str:
    merge = student.merge
    args = [*args, "--merge-method", merge.method]
    if merge.replace_deps:
        return python("merge_adapter.py", args, deps=merge.deps)
    return python("merge_adapter.py", args, extra=merge.deps)


def trace_jobs(run: Run, rev: str, args: argparse.Namespace) -> list[Job]:
    teacher = run.teacher
    settings = teacher.traces
    shards = args.trace_shards or settings.shards
    jobs = []
    for index in range(shards):
        generate = [
            "--dataset", q(run.dataset), "--split", "train", "--limit", "1000000", "--shard", f"{index}/{shards}",
            "--attempts-per-row", str(settings.attempts_per_row), "--concurrency", str(settings.concurrency),
            "--episode-timeout", str(settings.episode_timeout), "--max-tool-calls", str(settings.max_tool_calls),
            "--resume-attempted", "--model", q(teacher.opencode_model), "--hf-bill-to", JOB_NAMESPACE,
            "--output", f"/work/out/train-{index}.jsonl", "--output-repo", q(run.traces_repo),
            "--path-in-repo", f"raw/train-{index}.jsonl", "--public",
        ]  # fmt: skip
        if settings.accept_partial:
            generate.append("--accept-partial")
        if teacher.opencode_model_entry:
            generate += ["--opencode-model-entry", q(json.dumps(teacher.opencode_model_entry))]
        # A rerun resumes from the shard's published records instead of paying for those papers again.
        resume = (
            f"mkdir -p /work/out && ({HF_CLI} download {q(run.traces_repo)} raw/train-{index}.jsonl --repo-type dataset "
            f"--local-dir /work/prev >/dev/null 2>&1 && cp /work/prev/raw/train-{index}.jsonl /work/out/) || true"
        )
        command = "\n".join([prelude(run, rev), resume, python("generate_opencode_traces.py", generate)])
        jobs.append(
            Job("traces", f"{run.name}-traces-{index}", UV_IMAGE, settings.flavor, settings.timeout, command,
                {"HF_BILL_TO": JOB_NAMESPACE})  # fmt: skip
        )
    return jobs


def sft_job(run: Run, rev: str, args: argparse.Namespace) -> Job:
    student = run.student
    sft = student.sft
    traces = [
        f"{HF_CLI} download {q(run.traces_repo)} --repo-type dataset --include {q(args.traces_files)} "
        "--local-dir /work/traces >/dev/null",
        f"cat /work/traces/{args.traces_files} > /work/traces.jsonl",
        "wc -l /work/traces.jsonl",
    ]
    train = [
        "--traces", "/work/traces.jsonl", "--model", q(student.model), "--output", "/out/sft",
        "--hub-model-id", q(run.sft_adapter), "--public", "--lora-rank", str(student.lora.rank),
        "--lora-target-modules", q(student.lora.sft_target_modules), *sft.args,
    ]  # fmt: skip
    if sft.model_class:
        train += ["--model-class", sft.model_class]
    if sft.train_patch:
        train += ["--train-patch", sft.train_patch]
    if sft.keep_reasoning:
        train.append("--keep-reasoning")
    command = "\n".join(
        [prelude(run, rev), *traces, python("train_sft.py", train, deps=sft.deps or None), "echo SFT_DONE"]
    )
    return Job("sft", f"{run.name}-sft", sft.image, sft.flavor, sft.timeout, command, {})


def merge_job(run: Run, rev: str, args: argparse.Namespace) -> Job:
    """Merge the SFT adapter into the student and publish <run>-sft-merged (GRPO and the probe start from it)."""
    student = run.student
    merge = merger(
        student,
        ["--base-model", q(student.model), "--adapter", q(run.sft_adapter), "--card-base-model", q(student.model),
         "--card-adapter", q(run.sft_adapter), "--hub-model-id", q(run.sft_merged), "--public", "--output", "/model"],
    )  # fmt: skip
    command = "\n".join([prelude(run, rev), merge, "echo MERGE_DONE"])
    return Job("merge", f"{run.name}-merge-sft", student.merge.image, student.merge.flavor, student.merge.timeout, command, {})


def probe_jobs(run: Run, rev: str, args: argparse.Namespace) -> list[Job]:
    """Difficulty probe of the merged SFT policy, sharded; qualified IDs land in <run>-eval/difficulty/."""
    student = run.student
    probe = student.probe
    deps = [dep for dep in script_dependencies("evaluate_vllm.py") if not dep.startswith("vllm")] + student.serve.deps
    jobs = []
    for index in range(probe.shards):
        flags = [
            "--model", q(run.sft_merged), "--dataset", q(run.dataset), "--sample", str(probe.sample),
            "--shard", f"{index}/{probe.shards}", "--k", str(probe.k), "--concurrency", str(probe.concurrency),
            "--temperature", str(probe.temperature), "--episode-timeout", str(probe.episode_timeout),
            "--max-tool-calls", str(EVAL_RECIPE["max_tool_calls"]), "--max-model-len", str(student.serve.max_model_len),
            "--max-output-tokens", str(student.serve.max_output_tokens), *serve_flags(student, grpo=True),
            "--output-dir", "/probe", "--output-repo", q(run.eval_repo), "--tag", f"{PROBE_TAG}/shard-{index}",
        ]  # fmt: skip
        if student.serve.vllm_extra_args:
            flags.append(f"--vllm-extra-args={q(student.serve.vllm_extra_args)}")
        select: list[str] = []
        pool = Path(args.grpo_ids).name if args.grpo_ids and args.grpo_ids != "probe" else ""
        if args.probe_exclude_tiers:  # candidates: scoreable train papers outside these tiers (within the pool)
            select_flags = ["--dataset", q(run.dataset), "--split", "train", "--exclude-tiers", q(args.probe_exclude_tiers),
                            "--output", "probe-candidates.txt"] + (["--include-ids-file", pool] if pool else [])  # fmt: skip
            select = [python("-m harness.select", select_flags, deps=["datasets>=4.0.0", "pydantic>=2.10.0,<2.13"])]
            pool = "probe-candidates.txt"
        if pool:
            flags += ["--include-ids-file", pool]
        flags += probe.args
        command = "\n".join(
            [prelude(run, rev, image_vllm=student.serve.vllm == "image"), *select,
             python("difficulty_probe.py", flags, deps=deps), "echo PROBE_DONE"]  # fmt: skip
        )
        jobs.append(
            Job("probe", f"{run.name}-probe-{index}", student.serve.image, probe.flavor or student.serve.flavor,
                probe.timeout, command, {})  # fmt: skip
        )
    return jobs


def grpo_jobs(run: Run, rev: str, args: argparse.Namespace) -> list[list[Job]]:
    """GRPO from <run>-sft-merged (one wave), then merging the GRPO adapter (a second wave)."""
    student = run.student
    grpo = student.grpo
    fetch_model = f"{HF_CLI} download {q(run.sft_merged)} --local-dir /model >/dev/null"
    train = [
        "--launch", "--model", "/model", "--dataset", q(run.dataset), "--output-dir", "/grpo-run",
        "--hub-model-id", q(run.grpo_adapter), "--public", "--lora-r", str(student.lora.rank),
        "--max-model-len", str(student.serve.max_model_len),
        "--max-completion-length", str(student.serve.max_output_tokens),
        "--max-tool-calls", str(EVAL_RECIPE["max_tool_calls"]), "--trainer", grpo.backend,
        *serve_flags(student, grpo=True), *GRPO_RECIPE,
    ]  # fmt: skip
    if grpo.backend == "simple":  # it writes its own adapter cards; /model is not a valid Hub base_model
        train += ["--card-base-model", q(run.sft_merged)]
    if student.lora.grpo_target_modules:
        train += ["--lora-target-modules", q(student.lora.grpo_target_modules)]
    ids = grpo_ids_steps(run, args)
    if ids:
        train += ["--include-ids-file", ids[-1]]
    train += grpo.args + shlex.split(args.grpo_args)
    waves: list[list[Job]] = []
    if not args.grpo_merge_only:
        command = "\n".join(
            [prelude(run, rev, image_vllm=student.serve.vllm == "image"), fetch_model, *ids[:-1],
             python("train_grpo_harness.py", train, deps=grpo.deps or None), "echo GRPO_DONE"]  # fmt: skip
        )
        waves.append([Job("grpo", f"{run.name}-grpo", grpo.image, grpo.flavor, grpo.timeout, command, {})])
    # The adapter was trained on /model, so the merge runs against the same local copy of <run>-sft-merged.
    merge = merger(
        student,
        ["--base-model", "/model", "--adapter", q(run.grpo_adapter), "--card-base-model", q(student.model),
         "--card-adapter", q(run.grpo_adapter), "--hub-model-id", q(run.grpo_merged), "--public",
         "--output", "/grpo-merged"],
    )  # fmt: skip
    command = "\n".join([prelude(run, rev), fetch_model, merge, "echo MERGE_DONE"])
    waves.append([Job("grpo", f"{run.name}-merge-grpo", student.merge.image, student.merge.flavor,
                      student.merge.timeout, command, {})])  # fmt: skip
    return waves


def grpo_ids_steps(run: Run, args: argparse.Namespace) -> list[str]:
    """Shell steps producing the GRPO prompt pool; the last element is the file name passed to the trainer."""
    if not args.grpo_ids:
        return []
    if args.grpo_ids != "probe":
        return [Path(args.grpo_ids).name]
    return [
        f"{HF_CLI} download {q(run.eval_repo)} --repo-type dataset --include 'difficulty/{PROBE_TAG}/shard-*/qualified-ids.txt' "
        "--local-dir /work/probe >/dev/null",
        f"cat /work/probe/difficulty/{PROBE_TAG}/shard-*/qualified-ids.txt | sort -u > /work/probe-ids.txt",
        "wc -l /work/probe-ids.txt",
        "probe-ids.txt",
    ]


def eval_jobs(run: Run, rev: str, args: argparse.Namespace) -> list[Job]:
    student = run.student
    splits = [split for split in args.splits.split(",") if split]
    common = [
        "--dataset", q(run.dataset), "--limit", str(args.eval_limit), "--concurrency", str(EVAL_RECIPE["concurrency"]),
        "--episode-timeout", str(EVAL_RECIPE["episode_timeout"]), "--max-tool-calls", str(EVAL_RECIPE["max_tool_calls"]),
        "--output-repo", q(run.eval_repo), "--public",
    ]  # fmt: skip
    if args.eval_include_partial:
        common.append("--include-partial")
    # tag -> (served base model, adapter); the evaluator runs base and adapter on the same papers.
    variants = {
        "base": (student.model, "''"),
        "sft": (student.model, q(run.sft_adapter)),
        "grpo": (q(run.sft_merged), q(run.grpo_adapter)),
    }
    tags = {"base": "base/", "sft": "sft/", "grpo": "sft-grpo/", "teacher": "teacher/"}
    jobs = []
    for variant in [v for v in args.eval_variants.split(",") if v]:
        # One job per (variant, split): splits run in parallel and a failure reruns only its split.
        for split in splits:
            if variant == "teacher":
                jobs.append(teacher_eval_job(run, rev, split, args))
                continue
            base, adapter = variants[variant]
            evaluate = [
                *common, "--split", split, "--base-model", base, "--adapter", adapter, "--tag", tags[variant],
                "--output", f"/eval/{variant}/{split}.jsonl", *serve_flags(student),
            ]  # fmt: skip
            command = "\n".join(
                [prelude(run, rev, image_vllm=student.serve.vllm == "image"), evaluator(student, evaluate), "echo EVAL_DONE"]
            )
            jobs.append(
                Job("eval", f"{run.name}-eval-{variant}-{split.replace('_', '-')}", student.serve.image,
                    student.serve.flavor, student.serve.timeout, command, {})  # fmt: skip
            )
    return jobs


def teacher_eval_job(run: Run, rev: str, split: str, args: argparse.Namespace) -> Job:
    """The teacher baseline on one split under the students' evaluation settings (one attempt, same limits)."""
    teacher = run.teacher
    generate = [
        "--dataset", q(run.dataset), "--split", split, "--limit", str(args.eval_limit), "--attempts-per-row", "1",
        "--concurrency", str(teacher.traces.baseline_concurrency),
        "--episode-timeout", str(EVAL_RECIPE["episode_timeout"]),
        "--max-tool-calls", str(EVAL_RECIPE["max_tool_calls"]), "--model", q(teacher.opencode_model),
        "--hf-bill-to", JOB_NAMESPACE, "--evaluation-label", "teacher", "--no-resume",
        "--output", f"/eval/teacher/{split}.jsonl", "--output-repo", q(run.eval_repo),
        "--path-in-repo", f"teacher/{split}.jsonl", "--public",
    ]  # fmt: skip
    if args.eval_include_partial:  # the same paper population as the evaluator's --include-partial
        generate.append("--accept-partial")
    if teacher.opencode_model_entry:
        generate += ["--opencode-model-entry", q(json.dumps(teacher.opencode_model_entry))]
    summary = f"/eval/teacher/{split}-summary.json"
    command = "\n".join([
        prelude(run, rev), "mkdir -p /eval/teacher",
        python("generate_opencode_traces.py", generate),
        f"uv run --no-project --with 'pydantic>=2.10.0,<2.13' python -m harness.report summarize "
        f"/eval/teacher/{split}.jsonl --output {summary} >/dev/null",
        f"{HF_CLI} upload {q(run.eval_repo)} {summary} teacher/{split}-summary.json --repo-type dataset",
        "echo EVAL_DONE",
    ])  # fmt: skip
    return Job("eval", f"{run.name}-eval-teacher-{split.replace('_', '-')}", UV_IMAGE, run.teacher.traces.flavor, "8h",
               command, {"HF_BILL_TO": JOB_NAMESPACE})  # fmt: skip


# --------------------------------------------------------------------------------------------------
# Staging, launching, publishing
# --------------------------------------------------------------------------------------------------


def staged_paths(run: Run, args: argparse.Namespace) -> list[Path]:
    paths = [ROOT / name for name in STAGED_FILES]
    for package in STAGED_PACKAGES:
        paths += sorted(p for p in (ROOT / package).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths += [ROOT / name for name in run.student.requires if args.dry_run is False or (ROOT / name).exists()]
    if args.grpo_ids and args.grpo_ids != "probe":
        paths.append(Path(args.grpo_ids).resolve())
    return paths


def stage_code(run: Run, args: argparse.Namespace) -> str:
    """Upload the job code to the run's private code repo under code/<rev>/ and return <rev>."""
    from huggingface_hub import HfApi

    rev = time.strftime("%Y%m%d-%H%M%S")
    paths = staged_paths(run, args)
    if args.dry_run:
        print(f"[dry-run] would stage {len(paths)} files to {run.code_repo} (private) under code/{rev}/:")
        for path in paths:
            print(f"  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        return rev
    api = HfApi()
    api.create_repo(run.code_repo, private=True, exist_ok=True)  # AGENTS.md: code snapshots stay private
    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            target = Path(tmp) / (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        api.upload_folder(folder_path=tmp, path_in_repo=f"code/{rev}", repo_id=run.code_repo,
                          commit_message=f"Stage pipeline code {rev}")  # fmt: skip
    landed = set(api.list_repo_files(run.code_repo))
    missing = [str(p) for p in paths if f"code/{rev}/{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name}" not in landed]
    if missing:
        raise RuntimeError(f"code staging did not land: {missing}")
    print(f"staged {len(paths)} files to {run.code_repo}@code/{rev}", flush=True)
    return rev


def launch(jobs: list[Job], args: argparse.Namespace) -> list[str]:
    if args.dry_run:
        for job in jobs:
            print(f"\n[dry-run] job {job.name}: image={job.image} flavor={job.flavor} timeout={job.timeout} "
                  f"namespace={JOB_NAMESPACE} env={job.env} secrets=HF_TOKEN")  # fmt: skip
            print(job.command)
        return []
    from huggingface_hub import HfApi, get_token

    api, token = HfApi(), get_token()
    if not token:
        raise SystemExit("no Hugging Face token (set HF_TOKEN or run `hf auth login`)")
    ids = []
    for job in jobs:
        info = api.run_job(
            image=job.image,
            command=["bash", "-c", job.command],
            env=job.env or None,
            secrets={"HF_TOKEN": token},
            flavor=job.flavor,
            timeout=job.timeout,
            name=job.name,
            namespace=JOB_NAMESPACE,
        )
        print(f"launched {job.name}: {info.id} ({info.url})", flush=True)
        ids.append(info.id)
    return ids


def wait(ids: list[str]) -> None:
    from huggingface_hub import HfApi

    if not ids:
        return
    infos = HfApi().wait_for_job(ids, poll_interval=60, namespace=JOB_NAMESPACE)
    infos = infos if isinstance(infos, list) else [infos]
    failed = [(info.id, info.status.stage) for info in infos if str(info.status.stage).upper().find("COMPLETED") < 0]
    if failed:
        raise SystemExit(f"jobs did not complete: {failed}")


def publish(run: Run, args: argparse.Namespace) -> None:
    """Public models/datasets, private code repo, every existing artifact in the project collection."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    for repo, kind, note in run.artifacts():
        try:
            info = api.repo_info(repo, repo_type=kind)
        except RepositoryNotFoundError:
            print(f"skip {repo}: not created yet", flush=True)
            continue
        if args.dry_run:
            print(f"[dry-run] {repo}: private={info.private} -> public; collection note: {note}")
            continue
        if info.private:
            api.update_repo_settings(repo, repo_type=kind, private=False)
        if kind == "dataset":
            files = api.list_repo_files(repo, repo_type="dataset")
            if not any(name.startswith("sessions/") for name in files):
                raise SystemExit(f"{repo} has no agent-trace sessions; run `uv run trace_export.py --repo {repo}`")
        api.add_collection_item(COLLECTION, item_id=repo, item_type=kind, note=note, exists_ok=True)
        print(f"published {repo} ({kind}) in the collection", flush=True)
    try:
        if not api.repo_info(run.code_repo).private:
            raise SystemExit(f"{run.code_repo} must stay private (AGENTS.md)")
    except RepositoryNotFoundError:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, help="Teacher profile (profiles/teachers/*.toml)")
    parser.add_argument("--student", required=True, help="Student profile (profiles/students/*.toml)")
    parser.add_argument("--dataset", default="nielsr/paper-url-extraction-v1", help="Any dataset with the same schema")
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES), help=f"Comma-separated subset of {','.join(STAGES)}")
    parser.add_argument("--splits", default="test,validation", help="Evaluation splits, e.g. test,validation,test_hard")
    parser.add_argument("--eval-variants", default="sft,grpo,teacher", help="Comma list of base,sft,grpo,teacher")
    parser.add_argument("--eval-limit", type=int, default=50, help="Scoreable papers per split")
    parser.add_argument(
        "--eval-include-partial",
        action="store_true",
        help="Evaluate every paper with at least one verified requested label (scored on verified fields only)",
    )
    parser.add_argument("--trace-shards", type=int, default=0, help="Parallel trace jobs (default: teacher profile)")
    parser.add_argument("--traces-files", default="raw/train-*.jsonl", help="Trace files in --traces-repo for SFT")
    parser.add_argument(
        "--grpo-ids",
        help="GRPO prompt pool: an arXiv IDs file (e.g. scripts/rl-kimi-9b-ids.txt), or 'probe' for the qualified IDs "
        "of the probe stage (the default when probe runs); with probe, a file restricts the probed candidates",
    )
    parser.add_argument("--namespace", default="nielsr", help="Owner of the published repos")
    parser.add_argument("--run-name", help="Artifact name prefix (default: <student>-url-agent-<teacher>)")
    for name in ("traces-repo", "sft-adapter", "sft-merged", "grpo-adapter", "grpo-merged", "eval-repo", "code-repo"):
        parser.add_argument(f"--{name}", help=f"Override the {name.replace('-', ' ')} repo id")
    parser.add_argument("--grpo-args", default="", help="Extra train_grpo_harness.py arguments (last flag wins)")
    parser.add_argument("--probe-exclude-tiers", default="", help="Probe only train papers outside these tiers, e.g. 7")
    parser.add_argument("--grpo-merge-only", action="store_true", help="GRPO stage: merge the pushed adapter, no training")
    parser.add_argument("--rev", help="Reuse already staged code (code/<rev>/ in the code repo)")
    parser.add_argument("--commit", help="Code commit to record (default: git rev-parse HEAD)")
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="Stage the code, record the run manifest, and run the whole pipeline in a cpu-basic job",
    )
    parser.add_argument("--orchestrator-timeout", default="48h")
    parser.add_argument("--wait", action="store_true", help="Wait for the jobs of a single stage")
    parser.add_argument("--dry-run", action="store_true", help="Print the jobs; stage and launch nothing")
    args = parser.parse_args(argv)
    args.stages = [stage for stage in args.stages.split(",") if stage]
    unknown = set(args.stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages {sorted(unknown)}")
    args.stages = [stage for stage in STAGES if stage in args.stages]
    if "probe" in args.stages and not args.grpo_ids:
        args.grpo_ids = "probe"
    return args


def build_waves(stage: str, run: Run, rev: str, args: argparse.Namespace) -> list[list[Job]]:
    """A stage's jobs as waves: jobs in a wave run in parallel, a wave starts when the previous one completed."""
    if stage == "traces":
        return [trace_jobs(run, rev, args)]
    if stage == "sft":
        return [[sft_job(run, rev, args)]]
    if stage == "merge":
        return [[merge_job(run, rev, args)]]
    if stage == "probe":
        return [probe_jobs(run, rev, args)]
    if stage == "grpo":
        return grpo_jobs(run, rev, args)
    if stage == "eval":
        return [eval_jobs(run, rev, args)]
    raise ValueError(stage)


def build_jobs(stage: str, run: Run, rev: str, args: argparse.Namespace) -> list[Job]:
    return [job for wave in build_waves(stage, run, rev, args) for job in wave]


def git_commit() -> str:
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record_run(run: Run, rev: str, args: argparse.Namespace, argv: list[str]) -> None:
    """Write the run manifest (command, code commit, staged snapshot) to the eval repo: pipeline/run-<rev>.json."""
    from huggingface_hub import HfApi

    manifest = {
        "run": run.name, "teacher": run.teacher.name, "student": run.student.name, "dataset": run.dataset,
        "commit": args.commit, "code_snapshot": f"{run.code_repo}@code/{rev}", "stages": args.stages,
        "command": "uv run pipeline.py " + shlex.join(argv), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": {repo: kind for repo, kind, _ in run.artifacts()},
    }  # fmt: skip
    if args.dry_run:
        print(f"[dry-run] manifest for {run.eval_repo}/pipeline/run-{rev}.json: {json.dumps(manifest)}")
        return
    api = HfApi()
    api.create_repo(run.eval_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(manifest, indent=2).encode(), path_in_repo=f"pipeline/run-{rev}.json",
                    repo_id=run.eval_repo, repo_type="dataset", commit_message=f"Pipeline run {rev} at {args.commit}")  # fmt: skip
    print(f"recorded run manifest {run.eval_repo}/pipeline/run-{rev}.json (commit {args.commit})", flush=True)


def launch_orchestrator(run: Run, rev: str, args: argparse.Namespace, argv: list[str]) -> None:
    """Run this pipeline (all stages and waits) inside a small CPU job, from the pinned code snapshot."""
    child = [arg for arg in argv if arg != "--orchestrate"] + ["--rev", rev, "--commit", args.commit]
    command = "\n".join([prelude(run, rev), f"uv run pipeline.py {shlex.join(child)}"])
    launch([Job("orchestrate", f"{run.name}-pipeline", UV_IMAGE, "cpu-basic", args.orchestrator_timeout, command, {})], args)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    args.commit = args.commit or git_commit()
    run = make_run(args)
    missing = run.student.missing_requirements(ROOT)
    if missing and set(args.stages) - {"publish"}:
        message = f"student {run.student.name} needs {missing}, which are not in this checkout"
        if not args.dry_run:
            raise SystemExit(message)
        print(f"warning: {message}; a real run would stop here", flush=True)
    print(f"run {run.name}: {run.teacher.name} ({run.teacher.opencode_model}) -> {run.student.name} ({run.student.model}) "
          f"on {run.dataset}; stages {','.join(args.stages)}", flush=True)  # fmt: skip
    rev = args.rev
    if args.orchestrate:
        if args.commit.endswith("-dirty") and not args.dry_run:
            raise SystemExit("commit or stash local changes first: the run records its code commit")
        rev = rev or stage_code(run, args)
        record_run(run, rev, args, argv)
        launch_orchestrator(run, rev, args, argv)
        return
    for stage in args.stages:
        if stage == "publish":
            publish(run, args)
            continue
        rev = rev or stage_code(run, args)
        waves = build_waves(stage, run, rev, args)
        for number, wave in enumerate(waves, 1):
            ids = launch(wave, args)
            if number < len(waves) or len(args.stages) > 1 or args.wait:
                wait(ids)


if __name__ == "__main__":
    main()
