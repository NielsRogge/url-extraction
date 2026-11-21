"""Teacher and student profiles: one small TOML file per model under `profiles/{teachers,students}/`.

A profile holds everything about a model that the stage scripts used to hard-code: how OpenCode reaches a
teacher (provider pin, inline model entry), and for a student the vLLM serving flags, HF Jobs images and
GPU flavors, LoRA targets, model-specific hooks (adapter converter, training patch, merge method) and the
GRPO backend. Fields left out of a profile take the defaults below, which describe Qwen3.5 on H200.

    from harness.profiles import load_teacher, load_student
    load_student("qwen3.5-9b").serve.tool_call_parser   # "qwen3_coder"
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

QWEN35_TEXT_LORA_TARGETS = (
    "down_proj,gate_proj,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,k_proj,o_proj,out_proj,q_proj,up_proj,v_proj"
)
DEFAULT_TRAIN_IMAGE = "huggingface/trl"


@dataclass
class TraceSettings:
    """Teacher-trace generation (`generate_opencode_traces.py`) on the train split."""

    concurrency: int = 4
    attempts_per_row: int = 1
    accept_partial: bool = True
    episode_timeout: float = 180
    max_tool_calls: int = 20
    shards: int = 3
    flavor: str = "cpu-upgrade"
    timeout: str = "12h"


@dataclass
class Teacher:
    name: str
    model: str  # Hub id, for model cards and docs
    opencode_model: str  # OpenCode PROVIDER/MODEL, e.g. huggingface/org/model or huggingface/org/model:provider
    provider: str = ""  # Inference Providers routing pin ("" = automatic); informational, the pin is in opencode_model
    opencode_model_entry: dict[str, Any] = field(default_factory=dict)  # inline OpenCode model declaration
    notes: str = ""
    traces: TraceSettings = field(default_factory=TraceSettings)

    @property
    def slug(self) -> str:
        return slugify(self.name)


@dataclass
class ServeSettings:
    """vLLM serving for evaluation (and the GRPO rollout engine)."""

    image: str = DEFAULT_TRAIN_IMAGE
    vllm: str = "pep723"  # "pep723": the script installs vLLM itself; "image": use the image's `vllm` binary
    flavor: str = "h200"
    timeout: str = "4h"
    deps: list[str] = field(default_factory=list)  # extra deps for the evaluator env when vllm == "image"
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = ""
    enforce_eager: bool = True
    language_model_only: bool = True
    max_model_len: int = 65536
    max_output_tokens: int = 1024
    gpu_memory_utilization: float = 0.85
    vllm_extra_args: str = ""


@dataclass
class LoraSettings:
    rank: int = 32
    sft_target_modules: str = QWEN35_TEXT_LORA_TARGETS  # comma list or regex
    grpo_target_modules: str = ""  # regex; "" = the GRPO trainer's default (Qwen3.5 language model)
    adapter_converter: str = ""  # module:function(source, destination) making an adapter servable by vLLM


@dataclass
class SFTSettings:
    image: str = DEFAULT_TRAIN_IMAGE
    flavor: str = "h200"
    timeout: str = "8h"
    deps: list[str] = field(default_factory=list)  # [] = the script's own PEP 723 dependencies
    model_class: str = ""  # transformers class when SFTTrainer cannot load the model by id
    train_patch: str = ""  # module whose enable(chunk_size=...) patches training attention
    keep_reasoning: bool = False  # distil the teacher's reasoning_content
    args: list[str] = field(
        default_factory=lambda: ["--no-4bit", "--max-length", "32768", "--epochs", "1", "--full-trajectory",
                                 "--save-steps", "50"]  # fmt: skip
    )


@dataclass
class MergeSettings:
    method: str = "peft"  # or module:function(args, adapter_snapshot, token) -> output dir
    deps: list[str] = field(default_factory=lambda: ["torchvision", "pillow"])  # added to the script's deps
    replace_deps: bool = False  # True: `deps` is the whole environment (no PEP 723 deps)


@dataclass
class GRPOSettings:
    image: str = DEFAULT_TRAIN_IMAGE
    flavor: str = "h200x2"
    timeout: str = "8h"
    backend: str = "trl"  # "trl" (HarnessAsyncGRPOTrainer) or "simple" (synchronous loop, e.g. Inkling 2)
    deps: list[str] = field(default_factory=list)  # [] = the script's PEP 723 deps
    args: list[str] = field(default_factory=lambda: ["--max-concurrent-rollouts", "32"])


@dataclass
class Student:
    name: str
    model: str
    notes: str = ""
    requires: list[str] = field(default_factory=list)  # extra repo files the hooks need in job code staging
    serve: ServeSettings = field(default_factory=ServeSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)
    sft: SFTSettings = field(default_factory=SFTSettings)
    merge: MergeSettings = field(default_factory=MergeSettings)
    grpo: GRPOSettings = field(default_factory=GRPOSettings)

    @property
    def slug(self) -> str:
        return slugify(self.name)

    def missing_requirements(self, root: Path) -> list[str]:
        return [name for name in self.requires if not (root / name).exists()]


def slugify(name: str) -> str:
    return "".join(char for char in name.lower().replace(".", "") if char.isalnum() or char == "-")


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        default = fields[key].default_factory() if fields[key].default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value, f"{where}.{key}")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _load(kind: str, name: str, cls: type) -> Any:
    path = Path(name) if name.endswith(".toml") else PROFILES_DIR / kind / f"{name}.toml"
    if not path.exists():
        available = ", ".join(available_profiles(kind)) or "none"
        raise FileNotFoundError(f"No {kind[:-1]} profile {name!r} (available: {available})")
    with path.open("rb") as handle:
        return _build(cls, tomllib.load(handle), str(path))


def load_teacher(name: str) -> Teacher:
    return _load("teachers", name, Teacher)


def load_student(name: str) -> Student:
    return _load("students", name, Student)


def available_profiles(kind: str) -> list[str]:
    return sorted(path.stem for path in (PROFILES_DIR / kind).glob("*.toml"))
