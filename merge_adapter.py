#!/usr/bin/env python3
# /// script
# dependencies = [
#   "accelerate==1.14.0",
#   "huggingface-hub>=1.13.0",
#   "peft==0.20.0",
#   "safetensors>=0.5.0",
#   "torch>=2.7.0",
#   "transformers==5.14.1",
#   "torchvision",
#   "pillow",
# ]
# ///
"""Merge a URL-agent LoRA into its base model and publish a standalone HF model.

The default method merges with PEFT (Qwen3.5). A student profile can name another `--merge-method`
(`module:function(args, adapter_snapshot, token) -> output_dir`), e.g. a merge straight into a quantized
checkpoint's BF16 attention tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_ID = "nielsr/qwen35-4b-url-agent-lora"
OUTPUT_REPO = "nielsr/qwen35-4b-url-agent-merged"


def adapter_safetensors_files(snapshot: Path) -> list[Path]:
    index_path = snapshot / "adapter_model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return [snapshot / name for name in sorted(set(index["weight_map"].values()))]
    single_file = snapshot / "adapter_model.safetensors"
    if not single_file.exists():
        raise FileNotFoundError(f"No safetensors adapter weights found in {snapshot}")
    return [single_file]


def audit_visual_lora(snapshot: Path) -> dict[str, int]:
    """Ensure the legacy vision-tower LoRA has no trained output matrices."""
    counts = {"total": 0, "visual": 0, "visual_nonzero_lora_b": 0}
    for path in adapter_safetensors_files(snapshot):
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor_names = handle.keys()
            for name in tensor_names:
                counts["total"] += 1
                if ".visual." not in name:
                    continue
                counts["visual"] += 1
                if ".lora_B." in name and bool(handle.get_tensor(name).count_nonzero().item()):
                    counts["visual_nonzero_lora_b"] += 1
    if counts["visual_nonzero_lora_b"]:
        raise RuntimeError("The adapter contains trained visual LoRA outputs; refusing an unreviewed merge")
    return counts


def verify_adapter_base(snapshot: Path, requested_base: str) -> None:
    config = PeftConfig.from_pretrained(str(snapshot))
    configured_base = str(config.base_model_name_or_path).rstrip("/")
    if configured_base != requested_base.rstrip("/"):
        raise ValueError(f"Adapter expects {configured_base}, not {requested_base}")


def model_card(base_model: str, adapter: str) -> str:
    title = base_model.split("/")[-1]
    return f"""---
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
- peft
- merged
---

# {title} URL extraction agent

Standalone merged checkpoint of [{adapter}](https://huggingface.co/{adapter}),
fine-tuned on verified OpenCode URL-extraction traces.

The model is intended to run inside the OpenCode harness with Bash, Read,
Grep, Glob, WebFetch, and structured-output tools. See the source repository
for the system prompt and held-out base-versus-adapter evaluation.
"""


def is_inkling_adapter(snapshot: Path) -> bool:
    for path in adapter_safetensors_files(snapshot):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if any(".language_model.layers." in name and ".self_attn." in name for name in handle.keys()):
                return True
    return False


def merge_inkling(args: argparse.Namespace, adapter_snapshot: Path, token: str) -> Path:
    """Merge an attention-only Inkling 2 LoRA into the original checkpoint, keeping its quantized experts.

    The checkpoint stores attention in BF16 and only the routed experts in NVFP4/MXFP8, so W += (alpha/r) B A is
    applied to the attention tensors in place and every other file (experts, MTP drafter, quantization config,
    tokenizer) is copied unchanged. vLLM serves the result exactly like the original checkpoint.
    """
    import shutil

    from safetensors.torch import save_file

    from inkling_adapter import convert_name

    base = Path(args.base_model) if Path(args.base_model).is_dir() else Path(snapshot_download(args.base_model, token=token))
    config = json.loads((adapter_snapshot / "adapter_config.json").read_text())
    if config.get("use_rslora") or config.get("use_dora"):
        raise ValueError("Only plain LoRA adapters are supported")
    scale = config["lora_alpha"] / config["r"]
    lora: dict[str, dict[str, torch.Tensor]] = {}
    for path in adapter_safetensors_files(adapter_snapshot):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                converted = convert_name(name).removeprefix("base_model.model.")
                module, kind = converted.split(".lora_")
                lora.setdefault(f"{module}.weight", {})[kind[0]] = handle.get_tensor(name).float()
    output = args.output or Path(tempfile.mkdtemp(prefix="inkling-url-agent-merged-"))
    output.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.name.startswith(".") or item.suffix == ".safetensors" and item.name.startswith("model-"):
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, output / item.name)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(set(lora) - set(index))
    if missing:
        raise KeyError(f"Adapter targets tensors absent from the checkpoint: {missing[:5]}")
    merged = 0
    for shard in sorted(set(index.values())):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(base / shard), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if name in lora:
                    if tensor.dtype != torch.bfloat16:
                        raise TypeError(f"{name} is {tensor.dtype}; only BF16 attention weights can be merged")
                    delta = scale * (lora[name]["B"] @ lora[name]["A"])
                    tensor = (tensor.float() + delta).to(torch.bfloat16)
                    merged += 1
                tensors[name] = tensor
        save_file(tensors, str(output / shard), metadata=metadata)
    if merged != len(lora):
        raise RuntimeError(f"Merged {merged} of {len(lora)} adapter targets")
    print(f"Merged {merged} attention tensors (scale {scale:g}) into {output}", flush=True)
    return output


def inkling_model_card(base_model: str, adapter: str) -> str:
    return f"""---
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
- inkling2
- peft
- merged
---

# Inkling 2 Mini URL extraction agent

[{base_model}](https://huggingface.co/{base_model}) with the attention-only LoRA
[{adapter}](https://huggingface.co/{adapter}) merged into its BF16 attention weights. The routed experts keep
the original NVFP4/MXFP8 weights, so this checkpoint serves exactly like the base model (e.g. the vLLM
`inkling-2-mini` image with `--tool-call-parser inkling --reasoning-parser inkling`).

The model is intended to run inside the OpenCode harness with Bash, Read, Grep, Glob, WebFetch, and
structured-output tools. See the source repository for the system prompt and the held-out evaluation.
"""


# Model card per --merge-method (default: model_card).
MODEL_CARDS = {"merge_adapter:merge_inkling": inkling_model_card}


def merge(args: argparse.Namespace) -> Path:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be passed as a Job secret")

    adapter_snapshot = (
        Path(args.adapter) if Path(args.adapter).is_dir() else Path(snapshot_download(args.adapter, token=token))
    )
    card_base = args.card_base_model or args.base_model
    if args.hub_model_id and Path(card_base).exists():
        raise ValueError("--card-base-model is required to publish a merge of a local --base-model")
    if args.merge_method != "peft":
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from harness.hooks import resolve

        output = Path(resolve(args.merge_method)(args, adapter_snapshot, token))
        card = MODEL_CARDS.get(args.merge_method, model_card)
        (output / "README.md").write_text(card(card_base, args.card_adapter or args.adapter))
        return publish_merged(args, output, token)
    verify_adapter_base(adapter_snapshot, args.base_model)
    audit = audit_visual_lora(adapter_snapshot)
    print(f"Adapter audit: {json.dumps(audit, sort_keys=True)}", flush=True)

    output = args.output or Path(tempfile.mkdtemp(prefix="qwen-url-agent-merged-"))
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_snapshot))
    model = model.merge_and_unload(safe_merge=True, progressbar=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(output)
    # vLLM initializes Qwen3.5's conditional-generation processor even for
    # text-only requests. Preserve the base image/video processor metadata so
    # the standalone merged repository can boot without reaching back to the
    # base model repository.
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(output)
    # The Hub only accepts repo ids as `base_model`; a local merge base (e.g. /model) gets the card's override.
    (output / "README.md").write_text(model_card(card_base, args.card_adapter or args.adapter))
    return publish_merged(args, output, token)


def publish_merged(args: argparse.Namespace, output: Path, token: str) -> Path:
    if not args.hub_model_id:
        print(f"Saved merged checkpoint to {output}; skipping Hub upload", flush=True)
        return output
    api = HfApi(token=token)
    api.create_repo(args.hub_model_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output),
        repo_id=args.hub_model_id,
        repo_type="model",
        commit_message="Publish merged URL-agent checkpoint",
    )
    info: Any = api.model_info(args.hub_model_id, files_metadata=True)
    local = {p.name: p.stat().st_size for p in output.iterdir() if p.is_file()}
    remote = {s.rfilename: s.size for s in info.siblings}
    absent = [name for name, size in local.items() if remote.get(name) != size]
    if absent:
        raise RuntimeError(f"Upload did not land for {absent}")
    print(f"Published {args.hub_model_id} at commit {info.sha} ({len(local)} files verified)", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER_ID)
    parser.add_argument("--hub-model-id", default=OUTPUT_REPO)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--card-base-model", help="Repo id to list as base_model in the model card")
    parser.add_argument("--card-adapter", help="Repo id to link as the adapter in the model card")
    parser.add_argument("--merge-method", default="peft", help="'peft' or module:function from the student profile")
    parser.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
