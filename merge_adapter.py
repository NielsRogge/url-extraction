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
"""Merge the URL-agent LoRA into Qwen3.5 and publish a standalone HF model."""

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
    return f"""---
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3.5
- peft
- merged
---

# Qwen3.5 4B URL extraction agent

Standalone merged checkpoint of [{adapter}](https://huggingface.co/{adapter}),
fine-tuned by supervised learning on verified OpenCode URL-extraction traces.

The model is intended to run inside the OpenCode harness with Bash, Read,
Grep, Glob, WebFetch, and structured-output tools. See the source repository
for the system prompt and held-out base-versus-adapter evaluation.
"""


def merge(args: argparse.Namespace) -> Path:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be passed as a Job secret")

    adapter_snapshot = Path(snapshot_download(args.adapter, token=token))
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
    (output / "README.md").write_text(model_card(args.base_model, args.adapter))

    api = HfApi(token=token)
    api.create_repo(args.hub_model_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output),
        repo_id=args.hub_model_id,
        repo_type="model",
        commit_message="Publish merged URL-agent checkpoint",
    )
    info: Any = api.model_info(args.hub_model_id)
    print(f"Published {args.hub_model_id} at commit {info.sha}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER_ID)
    parser.add_argument("--hub-model-id", default=OUTPUT_REPO)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
