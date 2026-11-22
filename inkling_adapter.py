"""Convert a PEFT LoRA adapter trained on transformers' Inkling 2 loop model to vLLM's naming.

vLLM's Inkling2 LoRA (`vllm/models/inkling2/nvidia/lora.py`) expects the original checkpoint's module names
(`model.llm.layers.N.attn.wq_du`, ...), while transformers renames them on load
(`model.language_model.layers.N.self_attn.q_proj.linear`, ...). This inverts transformers' `inkling2_text`
renames for the attention projections, the only modules the URL agent's adapters target.

    python inkling_adapter.py SRC_DIR DST_DIR
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

# transformers name (after `model.language_model.layers.N.self_attn.`) -> checkpoint name (after `model.llm.layers.N.attn.`)
ATTENTION_NAMES = {
    "q_proj.linear": "wq_du",
    "k_proj.linear": "wk_dv",
    "v_proj.linear": "wv_dv",
    "o_proj.linear": "wo_ud",
    "o_proj.gate_proj": "gate_attn",
    "relative_position_bias.linear": "wr_du",
    "sink_proj": "attn_sink_proj",
}
_PATTERN = re.compile(
    r"^(?P<prefix>(?:base_model\.model\.)?)model\.language_model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<module>" + "|".join(re.escape(name) for name in ATTENTION_NAMES) + r")\.(?P<rest>lora_[AB]\..*)$"
)


def is_inkling_adapter(tensor_names: list[str]) -> bool:
    return any(".language_model.layers." in name and ".self_attn." in name for name in tensor_names)


def convert_name(name: str) -> str:
    match = _PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unsupported Inkling adapter tensor (only attention projections are servable): {name}")
    module = ATTENTION_NAMES[match["module"]]
    return f"{match['prefix']}model.llm.layers.{match['layer']}.attn.{module}.{match['rest']}"


def convert_adapter(source: str | Path, destination: str | Path) -> Path:
    from safetensors.torch import load_file, save_file

    source, destination = Path(source), Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("adapter_model.safetensors", "checkpoint-*"))
    tensors = load_file(source / "adapter_model.safetensors", device="cpu")
    converted = {convert_name(name): tensor.contiguous() for name, tensor in tensors.items()}
    save_file(converted, destination / "adapter_model.safetensors")
    config_path = destination / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["target_modules"] = sorted({name.split(".")[-3] for name in converted})
    config.pop("target_parameters", None)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"Converted {len(converted)} Inkling LoRA tensors to vLLM checkpoint names: {config['target_modules']}", flush=True)
    return destination


if __name__ == "__main__":
    convert_adapter(sys.argv[1], sys.argv[2])
