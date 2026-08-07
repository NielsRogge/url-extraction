#!/usr/bin/env python3
# /// script
# dependencies = [
#   "huggingface-hub>=1.13.0",
#   "mlx-lm>=0.30.7",
#   "python-dotenv>=1.1.0",
# ]
# ///
"""Convert the merged URL agent to a local 4-bit MLX model on Apple silicon."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

MERGED_MODEL = "nielsr/qwen35-4b-url-agent-merged"


def tokenizer_id(tokenizer: dict, token: str) -> int:
    for item in tokenizer.get("added_tokens", []):
        if item.get("content") == token:
            return int(item["id"])
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if token in vocab:
        return int(vocab[token])
    raise ValueError(f"Token {token!r} is missing from tokenizer.json")


def repair_special_token_ids(output: Path) -> None:
    """Align MLX stop IDs with the tokenizer strings used by Qwen's chat template."""
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())
    tokenizer = json.loads((output / "tokenizer.json").read_text())
    replacements = {
        "eos_token_id": tokenizer_id(tokenizer, tokenizer_config["eos_token"]),
        "pad_token_id": tokenizer_id(tokenizer, tokenizer_config["pad_token"]),
    }
    for filename in ("config.json", "generation_config.json"):
        path = output / filename
        config = json.loads(path.read_text())
        config.update(replacements)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main(args: argparse.Namespace) -> None:
    from mlx_lm import convert

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX conversion must run on an Apple-silicon Mac")
    load_dotenv()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # MLX-LM downloads only model files, but its save path later requests a
    # complete offline snapshot (including README and .gitattributes). Resolve
    # the full snapshot first and pass the local path to avoid that mismatch.
    model_path = snapshot_download(args.model)
    convert(
        model_path,
        mlx_path=str(args.output),
        quantize=True,
        q_bits=args.bits,
        q_group_size=args.group_size,
    )
    repair_special_token_ids(args.output)
    if args.upload_repo:
        api = HfApi()
        api.create_repo(args.upload_repo, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.upload_repo,
            folder_path=str(args.output),
            commit_message="Publish corrected MLX URL-agent checkpoint",
        )
    print(f"Converted local model: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MERGED_MODEL)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen35-4b-url-agent-mlx-4bit"))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--upload-repo", help="Optional Hub repo for the converted MLX model")
    parser.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
