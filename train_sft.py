#!/usr/bin/env python3
# /// script
# dependencies = [
#   "accelerate==1.14.0",
#   "bitsandbytes>=0.49.0; platform_system == 'Linux'",
#   "datasets>=4.0.0",
#   "peft==0.20.0",
#   "torch>=2.7.0",
#   "transformers==5.14.1",
#   "trl==1.9.2",
# ]
# ///
"""QLoRA SFT for Qwen3.5 from verified OpenCode trajectories."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by glob.",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "Fetch a public URL.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "StructuredOutput",
            "description": "Submit the final validated URL findings.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reasoning": {"type": "string"},
                    "github_url": {"type": ["string", "null"]},
                    "project_page_url": {"type": ["string", "null"]},
                },
                "required": ["reasoning"],
            },
        },
    },
]
MAX_TOOL_CONTENT_CHARS = 8_000


def load_records(path: Path | None, dataset_id: str | None, split: str) -> list[dict[str, Any]]:
    records = []
    if path is not None:
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("accepted"):
                    records.append(record)
    elif dataset_id is not None:
        records = [dict(record) for record in load_dataset(dataset_id, split=split) if record.get("accepted")]
    if not records:
        raise ValueError(f"No accepted trajectories in {path or dataset_id}")
    by_paper_id: dict[int, dict[str, Any]] = {}
    for record in records:
        by_paper_id.setdefault(int(record["paper_id"]), record)
    return list(by_paper_id.values())


def terminal_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop legacy synthetic JSON appended after OpenCode's terminal StructuredOutput call."""
    result: list[dict[str, Any]] = []
    structured_call_id: str | None = None
    for message in messages:
        result.append(message)
        for call in message.get("tool_calls", []):
            if call.get("function", {}).get("name", "").lower() == "structuredoutput":
                structured_call_id = call.get("id")
        if structured_call_id and message.get("role") == "tool" and message.get("tool_call_id") == structured_call_id:
            break
    return result


def render_examples(records: list[dict[str, Any]], tokenizer: Any, max_length: int | None = None) -> Dataset:
    examples: list[dict[str, str]] = []
    for record in records:
        history: list[dict[str, Any]] = []
        for raw_message in terminal_trajectory(record["trajectory"]):
            message = copy.deepcopy(raw_message)
            if message.get("role") == "tool" and len(message.get("content", "")) > MAX_TOOL_CONTENT_CHARS:
                content = message["content"]
                message["content"] = content[:6_000] + "\n[tool output compacted for SFT]\n" + content[-2_000:]
            for tool_call in message.get("tool_calls", []):
                arguments = tool_call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    tool_call["function"]["arguments"] = json.loads(arguments)
            if message.get("role") != "assistant":
                history.append(message)
                continue
            prompt = tokenizer.apply_chat_template(history, tools=TOOLS, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                history + [message], tools=TOOLS, tokenize=False, add_generation_prompt=False
            )
            if not full.startswith(prompt):
                raise ValueError(
                    "Chat template cannot isolate the assistant completion; use a compatible Qwen tokenizer"
                )
            completion = full[len(prompt) :]
            length = None
            if max_length is not None:
                length = len(tokenizer(prompt + completion, add_special_tokens=False).input_ids)
            if completion.strip() and (length is None or length <= max_length):
                examples.append({"prompt": prompt, "completion": completion})
            history.append(message)
    if not examples:
        raise ValueError("Accepted trajectories contained no assistant turns")
    return Dataset.from_list(examples)


def tokenize_examples(dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    """Build an exact completion mask from character offsets, avoiding BPE boundary drift."""
    rows: list[dict[str, list[int]]] = []
    for example in dataset:
        text = example["prompt"] + example["completion"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = list(encoded["input_ids"])
        boundary = len(example["prompt"])
        completion_mask = [int(end > boundary) for _, end in encoded["offset_mapping"]]
        if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
            input_ids.append(tokenizer.eos_token_id)
            completion_mask.append(1)
        if len(input_ids) <= max_length:
            rows.append({"input_ids": input_ids, "completion_mask": completion_mask})
    if not rows:
        raise ValueError("No tokenized examples fit the configured max length")
    return Dataset.from_list(rows)


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    rendered_dataset = render_examples(
        load_records(args.traces, args.trace_dataset, args.trace_split), tokenizer, max_length=args.max_length
    )
    train_dataset = tokenize_examples(rendered_dataset, tokenizer, args.max_length)
    quantization = None
    if not args.no_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    trainer = SFTTrainer(
        model=args.model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
        quantization_config=quantization,
        args=SFTConfig(
            output_dir=str(args.output),
            max_length=args.max_length,
            completion_only_loss=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            bf16=not args.fp16,
            fp16=args.fp16,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="steps",
            save_steps=args.save_steps,
            report_to="none",
            push_to_hub=bool(args.hub_model_id),
            hub_model_id=args.hub_model_id,
            hub_private_repo=bool(args.hub_model_id),
            model_init_kwargs={
                "torch_dtype": torch.bfloat16 if not args.fp16 else torch.float16,
                "trust_remote_code": True,
            },
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    if args.hub_model_id:
        trainer.push_to_hub()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--traces", type=Path)
    source.add_argument("--trace-dataset", help="HF dataset repo containing trace JSONL")
    parser.add_argument("--trace-split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen35-4b-url-agent-lora"))
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--hub-model-id", help="Push the final adapter and checkpoints to this HF model repo")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
