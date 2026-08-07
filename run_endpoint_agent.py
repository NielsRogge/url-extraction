#!/usr/bin/env python3
"""Run the URL-extraction OpenCode harness against a managed HF Endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from opencode_ai import AsyncOpencode

from evaluate_vllm import opencode_server, run_one
from url_extraction_agent.schema import AgentAttempt, PaperCandidate, ToolEvent, URLFindings

DEFAULT_PROVIDER_ID = "hf-endpoint"
DEFAULT_MODEL_ID = "nielsr/qwen35-4b-url-agent-merged"


def endpoint_config(base_url: str, model_id: str, provider_id: str = DEFAULT_PROVIDER_ID) -> dict[str, Any]:
    """Build a secret-free OpenCode provider config for an OpenAI-compatible Endpoint."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": [provider_id],
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Hugging Face Inference Endpoint",
                "options": {
                    "baseURL": base_url.rstrip("/") + "/v1",
                    "apiKey": "{env:HF_TOKEN}",
                    "headers": {"X-Scale-Up-Timeout": "600"},
                },
                "models": {
                    model_id: {
                        "name": "Qwen3.5 4B URL Agent SFT",
                        "limit": {"context": 65_536, "output": 1_024},
                    }
                },
            }
        },
    }


def unscored_row(candidate: PaperCandidate) -> dict[str, Any]:
    """Adapt a production candidate to the evaluator's label-agnostic harness path."""
    row = candidate.model_dump(mode="json")
    row.update(
        {
            "github_label_quality": "unknown",
            "project_page_label_quality": "unknown",
            "reference_github_url": None,
            "reference_project_page_url": None,
        }
    )
    return row


def attempt_from_episode(candidate: PaperCandidate, episode: dict[str, Any]) -> AgentAttempt:
    findings = URLFindings.model_validate(episode["findings"]) if episode.get("findings") else None
    return AgentAttempt(
        paper_id=candidate.paper_id,
        arxiv_id=candidate.arxiv_id,
        findings=findings,
        attempt_complete=bool(episode["attempt_complete"]),
        failure_reason=episode.get("failure_reason"),
        result_subtype="success" if findings is not None else "error",
        tool_events=[ToolEvent.model_validate(event) for event in episode.get("tool_events", [])],
    )


async def run_candidates(args: argparse.Namespace, candidates: list[PaperCandidate]) -> list[AgentAttempt]:
    config = endpoint_config(args.endpoint_url, args.model_id, args.provider_id)
    # OpenCode does not expand {env:...} placeholders inside
    # OPENCODE_CONFIG_CONTENT. Replace it only in this child-process config;
    # unlike a config file or auth-store entry, it is never persisted.
    config["provider"][args.provider_id]["options"]["apiKey"] = os.environ["HF_TOKEN"]
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    async with (
        opencode_server(args.opencode_binary) as base_url,
        AsyncOpencode(
            base_url=base_url,
            timeout=httpx.Timeout(args.request_timeout, connect=10),
            max_retries=1,
        ) as client,
    ):
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_candidate(candidate: PaperCandidate) -> AgentAttempt:
            async with semaphore:
                episode = await run_one(
                    client,
                    unscored_row(candidate),
                    (args.provider_id, args.model_id),
                    episode_timeout=args.episode_timeout,
                    max_tool_calls=args.max_tool_calls,
                )
                return attempt_from_episode(candidate, episode)

        return list(await asyncio.gather(*(run_candidate(candidate) for candidate in candidates)))


def load_candidates(args: argparse.Namespace) -> list[PaperCandidate]:
    if args.candidate_json:
        return [PaperCandidate.model_validate_json(args.candidate_json)]
    payload = json.loads(args.input.read_text())
    if not isinstance(payload, list):
        raise TypeError("--input must contain a JSON list of paper candidates")
    return [PaperCandidate.model_validate(item) for item in payload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-json")
    source.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint-url", default=os.environ.get("HF_ENDPOINT_URL"))
    parser.add_argument("--provider-id", default=DEFAULT_PROVIDER_ID)
    parser.add_argument("--model-id", default=os.environ.get("HF_ENDPOINT_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--opencode-binary", default="opencode")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--episode-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    args = parser.parse_args()
    if not args.endpoint_url:
        parser.error("set HF_ENDPOINT_URL or pass --endpoint-url")
    if not os.environ.get("HF_TOKEN"):
        parser.error("set HF_TOKEN; it is used at runtime and never written to OpenCode config")
    return args


def main() -> None:
    load_dotenv()
    args = parse_args()
    attempts = asyncio.run(run_candidates(args, load_candidates(args)))
    payload = [attempt.model_dump(mode="json") for attempt in attempts]
    rendered = json.dumps(payload[0] if args.candidate_json else payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
