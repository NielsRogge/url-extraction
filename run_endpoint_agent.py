#!/usr/bin/env python3
"""Run the URL-extraction OpenCode harness against a managed HF Endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from url_extraction_agent.opencode_endpoint import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    attempt_from_episode,
    endpoint_config,
    run_candidates,
)
from url_extraction_agent.schema import PaperCandidate

__all__ = ["attempt_from_episode", "endpoint_config", "unscored_row"]


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
    parser.add_argument("--endpoint-startup-timeout", type=float, default=900)
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
