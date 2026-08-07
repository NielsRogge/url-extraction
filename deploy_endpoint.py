#!/usr/bin/env python3
# /// script
# dependencies = [
#   "httpx>=0.27.0",
#   "huggingface-hub>=1.13.0",
#   "python-dotenv>=1.1.0",
# ]
# ///
"""Deploy the merged URL agent as a managed vLLM Inference Endpoint."""

from __future__ import annotations

import argparse
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from huggingface_hub import HfApi

API_ROOT = "https://api.endpoints.huggingface.cloud/v2/endpoint"
DEFAULT_NAME = "qwen35-4b-url-agent-vllm"
DEFAULT_REPO = "nielsr/qwen35-4b-url-agent-merged"


def endpoint_payload(args: argparse.Namespace, revision: str) -> dict[str, Any]:
    return {
        "name": args.name,
        "type": "authenticated",
        "provider": {"vendor": args.vendor, "region": args.region},
        "compute": {
            "accelerator": "gpu",
            "instanceType": args.instance_type,
            "instanceSize": args.instance_size,
            "scaling": {
                "minReplica": 0,
                "maxReplica": 1,
                "scaleToZeroTimeout": args.scale_to_zero_timeout,
                "measure": {"pendingRequests": 1.0},
            },
        },
        "model": {
            "repository": args.repo,
            "revision": revision,
            "task": "text-generation",
            "framework": "pytorch",
            "image": {
                "vLLM": {
                    "url": args.vllm_image,
                    "healthRoute": "/health",
                    "port": 8000,
                    "tensorParallelSize": 1,
                    "dataParallelSize": 1,
                    "maxNumSeqs": args.max_num_seqs,
                }
            },
            "args": [
                "--max-model-len",
                str(args.max_model_len),
                "--language-model-only",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
            ],
            "env": {},
            "secrets": {},
        },
        "tags": ["url-extraction", "qwen35", "sft"],
    }


def main(args: argparse.Namespace) -> None:
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required and must grant Inference Endpoint read/write access")
    headers = {"Authorization": f"Bearer {token}"}
    endpoint_url = f"{API_ROOT}/{args.namespace}/{args.name}"
    with httpx.Client(headers=headers, timeout=30) as client:
        existing = client.get(endpoint_url)
        if existing.status_code == 200:
            if args.update:
                revision = args.revision or HfApi(token=token).model_info(args.repo).sha
                desired = endpoint_payload(args, revision)
                response = client.put(
                    endpoint_url,
                    json={
                        "model": {
                            "revision": revision,
                            "image": desired["model"]["image"],
                            "args": desired["model"]["args"],
                        },
                        "compute": {"scaling": desired["compute"]["scaling"]},
                    },
                )
                response.raise_for_status()
                status = response.json().get("status") or {}
                print({"name": args.name, "state": status.get("state"), "url": status.get("url"), "updated": True})
                return
            status = existing.json().get("status") or {}
            print({"name": args.name, "state": status.get("state"), "url": status.get("url"), "existing": True})
            return
        if existing.status_code != 404:
            existing.raise_for_status()

        revision = args.revision or HfApi(token=token).model_info(args.repo).sha
        response = client.post(
            f"{API_ROOT}/{args.namespace}",
            json=endpoint_payload(args, revision),
        )
        response.raise_for_status()
        data = response.json()
        status = data.get("status") or {}
        print({"name": data.get("name"), "state": status.get("state"), "url": status.get("url")})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--namespace", default="nielsr")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision")
    parser.add_argument("--vendor", default="aws")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="nvidia-l4")
    parser.add_argument("--instance-size", default="x1")
    parser.add_argument("--vllm-image", default="vllm/vllm-openai:v0.26.0")
    parser.add_argument("--max-model-len", type=int, default=65_536)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--scale-to-zero-timeout", type=int, default=15)
    parser.add_argument("--update", action="store_true", help="Update an existing Endpoint to the desired config")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
