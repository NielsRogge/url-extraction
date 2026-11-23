#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub>=1.13.0"]
# ///
"""Deploy the demo Space (space/ plus the shared harness/ package).

By default the student runs on the managed Inference Endpoint created by deploy_endpoint.py and the
Space itself needs only a CPU: it proxies student calls to the endpoint (`STUDENT_URL`).

    uv run deploy_endpoint.py   # once: vLLM endpoint in the huggingface namespace
    uv run scripts/serve_space.py --model nielsr/qwen35-4b-url-agent-v3-grpo-merged

`--local-gpu` serves the student with vLLM on the Space's own GPU instead:

    uv run scripts/serve_space.py --model nielsr/qwen35-4b-url-agent-v3-grpo-merged --local-gpu

The Space stays private (AGENTS.md); the model repo it serves may be public. Its HF_TOKEN secret must be
allowed to call Inference Endpoints in the endpoint's namespace.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Merged student checkpoint (served by the endpoint or locally)")
    parser.add_argument("--space", default="nielsr/url-agent-race")
    parser.add_argument("--bill-to", default="huggingface")
    parser.add_argument("--endpoint", default="qwen35-4b-url-agent-vllm", help="Inference Endpoint serving the student")
    parser.add_argument("--endpoint-namespace", default="huggingface")
    parser.add_argument("--local-gpu", action="store_true", help="Serve the student with vLLM on the Space's own GPU")
    parser.add_argument("--hardware", help="Space hardware (default: cpu-basic, or l4x1 with --local-gpu)")
    parser.add_argument("--gpu-per-hour", default="0.80", help="Student GPU price (USD/h) for the cost readout; L4 x1")
    args = parser.parse_args()
    hardware = args.hardware or ("l4x1" if args.local_gpu else "cpu-basic")

    api = HfApi()
    print(f"serving {args.model}@{api.model_info(args.model).sha}")
    api.add_space_variable(args.space, "STUDENT_MODEL", args.model)
    api.add_space_variable(args.space, "HF_BILL_TO", args.bill_to)
    api.add_space_variable(args.space, "STUDENT_GPU_PER_HOUR", args.gpu_per_hour)
    if args.local_gpu:
        if "STUDENT_URL" in api.get_space_variables(args.space):
            api.delete_space_variable(args.space, "STUDENT_URL")
    else:
        endpoint = api.get_inference_endpoint(args.endpoint, namespace=args.endpoint_namespace)
        if endpoint.repository != args.model:
            raise SystemExit(f"endpoint {endpoint.name} serves {endpoint.repository}, not {args.model}; "
                             "update it with deploy_endpoint.py --repo ... --update")  # fmt: skip
        if not endpoint.url:
            raise SystemExit(f"endpoint {endpoint.name} has no URL yet (status {endpoint.status})")
        api.add_space_variable(args.space, "STUDENT_URL", endpoint.url)
        print(f"student endpoint {endpoint.url} ({endpoint.status})")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        shutil.copytree(ROOT / "space", bundle, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "harness", bundle / "harness", ignore=shutil.ignore_patterns("__pycache__", "profiles.py"))
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(bundle),
                          ignore_patterns=[".gitignore"], commit_message=f"Serve {args.model}")  # fmt: skip
    if hardware == "cpu-basic":  # free tier; its sleep time cannot be changed
        api.request_space_hardware(args.space, hardware)
        print(f"Space updated; hardware {hardware}")
    else:
        api.request_space_hardware(args.space, hardware, sleep_time=3600)
        print(f"Space updated; hardware {hardware}, sleeps after 1 h idle")


if __name__ == "__main__":
    main()
