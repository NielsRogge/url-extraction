#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface-hub>=1.13.0"]
# ///
"""Deploy the demo Space (space/ plus the shared harness/ package) serving a merged student on its own GPU.

    uv run scripts/serve_space.py --model nielsr/qwen35-4b-url-agent-v3-grpo-merged

The Space stays private (AGENTS.md); the model repo it serves may be public.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Merged student checkpoint to serve")
    parser.add_argument("--space", default="nielsr/url-agent-race")
    parser.add_argument("--hardware", default="l4x1")
    parser.add_argument("--bill-to", default="huggingface")
    args = parser.parse_args()

    api = HfApi()
    print(f"serving {args.model}@{api.model_info(args.model).sha}")
    api.add_space_variable(args.space, "STUDENT_MODEL", args.model)
    api.add_space_variable(args.space, "HF_BILL_TO", args.bill_to)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        shutil.copytree(ROOT / "space", bundle, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "harness", bundle / "harness", ignore=shutil.ignore_patterns("__pycache__", "profiles.py"))
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(bundle),
                          ignore_patterns=[".gitignore"], commit_message=f"Serve {args.model}")  # fmt: skip
    api.request_space_hardware(args.space, args.hardware, sleep_time=3600)
    print(f"Space updated; hardware {args.hardware}, sleeps after 1 h idle")


if __name__ == "__main__":
    main()
