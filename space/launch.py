"""Container entrypoint: fetch the student model, serve it with vLLM on the GPU, then start the web app.

start.sh hands the HF token over in /tmp/.hf_token and removes it from the environment, so neither
vLLM nor the agents' shell tools can see it. The web app reads and deletes that file at import.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = os.environ.get("STUDENT_MODEL", "nielsr/qwen35-4b-url-agent-grpo-v2-merged")
TOKEN_FILE = Path("/tmp/.hf_token")


def start_vllm() -> subprocess.Popen | None:
    if os.environ.get("STUDENT_URL"):  # an external student server was configured instead
        return None
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
    print(f"[launch] downloading {MODEL}", flush=True)
    path = snapshot_download(MODEL, token=token, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
    env = {k: v for k, v in os.environ.items() if "TOKEN" not in k.upper()}
    command = [
        shutil.which("vllm") or "/usr/local/bin/vllm", "serve", path,
        "--served-model-name", "url-agent",
        "--host", "127.0.0.1", "--port", "8000",
        "--max-model-len", "65536",
        "--gpu-memory-utilization", os.environ.get("GPU_MEMORY_UTILIZATION", "0.88"),
        "--max-num-seqs", "4",
        "--language-model-only",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
    ]  # fmt: skip
    print("[launch] " + " ".join(command), flush=True)
    return subprocess.Popen(command, env=env, start_new_session=True)


if __name__ == "__main__":
    vllm = start_vllm()
    import uvicorn

    import server  # noqa: F401 - reads and deletes the token file

    try:
        uvicorn.run(server.app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), log_level="warning")
    finally:
        if vllm is not None:
            vllm.terminate()
