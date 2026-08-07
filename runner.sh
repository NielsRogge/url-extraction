#!/bin/bash
set -euo pipefail
cd /code
python -c 'import io,tarfile,urllib.request,os; d=urllib.request.urlopen("https://github.com/anomalyco/opencode/releases/download/v1.18.14/opencode-linux-x64.tar.gz",timeout=180).read(); t=tarfile.open(fileobj=io.BytesIO(d),mode="r:gz"); p=t.extractfile("opencode").read(); os.makedirs("/tmp/opencode-bin",exist_ok=True); open("/tmp/opencode-bin/opencode","wb").write(p); os.chmod("/tmp/opencode-bin/opencode",0o755)'
uv run --with 'vllm>=0.22.0' python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-4B --served-model-name base --enable-lora --lora-modules url-agent=/adapter --max-lora-rank 64 --max-model-len 16384 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser qwen3_coder > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read()" 2>/dev/null; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then cat /tmp/vllm.log; exit 1; fi
  sleep 2
done
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read()" || { cat /tmp/vllm.log; exit 1; }
uv run --with 'datasets>=4' --with 'httpx>=0.27' --with 'opencode-ai==0.1.0a36' --with 'pydantic>=2.10,<2.13' --with 'python-dotenv>=1.1' python evaluate_opencode.py --opencode-command /tmp/opencode-bin/opencode --model local/url-agent --hf-bill-to huggingface --no-native-structured-output --limit 25 --episode-timeout 300 --output /artifacts/url-agent-eval/adapter-heldout.jsonl --summary /artifacts/url-agent-eval/adapter-summary.json
tail -40 /tmp/vllm.log
