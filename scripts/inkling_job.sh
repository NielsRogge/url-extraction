#!/usr/bin/env bash
# HF Jobs launcher for the Inkling-2-Small -> Inkling-2-Mini run.
#
# Code is staged in a private Hub repo (not the jobs bucket, which shares the account's storage quota)
# under code/<REV>/ and copied into the job. vLLM comes from the model-specific image
# `vllm/vllm-openai:inkling-2-mini` (no vLLM release supports `inkling2_loop` yet); the evaluator's own
# Python dependencies live in a separate uv environment so they cannot disturb the image's vLLM.
#
#   scripts/inkling_job.sh stage                 # upload code, print REV
#   scripts/inkling_job.sh smoke REV             # 1x H200: serve base Mini, 2 OpenCode test episodes
#   scripts/inkling_job.sh eval REV SPLIT TAG [extra evaluate_vllm.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."

NS=huggingface
CODE_REPO=nielsr/inkling-2-mini-url-agent-code
EVAL_REPO=nielsr/inkling-2-mini-url-agent-eval
VLLM_IMAGE=vllm/vllm-openai:inkling-2-mini
MINI=thinkingmachines/Inkling-2-Mini
TRACES_REPO=nielsr/inkling-2-mini-url-agent-traces
TRACES_FILE=${TRACES_FILE:-train-inkling.jsonl}
SFT_REPO=nielsr/inkling-2-mini-url-agent-lora
# The inkling-2-mini image ships Blackwell kernels only (sm_100/sm_103 + an SM12x path); H200 (sm_90)
# fails with "no kernel image is available" in csrc/inkling2/conv_mix.cu. Serve on RTX PRO 6000.
SERVE_FLAVOR=${SERVE_FLAVOR:-rtx-pro-6000}

# Shared job prelude: fetch code, set up the evaluator environment (outside the image's vLLM env).
PRELUDE='set -euo pipefail
unset UV_OVERRIDE UV_INDEX_STRATEGY
command -v uv >/dev/null || python3 -m pip install -q uv
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
vllm --version || true
mkdir -p /work && cd /work
uvx --from "huggingface-hub>=1.13" hf download '"$CODE_REPO"' --include "code/$REV/*" --local-dir /work/stage >/dev/null
cp -r /work/stage/code/$REV/. /work/ && ls /work
LMO=--language-model-only; vllm serve --help=all 2>/dev/null | grep -q -- "--language-model-only" || LMO=--no-language-model-only
EVAL="uv run --no-project --python 3.12 --with datasets>=4.0.0 --with httpx>=0.27.0 --with huggingface-hub>=1.13.0 --with opencode-ai==0.1.0a36 --with pydantic>=2.10.0,<2.13 --with python-dotenv>=1.1.0 --with safetensors>=0.4.0 python evaluate_vllm.py --vllm-binary $(command -v vllm) --tool-call-parser inkling --reasoning-parser inkling $LMO"
'

case "${1:-}" in
stage)
  REV=$(date +%Y%m%d-%H%M%S)
  uv run -q --no-project --with "huggingface-hub>=1.13" python - "$REV" <<'PY'
import sys
from huggingface_hub import HfApi
rev = sys.argv[1]
api = HfApi()
api.create_repo("nielsr/inkling-2-mini-url-agent-code", private=True, exist_ok=True)
files = ["evaluate_vllm.py", "generate_opencode_traces.py", "train_sft.py", "merge_adapter.py",
         "train_grpo_harness.py", "inkling_train_patch.py", "inkling_adapter.py", "scripts/rl-v2-ids.txt"]
files += [f for f in sys.argv[2:]]
for f in files:
    api.upload_file(path_or_fileobj=f, path_in_repo=f"code/{rev}/{f.split('/')[-1]}",
                    repo_id="nielsr/inkling-2-mini-url-agent-code")
landed = {s.rfilename for s in api.model_info("nielsr/inkling-2-mini-url-agent-code").siblings}
missing = [f for f in files if f"code/{rev}/{f.split('/')[-1]}" not in landed]
assert not missing, f"upload did not land: {missing}"
print(rev)
PY
  ;;
smoke)
  REV=${2:?REV}
  CMD="REV=$REV
$PRELUDE
\$EVAL --base-model $MINI --adapter '' --split test --limit 2 --concurrency 2 --no-enforce-eager \
  --episode-timeout 300 --max-tool-calls 15 --max-model-len 65536 --max-output-tokens 4096 \
  --output /eval/smoke.jsonl --output-repo $EVAL_REPO --tag smoke/
python3 - <<'PY'
import json
for line in open('/eval/smoke.jsonl'):
    r = json.loads(line)
    print('EPISODE', r['arxiv_id'], 'failure=', r['failure_reason'], 'tools=', len(r['tool_events']),
          'correct=', r['verification']['all_scoreable_correct'], 'secs=', r['elapsed_seconds'])
    for m in r['opencode_messages']:
        for p in m.get('parts', []):
            if p.get('type') in ('reasoning', 'text'):
                print('  ', p['type'], repr(p.get('text', '')[:300]))
PY
echo SMOKE_DONE"
  hf jobs run --namespace "$NS" --name inkling-mini-smoke --flavor "$SERVE_FLAVOR" --timeout 1h --secrets HF_TOKEN \
    --detach "$VLLM_IMAGE" -- bash -c "$CMD"
  ;;
eval)
  REV=${2:?REV}; SPLIT=${3:?SPLIT}; TAG=${4:?TAG}; shift 4
  CMD="REV=$REV
$PRELUDE
\$EVAL --split $SPLIT --limit 50 --concurrency 8 --episode-timeout 180 --max-tool-calls 15 \
  --max-model-len 65536 --max-output-tokens 4096 --output /eval/$SPLIT.jsonl --output-repo $EVAL_REPO --tag $TAG $*
echo EVAL_DONE"
  hf jobs run --namespace "$NS" --name "inkling-eval-${TAG%/}-$SPLIT" --flavor "$SERVE_FLAVOR" --timeout 3h --secrets HF_TOKEN \
    --detach "$VLLM_IMAGE" -- bash -c "$CMD"
  ;;
sft)
  # 1x H200: LoRA SFT of Inkling-2-Mini on the teacher's accepted trajectories (transformers only, no vLLM).
  REV=${2:?REV}; shift 2
  CMD="set -euo pipefail
command -v uv >/dev/null || pip install -q uv
mkdir -p /work && cd /work
uvx --from 'huggingface-hub>=1.13' hf download $CODE_REPO --include 'code/$REV/*' --local-dir /work/stage >/dev/null
cp -r /work/stage/code/$REV/. /work/
uvx --from 'huggingface-hub>=1.13' hf download $TRACES_REPO $TRACES_FILE --repo-type dataset --local-dir /work >/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv || true
uv run --no-project --python 3.12 --with torch --with transformers==5.24.0 --with trl==1.18.2 --with peft==0.22.0 \
  --with accelerate==1.16.0 --with datasets --with pillow --with torchvision --with safetensors \
  python train_sft.py --traces /work/$TRACES_FILE --model $MINI --full-trajectory --no-4bit --max-length 32768 \
    --epochs 1 --lora-rank 64 --gradient-accumulation 16 --save-steps 10 --save-total-limit 2 \
    --output /out/sft --hub-model-id $SFT_REPO $*
echo SFT_DONE"
  hf jobs run --namespace "$NS" --name inkling-mini-sft --flavor h200 --timeout 8h --secrets HF_TOKEN \
    --detach ghcr.io/astral-sh/uv:python3.12-bookworm -- bash -c "$CMD"
  ;;
*) echo "usage: $0 stage [extra files] | smoke REV | eval REV SPLIT TAG [args...] | sft REV [args...]" >&2; exit 2 ;;
esac
