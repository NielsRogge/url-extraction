#!/usr/bin/env bash
# Recreate SFT v2 + GRPO v2, publish it, evaluate it, and serve it in the demo Space.
#
# Needs an HF_TOKEN that can start Jobs in the `huggingface` namespace, write nielsr repos and
# buckets, and manage nielsr Inference Endpoints, plus ~10 GB of free private storage.
#
#   scripts/grpo_v2_full.sh launch      # one 2x H200 job (~1.5 h, ~$15); prints the job id
#   scripts/grpo_v2_full.sh serve       # after the job completes: serve the model on the demo Space's GPU
set -euo pipefail
cd "$(dirname "$0")/.."

NS=huggingface
BUCKET=nielsr/jobs-artifacts
MERGED=nielsr/qwen35-4b-url-agent-grpo-v2-merged
ADAPTER=nielsr/qwen35-4b-url-agent-grpo-v2-lora
EVAL_REPO=nielsr/qwen35-4b-url-agent-eval-v2
SPACE=nielsr/url-agent-race

case "${1:-}" in
launch)
  REV=$(date +%Y%m%d-%H%M)
  for f in train_grpo_harness.py evaluate_vllm.py merge_adapter.py; do
    hf buckets cp "$f" "hf://buckets/$BUCKET/url-grpo/code/$REV/$f" >/dev/null
  done
  # Prompt pool: train papers the teacher got wrong, skipped protocol on, or that have a verified project page.
  hf buckets cp scripts/rl-v2-ids.txt "hf://buckets/$BUCKET/url-grpo/code/$REV/rl-v2-ids.txt" >/dev/null
  CMD="set -e; pip install -q uv; cp -r /artifacts/url-grpo/code/$REV /work; cd /work
uv run --with torchvision --with pillow merge_adapter.py --adapter nielsr/qwen35-4b-url-agent-lora-v2 --hub-model-id '' --output /model
uv run train_grpo_harness.py --launch --model /model --include-ids-file rl-v2-ids.txt --protocol-weight 0.5 --num-generations 8 --max-concurrent-rollouts 32 --max-steps 100 --save-steps 50 --gradient-accumulation-steps 2 --report-to none --output-dir /grpo-run --hub-model-id $ADAPTER
uv run --with torchvision --with pillow merge_adapter.py --base-model /model --adapter /grpo-run --hub-model-id $MERGED --output /grpo-merged
for s in test validation; do uv run evaluate_vllm.py --split \$s --limit 50 --base-model /model --adapter /grpo-run --concurrency 8 --episode-timeout 180 --max-tool-calls 15 --max-model-len 65536 --max-output-tokens 1024 --output /eval/\$s.jsonl --output-repo $EVAL_REPO --tag sft2-grpo/; done
echo ALL_DONE"
  hf jobs run --namespace "$NS" --name url-grpo-v2-full --flavor h200x2 --timeout 6h --secrets HF_TOKEN \
    --volume "hf://buckets/$BUCKET:/artifacts" --detach huggingface/trl -- bash -c "$CMD"
  ;;
serve)
  # Serve the merged GRPO model on the demo Space's own GPU (no Inference Endpoint).
  python3 - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
sha = api.model_info("$MERGED").sha
print(f"serving $MERGED@{sha}")
api.add_space_variable("$SPACE", "STUDENT_MODEL", "$MERGED")
api.add_space_variable("$SPACE", "HF_BILL_TO", "huggingface")
api.upload_folder(repo_id="$SPACE", repo_type="space", folder_path="space",
                  ignore_patterns=["__pycache__/*", ".gitignore"], commit_message="Serve the GRPO student on this Space")
api.request_space_hardware("$SPACE", "l4x1", sleep_time=3600)
print("Space updated; hardware l4x1, sleeps after 1 h idle")
PY
  ;;
*) echo "usage: $0 launch|serve" >&2; exit 2 ;;
esac
