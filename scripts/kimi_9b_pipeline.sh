#!/usr/bin/env bash
# Kimi-K3.1-Pro teacher -> Qwen3.5-9B student: SFT + eval, then GRPO + eval, on HF Jobs.
#
# Needs an HF_TOKEN that can start Jobs in the `huggingface` namespace and write nielsr repos/buckets.
# The merged Kimi traces must already be at hf://buckets/nielsr/url-agent-kimi/data/train-kimi-k31.jsonl.
#
#   scripts/kimi_9b_pipeline.sh sft    # 1x H200: SFT (full-trajectory, bf16, 32k) + eval base vs SFT on test/validation
#   scripts/kimi_9b_pipeline.sh grpo   # 2x H200: merge SFT, GRPO, merge GRPO, eval GRPO vs merged SFT
#   scripts/kimi_9b_pipeline.sh grpo-finish  # 1x H200: redo merge + eval from the pushed GRPO adapter
set -euo pipefail
cd "$(dirname "$0")/.."

NS=huggingface
BUCKET=nielsr/url-agent-kimi  # public bucket: the account's private storage is full
BASE=Qwen/Qwen3.5-9B
SFT_ADAPTER=nielsr/qwen35-9b-url-agent-lora-kimi
GRPO_ADAPTER=nielsr/qwen35-9b-url-agent-grpo-lora-kimi
GRPO_MERGED=nielsr/qwen35-9b-url-agent-grpo-merged-kimi
EVAL_REPO=nielsr/qwen35-9b-url-agent-eval
IDS=${IDS:-scripts/rl-kimi-9b-ids.txt}
EVAL_FLAGS="--limit 50 --concurrency 8 --episode-timeout 180 --max-tool-calls 15 --max-model-len 65536 --max-output-tokens 1024"

stage_code() {
  REV=kimi9b-$(date +%Y%m%d-%H%M)
  for f in train_sft.py train_grpo_harness.py evaluate_vllm.py merge_adapter.py "$IDS"; do
    hf buckets cp "$f" "hf://buckets/$BUCKET/code/$REV/$(basename "$f")" >/dev/null
  done
  echo "$REV"
}

case "${1:-}" in
sft)
  REV=$(stage_code)
  CMD="set -e; pip install -q uv; cp -r /artifacts/code/$REV /work; cd /work
uv run train_sft.py --traces /artifacts/data/train-kimi-k31.jsonl --model $BASE --no-4bit --max-length 32768 --epochs 1 --full-trajectory --save-steps 50 --output /sft --hub-model-id $SFT_ADAPTER
for s in test validation; do uv run evaluate_vllm.py --split \$s $EVAL_FLAGS --base-model $BASE --adapter /sft --output /eval/\$s.jsonl --output-repo $EVAL_REPO --tag sft/; done
echo ALL_DONE"
  hf jobs run --namespace "$NS" --name url-kimi-9b-sft --flavor h200 --timeout 8h --secrets HF_TOKEN \
    --volume "hf://buckets/$BUCKET:/artifacts" --detach huggingface/trl -- bash -c "$CMD"
  ;;
grpo)
  REV=$(stage_code)
  IDS_NAME=$(basename "$IDS")
  CMD="set -e; pip install -q uv; cp -r /artifacts/code/$REV /work; cd /work
uv run --with torchvision --with pillow merge_adapter.py --base-model $BASE --adapter $SFT_ADAPTER --hub-model-id '' --output /model
uv run train_grpo_harness.py --launch --model /model --include-ids-file $IDS_NAME --protocol-weight 0.5 --num-generations 8 --max-concurrent-rollouts \${ROLLOUTS:-32} --max-steps 100 --save-steps 50 --gradient-accumulation-steps 2 --report-to none --output-dir /grpo-run --hub-model-id $GRPO_ADAPTER
uv run --with torchvision --with pillow merge_adapter.py --base-model /model --adapter /grpo-run --card-base-model $BASE --hub-model-id $GRPO_MERGED --output /grpo-merged
for s in test validation; do uv run evaluate_vllm.py --split \$s $EVAL_FLAGS --base-model /model --adapter /grpo-run --output /eval/\$s.jsonl --output-repo $EVAL_REPO --tag sft-grpo/; done
echo ALL_DONE"
  hf jobs run --namespace "$NS" --name url-kimi-9b-grpo --flavor h200x2 --timeout 8h --secrets HF_TOKEN \
    -e ROLLOUTS="${ROLLOUTS:-32}" --volume "hf://buckets/$BUCKET:/artifacts" --detach huggingface/trl -- bash -c "$CMD"
  ;;
grpo-finish)
  # Recovery path when GRPO trained and pushed $GRPO_ADAPTER but a later step failed: no retraining.
  REV=$(stage_code)
  CMD="set -e; pip install -q uv; cp -r /artifacts/code/$REV /work; cd /work
uv run --with torchvision --with pillow merge_adapter.py --base-model $BASE --adapter $SFT_ADAPTER --hub-model-id '' --output /model
uv run --with torchvision --with pillow merge_adapter.py --base-model /model --adapter $GRPO_ADAPTER --card-base-model $BASE --hub-model-id $GRPO_MERGED --output /grpo-merged
for s in test validation; do uv run evaluate_vllm.py --split \$s $EVAL_FLAGS --base-model /model --adapter $GRPO_ADAPTER --output /eval/\$s.jsonl --output-repo $EVAL_REPO --tag sft-grpo/; done
echo ALL_DONE"
  hf jobs run --namespace "$NS" --name url-kimi-9b-grpo-finish --flavor h200 --timeout 4h --secrets HF_TOKEN \
    --volume "hf://buckets/$BUCKET:/artifacts" --detach huggingface/trl -- bash -c "$CMD"
  ;;
*) echo "usage: $0 sft|grpo|grpo-finish" >&2; exit 2 ;;
esac
