# Legacy: one-step harness GRPO smoke test (`train_opencode.py`)

Superseded by `train_grpo_harness.py` (TRL `HarnessAsyncGRPOTrainer` + OpenEnv capture), which trains a LoRA
and sidesteps the Qwen3.5 weight-reload limitation described below. Kept for reference.

This repository adapts TRL's harness-training example to
[`nielsr/paper-url-extraction-v1`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v1).
The policy is run as the model behind the real OpenCode agent. OpenCode owns the
tool loop (`bash`, `read`, `write`, and `webfetch`); OpenEnv's transparent proxy
captures token IDs and log-probabilities, and TRL's `AsyncGRPOTrainer` updates
the policy from a deterministic URL verifier.

This is reinforcement learning in the deployment harness, not conventional
supervised fine-tuning. Claude Code is not supported by the upstream harness
integration yet, so this experiment uses OpenCode.


## Local setup

The training file is a self-contained uv script. The local project environment
is only needed for editing and static checks:

```bash
uv venv
uv sync
uv run python -m py_compile train_opencode.py
```

The actual training stack is Linux/CUDA-only. It uses the `huggingface/trl`
image because that image already contains a compatible CUDA/vLLM stack.

## One-step HF Jobs smoke test

The harness trainer uses separate GPUs for vLLM and the trainer. The smallest
current two-GPU Jobs flavor is `a10g-largex2` (currently $3/hour total):

```bash
hf jobs uv run \
  --namespace huggingface \
  --image huggingface/trl \
  --flavor a10g-largex2 \
  --timeout 45m \
  --secrets HF_TOKEN \
  --detach \
  train_opencode.py \
  --launch \
  --model Qwen/Qwen3.5-2B \
  --n-prompts 1 \
  --num-generations 2 \
  --max-inflight 2 \
  --max-steps 1 \
  --skip-weight-sync \
  --max-completion-length 4096
```

Monitor with `hf jobs logs <job-id>` and inspect with
`hf jobs inspect <job-id>`. The smoke test deliberately does not publish a
checkpoint. After it succeeds, add `--output-repo nielsr/<repo-name>` to persist
the trained model. Do not increase the number of steps while using
`--skip-weight-sync`.

The flag is intentionally restricted to one step. vLLM currently cannot
live-reload Qwen3.5 text-model weights: Transformers names them under `model.*`,
while vLLM's conditional-generation wrapper expects `language_model.model.*`.
The smoke remains valid because vLLM and the trainer begin from the identical
checkpoint, but the updated policy cannot be used for another rollout.

## Suggested multi-step fallback

Until vLLM supports live Qwen3.5 text-weight reloads, use the Qwen3 model
validated by TRL's upstream OpenCode example for an actual multi-step run:

```bash
hf jobs uv run \
  --namespace huggingface \
  --image huggingface/trl \
  --flavor a10g-largex2 \
  --timeout 8h \
  --secrets HF_TOKEN \
  --detach \
  train_opencode.py \
  --launch \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --n-prompts 64 \
  --num-generations 4 \
  --max-inflight 4 \
  --max-steps 50 \
  --output-repo nielsr/qwen3-4b-paper-url-opencode-grpo
```

The script disables Qwen3.5 thinking in proxy requests because current vLLM
tool parsing is more reliable in that mode. It also starts vLLM with
`qwen3_coder`, the parser recommended by the Qwen3.5 model card.

The TRL and OpenEnv dependencies are commit-pinned. In particular, both the
OpenEnv core package and `opencode_env` must come from the same OpenEnv commit;
mixing the source harness with the older PyPI core fails during import because
their `ResourceSessionFactory` APIs differ.

OpenCode is also pinned (`1.18.14`) and downloaded directly from its GitHub
release by Python. This avoids depending on `curl`, which is not included in
the current `huggingface/trl` Jobs image.

The vLLM server uses 70% GPU-memory utilization and eager execution. The
remaining headroom is required for TRL's initial NCCL policy-weight transfer;
vLLM's default allocation leaves too little free memory on a 24 GB A10G.
Training uses bitsandbytes' paged 8-bit AdamW; full-precision Adam moment
buffers exceed the remaining memory on the trainer's 24 GB A10G.

The working smoke test completed as HF Job
[`6a7455a56b79c09949c23efa`](https://huggingface.co/jobs/nielsr/6a7455a56b79c09949c23efa):
one optimizer step over a 7,235-token training sequence in 203.5 seconds.
