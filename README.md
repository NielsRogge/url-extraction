# Paper URL extraction with OpenCode + OpenEnv + TRL

The practical path is now split into two stages:

1. Distill verified OpenCode trajectories from a strong teacher with
   `generate_opencode_traces.py`.
2. QLoRA fine-tune Qwen3.5-4B with `train_sft.py`, evaluate it in the same
   OpenCode harness, and only then optionally continue with the existing
   harness-level GRPO script.

This is preferable to starting with GRPO from the base model: SFT first teaches
the small model the tool protocol, research policy, and final JSON contract.
The copied Claude implementation remains in `url_extraction_agent/`; the new
OpenCode path imports its Pydantic schema and exact system prompt.

## 1. Generate verified OpenCode traces

Create the uv environment (the OpenCode SDK is currently a prerelease):

```bash
uv venv
uv sync --extra traces --prerelease allow
opencode models
```

Put an inference-enabled Hugging Face token in `.env` as `HF_TOKEN`. Its
fine-grained permission `Make calls to Inference Providers` must be assigned to
your own user account, not only to an organization. The trace
runner loads it without copying the credential into OpenCode's auth store. Its
default teacher is `huggingface/zai-org/GLM-5.2`; override `--model` with any
`provider/model` string shown by `opencode models`.

To charge routed inference to a Team or Enterprise organization, also set
`HF_BILL_TO=your-org-name` in `.env` (or pass `--hf-bill-to`). The harness adds
the required `X-HF-Bill-To` header to OpenCode's Hugging Face provider without
placing the token in its configuration. A fine-grained token must grant
`inference.serverless.write` for that organization.

If the token saved by `hf auth login` has the correct permission, it can be
used in memory without copying it to `.env`:

```bash
uv run --prerelease allow python generate_opencode_traces.py \
  --use-hf-cli-token \
  --limit 25 \
  --output artifacts/traces/train.jsonl
```

```bash
uv run --prerelease allow python generate_opencode_traces.py \
  --limit 25 \
  --attempts-per-row 3 \
  --output artifacts/traces/train.jsonl
```

The script uses the official `opencode-ai` Python SDK to create sessions and
send messages to an isolated `opencode serve` process. It captures raw OpenCode
parts plus a portable chat trajectory. It is resumable: a paper is skipped only
after an accepted trajectory exists.

Only the 346 training rows for which *every requested field* is `gold`,
`silver`, or `verified_null` are selected by default. A trace is accepted only
when all scoreable URLs match and the observed tool calls satisfy the copied
research policy. Failed attempts remain in JSONL for diagnosis but are ignored
by SFT. Use `--include-unscoreable` only for qualitative exploration.

The currently configured free OpenCode model can be used to test plumbing:

```bash
uv run --prerelease allow python generate_opencode_traces.py \
  --model opencode/big-pickle \
  --limit 1 \
  --output artifacts/traces/pilot.jsonl
```

It is not the recommended teacher for the full dataset. Configure GLM-5.2 (or
another model demonstrably stronger than Qwen3.5-4B) in OpenCode first.

## 2. Supervised QLoRA fine-tuning

`train_sft.py` converts every accepted multi-turn trajectory into
prompt/completion examples. Loss is applied only to assistant completions, so
paper text and tool outputs provide context rather than training targets.

Run on a Linux CUDA machine:

```bash
uv run --extra train --prerelease allow python train_sft.py \
  --traces artifacts/traces/train.jsonl \
  --model Qwen/Qwen3.5-4B \
  --output artifacts/qwen35-4b-url-agent-lora
```

Or submit the self-contained uv script to HF Jobs (an L4 is a sensible first
QLoRA run; adjust the flavor if sequence lengths cause OOM):

```bash
hf repos create nielsr/paper-url-opencode-traces --repo-type dataset --private
hf upload nielsr/paper-url-opencode-traces \
  artifacts/traces/train.jsonl train.jsonl \
  --repo-type dataset

hf jobs uv run \
  --image huggingface/trl \
  --flavor l4x1 \
  --timeout 8h \
  --secrets HF_TOKEN \
  --detach \
  train_sft.py \
  --trace-dataset nielsr/paper-url-opencode-traces \
  --model Qwen/Qwen3.5-4B \
  --output /tmp/qwen35-4b-url-agent-lora \
  --hub-model-id nielsr/qwen35-4b-url-agent-lora
```

Keep the trace dataset private until its raw tool outputs have been reviewed.
Do not put teacher API keys into the trace dataset.

## 3. Evaluate before RL

Reserve the dataset's validation and test splits. Compare the base model and
adapter on canonical URL accuracy, research completeness, tool-call success,
and invalid/hallucinated URL rate. Merge or serve the adapter behind OpenCode
only after it improves held-out results. If SFT plateaus, use the existing
`train_opencode.py` for harness-level GRPO.

`evaluate_vllm.py` runs that comparison on one GPU. It serves the base model
and LoRA together with vLLM, points OpenCode's OpenAI-compatible provider at
the local server, and publishes resumable episode records plus a summary to a
private Hub dataset. The first training run attached LoRA tensors to the
unused vision tower; their B matrices remained exactly zero, so the evaluator
verifies that fact and removes only those inert tensors before vLLM loads the
adapter.

```bash
hf jobs uv run \
  --image huggingface/trl \
  --flavor l4x1 \
  --timeout 4h \
  --secrets HF_TOKEN \
  --volume hf://buckets/nielsr/jobs-artifacts:/artifacts \
  --detach \
  evaluate_vllm.py \
  --limit 50 \
  --concurrency 2 \
  --episode-timeout 180 \
  --max-tool-calls 15 \
  --max-model-len 65536 \
  --max-output-tokens 1024 \
  --output /artifacts/qwen35-4b-url-agent-eval/validation-v2-64k.jsonl \
  --output-repo nielsr/qwen35-4b-url-agent-eval
```

The 64k context applies to the whole multi-turn conversation; the 1,024-token
limit applies to each tool-selection or final-answer turn. Accepted teacher
traces used a median of 3 tool calls and a maximum of 15, which is why the
evaluation watchdog uses 15 rather than allowing runaway research loops.

The completed 50-paper paired validation run improved field accuracy from
**64% for the base model to 81% for the SFT adapter**. Exact paper-level match
improved from 28/50 to 35/50, while mean tool calls fell from 5.20 to 3.56.
The underlying 100 episode records are in
[`nielsr/qwen35-4b-url-agent-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-eval),
and the completed Job is
[`6a75ab25da2af92a634ee9d7`](https://huggingface.co/jobs/nielsr/6a75ab25da2af92a634ee9d7).

After the held-out adapter comparison passes, merge the LoRA into a standalone
Transformers checkpoint on a GPU Job:

```bash
hf jobs uv run \
  --flavor l4x1 \
  --timeout 2h \
  --secrets HF_TOKEN \
  merge_adapter.py
```

The merge verifies that the adapter belongs to the requested base model, uses
PEFT's safe merge, and publishes a private model by default. The evaluation is
deliberately the gate for this step: do not package a model that has not
improved over the unfine-tuned baseline. This run published the standalone
[`nielsr/qwen35-4b-url-agent-merged`](https://huggingface.co/nielsr/qwen35-4b-url-agent-merged)
checkpoint from Job
[`6a75b6a1da2af92a634eea37`](https://huggingface.co/jobs/nielsr/6a75b6a1da2af92a634eea37).

On an Apple-silicon Mac, convert the merged model to a local 4-bit MLX model
and serve its OpenAI-compatible endpoint:

```bash
uv run --extra local convert_mlx.py
uv run --extra local mlx_lm.server \
  --model artifacts/qwen35-4b-url-agent-mlx-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 1024
```

Point an OpenCode `@ai-sdk/openai-compatible` provider at
`http://127.0.0.1:8080/v1` and configure its model ID as `default_model`.
MLX is used for the final laptop runtime because vLLM requires CUDA; vLLM
remains the reproducible GPU serving backend for the HF Jobs evaluation.
The checked-in example can be selected without replacing an existing OpenCode
configuration:

```bash
export OPENCODE_CONFIG="$PWD/opencode.local.example.json"
opencode
```

The checked-in converter also corrects Qwen3.5's generated MLX stop-token
metadata so the server stops on `<|im_end|>` instead of continuing to its
per-turn token limit. A real local OpenCode smoke test on held-out paper
`2604.11004` completed in 29 seconds, used Bash plus structured output, and
matched both expected URLs.

## Harness-level GRPO (optional third stage)

This repository adapts TRL's harness-training example to
[`nielsr/paper-url-extraction-v1`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v1).
The policy is run as the model behind the real OpenCode agent. OpenCode owns the
tool loop (`bash`, `read`, `write`, and `webfetch`); OpenEnv's transparent proxy
captures token IDs and log-probabilities, and TRL's `AsyncGRPOTrainer` updates
the policy from a deterministic URL verifier.

This is reinforcement learning in the deployment harness, not conventional
supervised fine-tuning. Claude Code is not supported by the upstream harness
integration yet, so this experiment uses OpenCode.

## Important dataset semantics

The verifier only scores a URL field when its label quality is `gold`, `silver`,
or `verified_null`. An `unknown` label is excluded from the reward; it is never
treated as a negative target. Of the 4,000 training rows, 1,732 have at least
one scoreable target (1,628 GitHub and 450 project-page targets).

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
