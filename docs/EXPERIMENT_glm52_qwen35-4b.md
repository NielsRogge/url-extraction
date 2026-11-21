# Experiment: GLM-5.2 teacher -> Qwen3.5-4B student

The first and most complete run of the recipe: distil verified OpenCode trajectories from GLM-5.2
(served through Hugging Face Inference Providers), fine-tune `Qwen/Qwen3.5-4B` with SFT, evaluate in the
same harness, then continue with GRPO (TRL + OpenEnv). It ran in two phases: August 2026 (SFT v1,
Inference Endpoint deployment) and November 2026 (more traces, GRPO with the new OpenEnv/TRL stack).

Deployment and production-batch notes for this model are in [deployment.md](deployment.md).

## August 2026: SFT v1

SFT first teaches the small model the tool protocol, research policy, and final JSON contract;
this is preferable to starting GRPO from the base model.

### 1. Generate verified OpenCode traces

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
default teacher is `huggingface/zai-org/GLM-5.3` (the traces and results below were produced with
GLM-5.2, its predecessor); override `--model` with any
`provider/model` string shown by `opencode models`.

Routed inference is billed to the `huggingface` organization by default; override
it with `HF_BILL_TO=<org>` in `.env` (or pass `--hf-bill-to`). The harness adds
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

### 2. Supervised QLoRA fine-tuning

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
  --namespace huggingface \
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

### 3. Evaluate before RL

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
  --namespace huggingface \
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

The untouched 42-paper test split independently confirmed the gain. Across
the 84 paired OpenCode episodes, the SFT adapter improved field accuracy from
**56.0% to 76.2%** and exact paper-level match from **17/42 to 28/42**
(+26.2 percentage points). GitHub accuracy rose from 30/42 to 35/42, project
page accuracy from 17/42 to 29/42, complete episodes from 21/42 to 27/42, and
mean tool calls fell from 6.24 to 3.98. Tool-call execution reliability was
similar (96.2% base versus 95.2% SFT). The test records and machine-readable
summary are published in
[`nielsr/qwen35-4b-url-agent-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-eval),
and the completed test Job is
[`6a75c0e03e1f34a7e32bd241`](https://huggingface.co/jobs/nielsr/6a75c0e03e1f34a7e32bd241).

After the held-out adapter comparison passes, merge the LoRA into a standalone
Transformers checkpoint on a GPU Job:

```bash
hf jobs uv run \
  --namespace huggingface \
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

## November 2026: teacher parity, and SFT -> RL with the new OpenEnv/TRL stack

All numbers in this section were measured on the same day in the same OpenCode harness
(180 s episode budget, 15 tool calls, 64k context), on the fully scoreable papers of the
held-out splits. The base model scores much higher than in August (test field accuracy
56.0% -> 78.6%): current HF paper metadata links far more repositories, so the August
numbers above are not comparable with these.

| Model | Test field acc. (84 fields) | Test exact (42) | Test tool calls | Val field acc. (100 fields) | Val exact (50) | Val tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B base | 78.6% | 26 | 8.4 | 71.0% | 30 | 6.5 |
| SFT v1 (102 traces, merged) | 83.3% | 29 | 3.2 | 88.0% | 40 | 2.6 |
| SFT v1 + GRPO (100 steps) | 83.3% | 29 | 2.3 | 90.0% | 40 | 2.2 |
| **SFT v2 (873 traces)** | **85.7%** | **32** | 4.7 | **90.0%** | **40** | 3.7 |
| SFT v2, merged (paired control for the row below) | 84.5% | 32 | 5.3 | 90.0% | 40 | 3.9 |
| **SFT v2 + GRPO v2** (harder prompts, protocol-aware reward) | 84.5% | 30 | **3.7** | **93.0%** | **43** | **3.5** |
| GLM-5.2 teacher | 83.3% | 32 | 4.7 | - | - | - |

With 42-50 papers per split, differences of a few points are within noise. The supported
conclusion is that the 4B student now matches the frontier teacher on this benchmark; it is
not that it beats it. Records and summaries:
[`nielsr/qwen35-4b-url-agent-eval-v2`](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-eval-v2)
(`sft-v2/`, `sft1-grpo/`). The teacher costs about $0.03 per paper through Inference Providers.

### Demo Space: student vs. teacher, side by side

[`nielsr/url-agent-race`](https://huggingface.co/spaces/nielsr/url-agent-race) (private, source in
`space/`) races the fine-tuned student (served on the Space's own GPU) against GLM-5.3 on any arXiv
paper. Both run in real OpenCode with the evaluation prompt, tools and schema, and each column streams
reasoning and tool calls with time, tokens and cost. The agents never see a credential: a local proxy
attaches `HF_TOKEN` and the bill-to header. The Space needs an `HF_TOKEN` secret (scoped to Inference
Providers billing and endpoint inference) before it can run.

### SFT v2: more teacher traces

`generate_opencode_traces.py --accept-partial` also distills papers where only some requested
fields carry verified labels (1,732 instead of 346 train papers). One GLM-5.2 attempt per paper
gave 873 accepted trajectories (previously 102), published as `train-v2.jsonl` in
`nielsr/paper-url-opencode-traces`. SFT v2 trains `Qwen/Qwen3.5-4B` in bf16 for one epoch with a
32k max length on one H200 (~55 min), producing
[`nielsr/qwen35-4b-url-agent-lora-v2`](https://huggingface.co/nielsr/qwen35-4b-url-agent-lora-v2).
`train_sft.py --full-trajectory` supervises every assistant turn of a trajectory in one example,
which trains on identical text with roughly half the forwarded tokens.

### GRPO with TRL `HarnessAsyncGRPOTrainer` + OpenEnv capture

`train_grpo_harness.py` replaces the old one-step smoke script:

- OpenCode runs unmodified (one `opencode serve` per rollout, own workspace and database) and reaches
  the policy only through OpenEnv's capture proxy, with the capture session id as its API key.
- TRL 1.18.1's `HarnessAsyncGRPOTrainer` trains on the proxy's validated `TrainingTrace`
  (engine token ids, logprobs, loss masks; nothing is re-tokenized).
- The policy is a LoRA (r=32) on the merged SFT checkpoint. Each new adapter version is loaded into
  vLLM by name, which sidesteps the Qwen3.5 full-weight reload problem (see [legacy-train-opencode.md](legacy-train-opencode.md)).
- Reward: fraction of verified URL fields correct, plus `0.1 * (1 - tool_calls / 15)` when all are
  correct. A rule-based guard (in the spirit of safeagents) scores 0 for any authored tool input
  that looks up the label dataset or touches `HF_TOKEN`.
- OpenCode's structured-output mode forces `tool_choice: "required"`, which OpenEnv refuses to
  train on (constrained logits). A small ASGI middleware rewrites it to `"auto"`; a turn that
  answers in prose fails OpenCode's structured-output check and earns no reward.
- vLLM 0.33 runs from its own `uvx` environment because it and OpenEnv 0.14 pin incompatible `mcp`
  majors. Use H200s: TRL selects FlashAttention-4 on RTX PRO 6000 (sm_120), which fails there.

```bash
hf jobs run --namespace huggingface --flavor h200x2 --timeout 4h --secrets HF_TOKEN \
  --volume hf://buckets/nielsr/jobs-artifacts:/artifacts --detach huggingface/trl -- \
  bash -c "pip install -q uv && cp -r /artifacts/url-grpo/code/REV /work && cd /work && \
    uv run train_grpo_harness.py --launch --num-generations 8 --max-concurrent-rollouts 32 \
      --max-steps 100 --gradient-accumulation-steps 2 --report-to none \
      --hub-model-id nielsr/qwen35-4b-url-agent-grpo-lora"
```

The 100-step run from SFT v1 took 58 minutes on 2x H200 (importance ratio ~0.9997). It kept
accuracy and cut tool calls by 29% on test and 17% on validation, mirroring the harness-adapter
results. Two caveats:

- Training reward barely moved. Most training papers are solved by all eight rollouts, so the
  efficiency bonus carried most of the signal. A follow-up run should sample papers the policy
  solves only sometimes.
- The policy learned to skip parts of the research protocol: full-protocol episodes fell from 36/42
  to 25/42 on test, mostly by not opening the paper. URL accuracy did not suffer, but if that
  protocol matters it has to be part of the reward.

### GRPO v2: from SFT v2, harder prompts, protocol in the reward

The second GRPO run starts from the merged SFT v2 checkpoint and fixes both caveats of the first:

- Prompts (`--include-ids-file`) are the 1,123 train papers where the teacher got a URL wrong,
  skipped part of the research protocol, or that have a verified project page. The 599 GitHub-only
  papers the teacher solved cleanly (and SFT v2 trained on) are excluded.
- `--protocol-weight 0.5` halves the credit for correct URLs found without the research protocol
  (`derive_attempt_complete`, the evaluator's own check), and the efficiency bonus requires it.

Training reward rose from 0.59 (steps 0-20) to 0.94-1.0 (steps 60-100), with within-group reward
std of 0.42 at the start. The run took 43 minutes on 2x H200. Against its own merged SFT v2
starting point, GRPO v2 cut tool calls by 29% on test and 10% on validation, raised validation
field accuracy from 90% to 93% (exact 40 -> 43 of 50), kept test field accuracy at 84.5%
(exact 32 -> 30 of 42), and increased full-protocol episodes (test 32 -> 38, validation 44 -> 45).
Unlike GRPO v1, it became more efficient without dropping research steps. These gaps are within
the noise of 42-50 paper splits.

Because the account's private Hub and bucket storage was full, merge, GRPO and evaluation ran in
one job on local disk (`--hub-model-id ''`, `--output-repo ''`), and the summaries were read from
the job log. The GRPO v2 adapter was therefore not persisted; rerunning requires free storage.

### GRPO v2, retrained and published

After storage was freed, GRPO v2 was retrained with checkpoint pushes to public repos:
[`nielsr/qwen35-4b-url-agent-grpo-v2-lora`](https://huggingface.co/nielsr/qwen35-4b-url-agent-grpo-v2-lora)
and the merged [`nielsr/qwen35-4b-url-agent-grpo-v2-merged`](https://huggingface.co/nielsr/qwen35-4b-url-agent-grpo-v2-merged)
(the model served in the demo Space), via the former `scripts/grpo_v2_full.sh` (now
`pipeline.py --stages grpo,eval`, see [pipeline.md](pipeline.md#reproducing-the-existing-pairs)). Its evaluation against the merged
SFT v2 checkpoint was mixed: test field accuracy 79.8% -> 82.1%, validation 93.0% -> 87.0%, with 25-35% fewer
tool calls. As with the first run, accuracy differences are within the noise of 42-50 paper splits; the
consistent effect of GRPO is shorter episodes.
