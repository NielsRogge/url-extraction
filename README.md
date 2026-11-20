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

### Managed Inference Endpoint (recommended production runtime)

`deploy_endpoint.py` creates an authenticated, OpenAI-compatible vLLM
Inference Endpoint on one L4. It pins the current merged-model revision, caps
the context at the evaluated 65,536 tokens, and scales to zero after 15 idle
minutes. The script is idempotent and will report an existing endpoint rather
than creating a duplicate.

The `HF_TOKEN` used for deployment must grant read and write access to
Inference Endpoints in the target namespace.

```bash
uv run deploy_endpoint.py
hf endpoints describe qwen35-4b-url-agent-vllm --namespace nielsr
```

Use the stable URL returned by `hf endpoints describe` with the production
OpenCode runner. The token remains an environment variable; it is never
written to disk or included in output artifacts. The runner injects it only
into the short-lived OpenCode server process.

```bash
export HF_ENDPOINT_URL="https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud"
uv run --extra traces python run_endpoint_agent.py \
  --input candidates.json \
  --output attempts.json
```

For interactive OpenCode use, set `HF_ENDPOINT_BASE_URL` to the same URL with
`/v1` appended and select the checked-in provider configuration:

```bash
export HF_ENDPOINT_BASE_URL="https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud/v1"
export OPENCODE_CONFIG="$PWD/opencode.endpoint.example.json"
opencode
```

The Endpoint is the intended hosted runtime. OpenCode still owns and executes
the Bash, Read, Grep, Glob, and WebFetch tool loop; only model inference is
remote. Scale-to-zero saves idle GPU cost but introduces a cold start, so a
production caller should retry transient 502/503 responses with a bounded
backoff. The provider sends `X-Scale-Up-Timeout: 600`, allowing Hugging Face's
proxy to hold the first request while a scaled-to-zero replica starts.

The deployed Endpoint passed an end-to-end OpenCode smoke test on held-out
paper `2604.11004`: it executed two `hf papers` Bash calls, returned structured
output, and matched both expected URLs.

### Resumable production batches on HF Jobs

The original `nielsr/paper-url-extraction-v1` dataset is no longer an unseen
canary source: its train split supplied SFT data and its validation and test
splits supplied the paired evaluations. `prepare_canary.py` instead reads the
private `huggingface/community-science-llm-traces` URL-extraction shards,
parses the JSON object from each `output_raw` value, canonicalizes the
`arxiv_id`, and excludes IDs present in any original split. Because the trace
dataset contains repeated model calls, it accepts a field only when at least
three traces agree with at least 60% support. The trace outputs are pseudo-labels,
not gold annotations: `prepare_canary.py` enriches them with current HF paper
metadata, records the provenance for each field, and retains both URLs when the
trace consensus and HF metadata disagree. Evaluation also rejects GitHub URLs
outside `github.com` and project pages on GitHub, arXiv, or Hugging Face.

The current trace snapshot has 2,039 rows and 201 valid arXiv IDs. Six overlap
the original dataset, leaving 195 unseen papers; 168 have consensus labels for
both URL fields. A small canary can be balanced across papers with both URLs,
GitHub only, project page only, and neither:

```bash
uv run --extra traces python prepare_canary.py \
  --source-parquet data/step_1_url_extraction/*.parquet \
  --limit 20 \
  --output canary-v1.jsonl
hf buckets cp canary-v1.jsonl \
  hf://buckets/nielsr/jobs-artifacts/qwen35-4b-url-agent-production/canary-v1-input.jsonl
```

The production batch module runs OpenCode on CPU while the managed L4
Endpoint performs inference. It first wakes the Endpoint and waits up to 15
minutes for `/health`, keeping cold-start time outside every paper's episode
budget. It atomically checkpoints every paper to the mounted bucket, skips
existing results after a restart, and optionally
publishes the completed results and summary to a private Hub dataset. Build a
wheel and store it under a revision directory so Jobs install immutable code
without requiring access to the private GitHub repository:

```bash
uv build --wheel --out-dir /tmp/url-agent-wheel
hf buckets cp /tmp/url-agent-wheel/url_extraction-0.1.0-py3-none-any.whl \
  hf://buckets/nielsr/jobs-artifacts/qwen35-4b-url-agent-production/packages/REVISION/url_extraction-0.1.0-py3-none-any.whl
```

Run that pinned wheel on a CPU Job:

```bash
hf jobs run \
  --flavor cpu-upgrade \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ENDPOINT_URL=https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud \
  --volume hf://buckets/nielsr/jobs-artifacts:/artifacts \
  --detach \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv run \
  --with 'url-extraction[traces] @ file:///artifacts/qwen35-4b-url-agent-production/packages/REVISION/url_extraction-0.1.0-py3-none-any.whl' \
  python -m url_extraction_agent.batch \
  --input /artifacts/qwen35-4b-url-agent-production/canary-v1-input.jsonl \
  --output /artifacts/qwen35-4b-url-agent-production/canary-v1-results.jsonl \
  --output-repo nielsr/qwen35-4b-url-agent-production
```

Pass `--retry-incomplete` only when intentionally rerunning incomplete
episodes. Completed records remain deduplicated by arXiv ID.

After a full run, use the checked-in finalizer to verify that inputs and results
contain the same unique arXiv IDs and to recompute enriched, provenance-aware
scores without another inference call:

```bash
uv run --extra traces python finalize_evaluation.py \
  --input unseen-v2-input.jsonl \
  --results unseen-v2-first-pass-results.jsonl \
  --output-results unseen-v2-first-pass-final-results.jsonl \
  --output-summary unseen-v2-first-pass-summary.json
```

The balanced 20-paper canary produced structured findings for 19 papers and
completed the full research protocol for 15. After enriching the trace labels
with current HF metadata and enforcing URL-field semantics, the unchanged
predictions scored 16/20 exact matches, 16/20 GitHub fields, and 20/20 project
page fields. The original trace-only score was 8/20 exact because five GitHub
repositories and two project pages present in current HF metadata were labelled
null by the older teacher traces. This label audit is therefore required before
using these traces as an evaluation or reward signal.

The complete unseen evaluation contains 168 papers and preserves its one-shot
result before any retry. The immutable first pass and the separate operational
artifact are both published in
[`nielsr/qwen35-4b-url-agent-production`](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-production):

| Metric | Immutable first pass | One operational retry |
| --- | ---: | ---: |
| Exact paper-level match | 134/168 (79.8%) | 138/168 (82.1%) |
| GitHub URL correct | 145/168 (86.3%) | 147/168 (87.5%) |
| Project-page URL correct | 149/168 (88.7%) | 153/168 (91.1%) |
| Structured output | 144/168 (85.7%) | 159/168 (94.6%) |
| Full research protocol | 86/168 (51.2%) | 123/168 (73.2%) |
| Error result | 24/168 (14.3%) | 9/168 (5.4%) |

The first pass ran in
[`6a75dc58da2af92a634eeb68`](https://huggingface.co/jobs/nielsr/6a75dc58da2af92a634eeb68).
After freezing that raw artifact, Job
[`6a7617bc3e1f34a7e32bd914`](https://huggingface.co/jobs/nielsr/6a7617bc3e1f34a7e32bd914)
retried each of the 82 protocol-incomplete episodes exactly once in a separate
file. The operational result is useful for production reliability, but it does
not replace the one-shot benchmark.

Label provenance remains important. On the operational artifact, GitHub
accuracy was 40/40 for HF-metadata labels, 61/65 where traces and HF metadata
agreed, 15/16 on trace/HF conflicts, and 31/47 for trace-only consensus labels.
Project-page accuracy was respectively 14/14, 60/61, 3/3, and 76/90. Retries
improved the high-confidence HF-backed groups while reducing accuracy on a few
trace-only pseudo-labels, reinforcing that these results should not be treated
as a gold benchmark without human adjudication.

### Optional Apple-silicon smoke runtime

MLX was used only to prove that the merged checkpoint could run locally
without CUDA. On an Apple-silicon Mac, convert it to a local 4-bit model and
serve its OpenAI-compatible endpoint:

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

## November 2026 update: teacher parity, and SFT -> RL with the new OpenEnv/TRL stack

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
`space/`) races the fine-tuned student (Inference Endpoint, now 1x A10G) against GLM-5.2 on any arXiv
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
  vLLM by name, which sidesteps the Qwen3.5 full-weight reload problem described below.
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

## November 2026, part 2: Kimi-K3.1-Pro teacher -> Qwen3.5-9B student

The same teacher -> SFT -> GRPO pipeline, with `moonshotai/Kimi-K3.1-Pro` (Inference Providers,
served by Together; OpenCode model `huggingface/moonshotai/Kimi-K3.1-Pro`) as teacher and
`Qwen/Qwen3.5-9B` as student. Harness and metrics are identical to the section above (OpenCode 1.2.4,
180 s episodes, 15 tool calls, 64k context, fully scoreable held-out papers).

| Model | Test field acc. (84 fields) | Test exact (42) | Test tool calls | Val field acc. (100 fields) | Val exact (50) | Val tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B base | 75.0% | 24 | 6.8 | 88.0% | 39 | 3.3 |
| **9B SFT** (815 Kimi traces, adapter) | 85.7% | 30 | 3.8 | **93.0%** | **43** | 2.7 |
| 9B SFT, merged (paired control for the row below) | **88.1%** | 33 | 3.2 | 88.0% | 39 | 2.9 |
| 9B SFT + GRPO (100 steps) | 82.1% | 29 | **3.0** | 88.0% | 39 | **2.6** |
| Kimi-K3.1-Pro teacher | **88.1%** | **34** | 7.5 | - | - | - |

- **Teacher.** Kimi beats GLM-5.2 on test (88.1% vs 83.3% field accuracy, 34 vs 32 exact) but uses
  7.5 tool calls instead of 4.7 and costs ~$0.10 per paper instead of ~$0.03. It often breaks the
  research protocol by fetching the project page after finding the GitHub repository (protocol-complete
  on only 21/42 test papers). Two test episodes returned no valid structured output; their fields count
  as wrong.
- **SFT.** Distillation keeps only protocol-complete, fully correct trajectories, so the student follows
  the protocol far more often than its teacher (test 36/42, validation 44/50) with half the tool calls.
  SFT gains 10.7 points on test and 5 on validation over the base 9B, almost entirely on project pages
  (test 25 -> 33 of 42). The 9B SFT ties the 4B SFT v2 on test (85.7%) and is 3 points higher on validation.
- **GRPO** did not improve accuracy. Against its own merged SFT checkpoint it lost 6 points on test
  (88.1% -> 82.1%) and tied on validation (88.0%), while trimming tool calls (3.2 -> 3.0, 2.9 -> 2.6)
  and raising protocol-complete episodes (test 35 -> 37, validation 44 -> 48). Training reward stayed
  flat (0.78 over steps 0-20, 0.75 over steps 80-100), unlike the 4B run (0.59 -> ~0.95): the 9B SFT starts
  near the ceiling on these prompts, and often all 8 rollouts of a prompt got the same reward, leaving
  no learning signal.
- **Noise.** The same SFT weights scored 85.7% (adapter) and 88.1% (merged) on test, and 93% vs 88% on
  validation, in two runs of the same evaluation. Run-to-run variance on 42-50 papers is therefore at
  least 5 points, which covers every SFT-vs-GRPO and student-vs-teacher difference in this table. The
  supported conclusions are that SFT clearly helps the 9B, that it reaches the teacher's accuracy with
  about half the tool calls, and that GRPO's main measurable effect is slightly shorter, more
  protocol-complete episodes.

Artifacts (public; the account's private storage was full):

- Traces: [`nielsr/paper-url-opencode-traces-kimi`](https://huggingface.co/datasets/nielsr/paper-url-opencode-traces-kimi)
  (`train-kimi-k31.jsonl`, 815 accepted of 1,732 train papers; `teacher-test-kimi-k31.jsonl`) and the
  bucket `nielsr/url-agent-kimi` (`data/`, staged job code under `code/`).
- SFT adapter: [`nielsr/qwen35-9b-url-agent-lora-kimi`](https://huggingface.co/nielsr/qwen35-9b-url-agent-lora-kimi).
- GRPO adapter (on the merged SFT model): [`nielsr/qwen35-9b-url-agent-grpo-lora-kimi`](https://huggingface.co/nielsr/qwen35-9b-url-agent-grpo-lora-kimi);
  standalone merged GRPO model: [`nielsr/qwen35-9b-url-agent-grpo-merged-kimi`](https://huggingface.co/nielsr/qwen35-9b-url-agent-grpo-merged-kimi).
- Evaluation records and summaries: [`nielsr/qwen35-9b-url-agent-eval`](https://huggingface.co/datasets/nielsr/qwen35-9b-url-agent-eval)
  (`sft/`: base vs SFT adapter; `sft-grpo/`: merged SFT vs GRPO adapter).

Reproduce:

```bash
# Teacher baseline and distillation (OpenCode 1.2.4; HF_BILL_TO set). Three workers on disjoint slices.
uv run generate_opencode_traces.py --split test --limit 500 --attempts-per-row 1 --concurrency 4 \
  --model huggingface/moonshotai/Kimi-K3.1-Pro --output runs/kimi-test.jsonl
for i in 0 1 2; do uv run generate_opencode_traces.py --split train --start $((i*578)) --limit 578 \
  --accept-partial --attempts-per-row 1 --concurrency 4 --resume-attempted \
  --model huggingface/moonshotai/Kimi-K3.1-Pro --output runs/kimi-train-$i.jsonl & done; wait
scripts/kimi_9b_pipeline.sh sft                 # 1x H200: SFT + eval base vs SFT
ROLLOUTS=16 scripts/kimi_9b_pipeline.sh grpo    # 2x H200: merge, GRPO, merge, eval vs merged SFT
```

`scripts/rl-kimi-9b-ids.txt` is the GRPO prompt pool: the 1,134 train papers where Kimi got a URL
wrong, broke the research protocol, or that have a verified project page; the 598 GitHub-only papers it
solved cleanly are excluded.

Cost and compute: teacher API $187.66 for the train split plus $4.22 for the test baseline. SFT took 83
minutes (job 1 h 38 min with both evaluations) on one H200. GRPO took 54 minutes for 100 steps on 2x H200
(job 1 h). Merging and the GRPO evaluation took 13 minutes on one H200.

Notes from this run:

- With 32 concurrent rollouts the 9B GRPO job ran out of host resources at step 6 (OpenCode killed with
  -9, `posix_spawn` returning `EAGAIN`). With `--max-concurrent-rollouts 16` it ran cleanly at the same
  speed, because the trainer, not rollout generation, is the bottleneck.
- `merge_adapter.py` wrote the local path `/model` as `base_model` in the model card, which the Hub
  rejects. `--card-base-model` now names the Hub model. `scripts/kimi_9b_pipeline.sh grpo-finish` redoes
  the merge and evaluation from the pushed GRPO adapter without retraining.
- `generate_opencode_traces.py --resume-attempted` resumes without paying again for papers that were
  attempted but rejected. Trace records containing lone UTF-16 surrogates (math symbols in paper text)
  are now written in escaped form instead of crashing the worker.

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
