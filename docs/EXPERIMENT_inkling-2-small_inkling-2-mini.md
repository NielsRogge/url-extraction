# Experiment: Inkling-2-Small teacher -> Inkling-2-Mini student

The same teacher -> SFT -> GRPO pipeline with `thinkingmachines/Inkling-2-Small` (594B, Inference Providers;
OpenCode model `huggingface/thinkingmachines/Inkling-2-Small:deepinfra`) as teacher and
`thinkingmachines/Inkling-2-Mini` (26.4B MoE, 3.4B active, a looped transformer: `inkling2_loop`) as student.
Harness and metrics are identical to the [GLM-5.2 -> Qwen3.5-4B experiment](EXPERIMENT_glm52_qwen35-4b.md)
(OpenCode 1.2.4, 180 s episodes, 15 tool calls, 64k context, fully scoreable held-out papers). Unlike the
Qwen students, Inkling-2-Mini reasons natively, and this time the teacher's reasoning is distilled too.

| Model | Test field acc. (84 fields) | Test exact (42) | Test tool calls | Val field acc. (100 fields) | Val exact (50) | Val tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Inkling-2-Mini base | 71.4% | 27 | 4.8 | 87.0% | 40 | 3.3 |
| **Inkling-2-Mini SFT** (1,215 traces, adapter) | **82.1%** | **31** | 4.3 | 87.0% | 41 | 4.4 |
| Inkling-2-Mini SFT, merged (paired control for the row below) | 82.1% | 31 | 4.4 | 88.0% | 41 | 4.2 |
| **Inkling-2-Mini SFT + GRPO** (90 steps) | **85.7%** | 32 | 4.1 | **92.0%** | **43** | **3.0** |
| Inkling-2-Small teacher (DeepInfra) | **85.7%** | **34** | 7.1 | - | - | - |
| Inkling-2-Small teacher (default routing) | 60.7% | 22 | 7.7 | - | - | - |

With the validated labels (`nielsr/paper-url-extraction-v1-validated`: redirect aliases accepted, contradicted
labels dropped; 75 test and 94 validation fields), base scores 74.7% / 89.4%, SFT 85.3% / 90.4% (exact 35/42,
45/50), merged SFT 86.7% / 90.4%, SFT + GRPO 90.7% / 94.7% (exact 37/42, 46/50) and the teacher 89.3% (38/42). `scripts/score_records.py --rescore DATASET` recomputes any record file
against a label set.

- **Teacher provider matters.** Routed through Together or OpenRouter, Inkling-2-Small often answers with an
  empty `StructuredOutput({})` call (16 and 7 of 42 test papers), which scores as no answer. DeepInfra never
  does, so distillation pins `:deepinfra`. Pinned to DeepInfra the teacher matches GLM-5.2 (85.7% vs 83.3%,
  34 vs 32 exact) with more tool calls (7.1 vs 4.7).
- **Distillation keeps the reasoning.** `generate_opencode_traces.py` now carries OpenCode `reasoning` parts
  into `reasoning_content`, which the Inkling 2 chat template renders as `<|content_thinking|>` blocks (the
  previous Qwen students never saw or produced reasoning). One attempt per train paper gave 1,215 accepted
  trajectories (GLM-5.2: 873; Kimi: 815) from 1,726 attempted papers, all with reasoning.
- **SFT** gains 10.7 points on test over the base model and halves test timeouts (8 -> 4); validation was
  already high and barely moves. The student keeps reasoning in almost every call, about 25% shorter than
  the base model's. With 42-50 papers per split, differences of a few points are within noise.

### Harder held-out papers (`test_hard`)

`nielsr/paper-url-extraction-v2` adds a `test_hard` split of 175 papers (94 fully scoreable; the others have at
least one requested field without a verifiable label and are scored on the remaining fields, `--include-partial`).

| Model | Field acc. (269 fields, 175 papers) | Exact (175) | Field acc. (188 fields, 94 fully scoreable) | Exact (94) | Tool calls | Timeouts |
|---|---:|---:|---:|---:|---:|---:|
| Inkling-2-Mini base | 76.2% | 123 | **83.0%** | **74** | 7.9 | 33 |
| Inkling-2-Mini SFT (adapter) | 72.1% | 115 | 77.1% | 66 | 7.9 | 38 |
| Inkling-2-Mini SFT, merged (paired control) | 70.6% | 113 | 76.1% | 66 | 8.5 | 42 |
| **Inkling-2-Mini SFT + GRPO** (90 steps) | **84.8%** | **141** | **86.2%** | **75** | 7.7 | **8** |
| Inkling-2-Small teacher (DeepInfra) | 80.7% | 132 | 84.0% | 73 | 11.3 | - |

On these harder papers SFT alone does not help: it follows the research protocol more often (protocol-complete
episodes 97 -> 109, project-page-after-GitHub violations 29 -> 15) but times out more (33 -> 38) and loses 4-6
points of field accuracy. The paired difference is not significant (base-only exact 20 papers, SFT-only 12;
two-sided binomial p = 0.22). The teacher needs 11.3 tool calls per paper here, so many `test_hard` papers need
more than the student's 180 s / 15-tool budget; base Inkling-2-Mini comes within one point of the teacher on
the fully scoreable subset. GRPO fixes exactly this (next section): the 3.4B-active student then beats its
594B teacher on `test_hard` (84.8% vs 80.7%) with fewer tool calls (7.7 vs 11.3).

## What is different about Inkling-2-Mini

- **Serving.** No vLLM release supports `inkling2_loop`; the only build is the `vllm/vllm-openai:inkling-2-mini`
  image (flags `--tool-call-parser inkling --reasoning-parser inkling`). It ships Blackwell kernels only
  (`TORCH_CUDA_ARCH_LIST=10.0+PTX 10.3+PTX` plus an SM12x path): on H200 it fails with
  `no kernel image is available` (`csrc/inkling2/conv_mix.cu`), and the bundled DSpark drafter fails the same
  way on RTX PRO 6000. Serving and GRPO therefore run on RTX PRO 6000, where prompt processing is slow
  (~1-2k tokens/s): an episode takes ~70-100 s at 8 concurrent sessions.
- **Checkpoint.** The release is ~20 GB with routed experts in NVFP4/MXFP8 and everything else in BF16.
  transformers 5.24 dequantizes the experts to BF16 when loading (~53 GB), which SFT trains on.
- **Training attention.** transformers implements Inkling attention eagerly only (per-token sinks and a
  hidden-state-dependent relative position bias), materializing `[heads, L, L]` scores. `inkling_train_patch.py`
  computes the same math in query chunks under activation checkpointing and restricts sliding-window layers
  to their window, so 32k-token trajectories train on one H200 (exact to ~1e-6 against eager on a tiny model).
- **LoRA.** vLLM's Inkling2 LoRA can add expert deltas only to BF16 experts, but its loader accepts only the
  checkpoint's quantized experts, so adapters are attention-only (q/k/v/o and the output gate, rank 64; the
  looped layers reuse them on each pass). `inkling_adapter.py` renames PEFT modules to the checkpoint names vLLM
  expects, and `merge_adapter.py` merges an adapter straight into the checkpoint's BF16 attention tensors,
  leaving the quantized experts untouched, so the merged model serves exactly like the original.
- **GRPO.** TRL's `AsyncGRPOTrainer` packs rollouts into a prefix tree for FlashAttention-3 and rejects models
  with attention sinks or short convolutions, which rules out Inkling. `train_grpo_harness.py --trainer simple`
  keeps the OpenCode rollouts through the OpenEnv capture proxy, the reward and vLLM LoRA hot-loading, and
  replaces only the optimizer loop (group-normalized advantages, clipped importance-weighted policy gradient
  against the engine's logprobs, token-level normalization). vLLM runs NVFP4 experts while the trainer uses the
  dequantized BF16 ones; the importance ratio still averages ~1.00 with ~5% of tokens clipped.

## GRPO, following the harness-adapters recipe

1. **Validated tasks.** GRPO and the probe use `nielsr/paper-url-extraction-v1-validated` (`validate_labels.py`,
   the `openenv validate` analogue: labels re-checked against current HF metadata, GitHub redirects as aliases,
   oracle / floor / canary checks of the reward).
2. **Difficulty probe** (`difficulty_probe.py`, 4x RTX PRO 6000 shards). The merged SFT policy attempted 300
   papers sampled from the 788-paper candidate pool (teacher failures and project-page papers) 4 times each
   under the rollout conditions (temperature 1.0, 4,096 output tokens, 24 model calls, 300 s). Successes per
   paper: 0 -> 36, 1 -> 11, 2 -> 29, 3 -> 58, 4 -> 166. The 98 papers solved sometimes but not always form the
   prompt pool (`scripts/rl-inkling-qualified-ids.txt`); tier 7 (almost never solved) is excluded and tiers are
   reweighted (`--tier-weights 0:3,2:2,3:3,4:3,5:3,6:3`).
3. **Training.** `train_grpo_harness.py --trainer simple` from the merged SFT model on 2x RTX PRO 6000:
   LR 2e-5, LoRA r64 (attention only), 2 prompts x 4 rollouts per step, `--protocol-weight 0.5`, 32k-token
   training cap, checkpoints pushed every 10 steps. The 10 h job limit stopped it at step 95 of 100;
   checkpoint 90 is the final adapter. Training reward stayed flat (0.91 over the first and the last 20 steps;
   every group had reward variance), but episodes got much shorter: tool calls 6.8 -> 5.4 and wall time per
   rollout 101 s -> 47 s. The importance ratio between the vLLM sampler (NVFP4 experts) and the trainer (BF16)
   averaged 1.00 throughout with ~5% of tokens clipped.

**Results** (GRPO adapter on the merged SFT model vs the merged SFT model, same job, same papers):

| Split | SFT exact | GRPO exact | GRPO-only / SFT-only solved | Timeouts SFT -> GRPO | Seconds per episode |
|---|---:|---:|---:|---:|---:|
| test (42) | 31 | 32 | 3 / 2 | 4 -> 1 | 94 -> 58 |
| validation (50) | 41 | 43 | 2 / 0 | 3 -> 1 | 82 -> 41 |
| test_hard (175) | 113 | **141** | **32 / 4** (p = 2e-6) | **42 -> 8** | 102 -> 57 |

- **GRPO made the agent fast, and that is what pays off.** The policy learned to finish in about half the
  time with fewer tool calls, which removes most 180 s evaluation timeouts. On `test_hard`, 24 of the 32 papers
  only GRPO solves are papers where the SFT model timed out; on the 130 papers where neither timed out, GRPO
  still solves slightly more (117 vs 111). The trade is more episodes ending at the 15-tool-call limit (11 -> 21)
  and slightly more protocol violations (17 -> 21) on the hardest papers.
- **This is the first GRPO run in this project that clearly improves accuracy.** The test and validation gains
  (+3.6 and +4.0 points) are within noise on 42-50 papers, but the `test_hard` gain (+14.2 points field
  accuracy, 113 -> 141 exact, McNemar p = 2e-6) is not. None of the `test_hard` papers are in the train split
  or the GRPO pool. The Kimi -> Qwen3.5-9B run, which used the old pool without a difficulty probe and with
  the default LR, saw no gain.
- **Cost.** The GRPO job ran 10 h on 2x RTX PRO 6000 ($5.50/h, ~$55); two earlier attempts (test runs and a
  run that crashed at step 10 on an invalid model card) cost about as much again. Steps took ~6 min
  (rollouts ~2.5 min, the rest the synchronous update), several times slower than the Qwen runs on H200.

## Artifacts

All public and in the [Hub collection](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e):

- Traces: [`nielsr/inkling-2-mini-url-agent-traces`](https://huggingface.co/datasets/nielsr/inkling-2-mini-url-agent-traces)
  (`train-inkling.jsonl`: 1,215 accepted trajectories; `raw/`: every distillation attempt and the teacher test
  baselines per provider; `sessions/`: agent-trace viewer sessions).
- SFT adapter: [`nielsr/inkling-2-mini-url-agent-lora`](https://huggingface.co/nielsr/inkling-2-mini-url-agent-lora);
  merged SFT model: [`nielsr/inkling-2-mini-url-agent-sft-merged`](https://huggingface.co/nielsr/inkling-2-mini-url-agent-sft-merged).
- GRPO adapter (on the merged SFT model, checkpoints every 10 steps): [`nielsr/inkling-2-mini-url-agent-grpo-lora`](https://huggingface.co/nielsr/inkling-2-mini-url-agent-grpo-lora);
  merged GRPO model: [`nielsr/inkling-2-mini-url-agent-grpo-merged`](https://huggingface.co/nielsr/inkling-2-mini-url-agent-grpo-merged).
- Evaluation records: [`nielsr/inkling-2-mini-url-agent-eval`](https://huggingface.co/datasets/nielsr/inkling-2-mini-url-agent-eval)
  (`base/`, `sft/`: base vs SFT adapter; `sft-grpo/`: merged SFT vs GRPO adapter; `difficulty/`: probe
  attempts; `test_hard` files for every model; `sessions/`: agent-trace viewer sessions).

## Reproduce

```bash
# Teacher baseline and distillation (OpenCode 1.2.4, HF_BILL_TO=huggingface). The `:deepinfra` model id needs an
# inline OpenCode model entry: profiles/teachers/inkling-2-small.toml has it (pipeline.py passes it as
# --opencode-model-entry).
uv run generate_opencode_traces.py --split test --limit 500 --attempts-per-row 1 --concurrency 4 \
  --model huggingface/thinkingmachines/Inkling-2-Small:deepinfra --output runs/teacher-test.jsonl
for i in 0 1 2; do uv run generate_opencode_traces.py --split train --accept-partial --attempts-per-row 1 \
  --start $((i*578)) --limit 578 --concurrency 6 --model huggingface/thinkingmachines/Inkling-2-Small:deepinfra \
  --output runs/distill-w$i.jsonl & done; wait
python scripts/merge_traces.py runs/train-inkling.jsonl runs/distill-w*.jsonl

# SFT, merge, difficulty probe, GRPO and evaluation. The run above used the former scripts/inkling_job.sh; the
# inkling-2-mini profile carries the same settings (image, flavors, parsers, LoRA, hooks, GRPO arguments), and the
# default repo names under this run name are the ones listed in Artifacts.
uv run pipeline.py --teacher inkling-2-small --student inkling-2-mini --run-name inkling-2-mini-url-agent \
  --dataset nielsr/paper-url-extraction-v1-validated \
  --traces-repo nielsr/inkling-2-mini-url-agent-traces --traces-files train-inkling.jsonl \
  --grpo-ids scripts/rl-inkling-qualified-ids.txt --stages sft,merge,grpo,eval
# Select the GRPO pool with the probe instead: --grpo-ids scripts/rl-inkling-ids.txt --stages sft,merge,probe,grpo,eval
# If GRPO stops early, merge the last pushed checkpoint: --stages grpo --grpo-merge-only
# test_hard: --dataset nielsr/paper-url-extraction-v2 --splits test,validation,test_hard --eval-include-partial \
#   --eval-limit 1000 --stages eval; `python -m harness.report table` (or scripts/score_records.py --fully-scoreable)
#   adds the fully scoreable subset.
```
