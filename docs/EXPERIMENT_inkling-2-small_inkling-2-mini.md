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
| Inkling-2-Mini SFT + GRPO | _pending_ | | | | | |
| Inkling-2-Small teacher (DeepInfra) | **85.7%** | **34** | 7.1 | - | - | - |
| Inkling-2-Small teacher (default routing) | 60.7% | 22 | 7.7 | - | - | - |

With the validated labels (`nielsr/paper-url-extraction-v1-validated`: redirect aliases accepted, contradicted
labels dropped; 75 test and 94 validation fields), base scores 74.7% / 89.4%, SFT 85.3% / 90.4% (exact 35/42,
45/50) and the teacher 89.3% (38/42). `scripts/score_records.py --rescore DATASET` recomputes any record file
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

_Pending: difficulty probe of the merged SFT policy (4 attempts per paper under the rollout conditions, on
the validated labels), then GRPO on the papers it solves sometimes, at LR 2e-5._

## Artifacts

All public and in the [Hub collection](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e):

- Traces: [`nielsr/inkling-2-mini-url-agent-traces`](https://huggingface.co/datasets/nielsr/inkling-2-mini-url-agent-traces)
  (`train-inkling.jsonl`: 1,215 accepted trajectories; `raw/`: every distillation attempt and the teacher test
  baselines per provider; `sessions/`: agent-trace viewer sessions).
- SFT adapter: [`nielsr/inkling-2-mini-url-agent-lora`](https://huggingface.co/nielsr/inkling-2-mini-url-agent-lora);
  merged SFT model: [`nielsr/inkling-2-mini-url-agent-sft-merged`](https://huggingface.co/nielsr/inkling-2-mini-url-agent-sft-merged).
- Evaluation records: [`nielsr/inkling-2-mini-url-agent-eval`](https://huggingface.co/datasets/nielsr/inkling-2-mini-url-agent-eval)
  (`base/`, `sft/`, `difficulty/`).

## Reproduce

```bash
# Teacher baseline and distillation (OpenCode 1.2.4, HF_BILL_TO=huggingface). The `:deepinfra` model id needs an
# inline OpenCode model entry (see OPENCODE_CONFIG_CONTENT in the commit history of this experiment).
uv run generate_opencode_traces.py --split test --limit 500 --attempts-per-row 1 --concurrency 4 \
  --model huggingface/thinkingmachines/Inkling-2-Small:deepinfra --output runs/teacher-test.jsonl
for i in 0 1 2; do uv run generate_opencode_traces.py --split train --accept-partial --attempts-per-row 1 \
  --start $((i*578)) --limit 578 --concurrency 6 --model huggingface/thinkingmachines/Inkling-2-Small:deepinfra \
  --output runs/distill-w$i.jsonl & done; wait
python scripts/merge_traces.py runs/train-inkling.jsonl runs/distill-w*.jsonl

REV=$(scripts/inkling_job.sh stage)                       # code snapshot in a private Hub repo
scripts/inkling_job.sh sft $REV                            # 1x H200, ~92 min
scripts/inkling_job.sh eval $REV test base/ --base-model thinkingmachines/Inkling-2-Mini --adapter '' --no-enforce-eager
scripts/inkling_job.sh eval $REV test sft/ --base-model thinkingmachines/Inkling-2-Mini \
  --adapter nielsr/inkling-2-mini-url-agent-lora --adapter-only --max-lora-rank 64 --no-enforce-eager
scripts/inkling_job.sh merge $REV nielsr/inkling-2-mini-url-agent-lora nielsr/inkling-2-mini-url-agent-sft-merged
for i in 0 1 2 3; do scripts/inkling_job.sh probe $REV $i/4; done   # difficulty probe, 4x RTX PRO 6000
```
