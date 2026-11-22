# Experiment: Kimi-K3.1-Pro teacher -> Qwen3.5-9B student

The same teacher -> SFT -> GRPO pipeline, with `moonshotai/Kimi-K3.1-Pro` (Inference Providers,
served by Together; OpenCode model `huggingface/moonshotai/Kimi-K3.1-Pro`) as teacher and
`Qwen/Qwen3.5-9B` as student. Harness and metrics are identical to the [GLM-5.2 -> Qwen3.5-4B experiment](EXPERIMENT_glm52_qwen35-4b.md) (OpenCode 1.2.4,
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
# SFT (1x H200), GRPO (2x H200), evaluation: see pipeline.md for the full command with this run's repo names
uv run pipeline.py --teacher kimi-k3.1-pro --student qwen3.5-9b --grpo-ids scripts/rl-kimi-9b-ids.txt \
  --traces-repo nielsr/paper-url-opencode-traces-kimi --traces-files train-kimi-k31.jsonl --stages sft,merge,grpo,eval
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
  rejects. `--card-base-model` now names the Hub model. `pipeline.py --stages grpo,eval --grpo-merge-only` redoes
  the merge and evaluation from the pushed GRPO adapter without retraining.
- `generate_opencode_traces.py --resume-attempted` resumes without paying again for papers that were
  attempted but rejected. Trace records containing lone UTF-16 surrogates (math symbols in paper text)
  are now written in escaped form instead of crashing the worker.

## GRPO v3 on paper-url-extraction-v2

Same recipe as the 4B GRPO v3 run, from the published SFT adapter (merged first, since no merged 9B SFT
checkpoint existed):

```bash
uv run pipeline.py --orchestrate --teacher kimi-k3.1-pro --student qwen3.5-9b --dataset nielsr/paper-url-extraction-v2 \
  --run-name qwen35-9b-url-agent-v3 --sft-adapter nielsr/qwen35-9b-url-agent-lora-kimi \
  --stages merge,probe,grpo,eval --probe-exclude-tiers 7 --probe-shards 8 --probe-args '--sample 0' \
  --grpo-args '--learning-rate 2e-5 --max-steps 150 --exclude-tiers 7 --tier-weights 0:2,2:2,3:2,4:2,5:2,6:2' \
  --splits test,validation,test_hard --eval-include-partial --eval-limit 1000 --eval-variants sft,grpo,teacher
```

- **Probe**: all 2,696 scoreable non-tier-7 v2 train papers, 4 attempts each (8 jobs, ~75-95 min each); mean
  success 61%; 1,292 papers qualified (tiers 0-6: 94, 708, 157, 192, 16, 79, 46).
- **Training reward** per 25 steps: 0.75, 0.89, 0.74, 0.75, 0.89, 0.81 (no upward trend).
- **Evaluation** (field accuracy all / fully scoreable, tool calls; fully scoreable: test 33, validation 44,
  test_hard 94 papers):

| Model | Test | Validation | test_hard | Tool calls (test / val / hard) |
|---|---|---|---|---|
| SFT + GRPO v3 | 85.8% / 81.8% | 90.3% / 93.2% | 66.5% / 67.6% | 3.9 / 3.9 / 5.6 |
| SFT (merged, paired control) | 87.0% / 93.9% | 91.4% / 95.5% | 73.2% / 75.5% | 4.0 / 3.9 / 5.2 |
| SFT (adapter) | 87.7% / 92.4% | 89.9% / 95.5% | 71.7% / 75.0% | 4.2 / 3.7 / 5.5 |
| Qwen3.5-9B base | 88.9% / 86.4% | 87.5% / 86.4% | 71.0% / 75.5% | 5.2 / 5.1 / 8.7 |
| Kimi-K3.1-Pro teacher | 92.5% / 92.4% | 94.2% / 96.6% | 80.3% / 81.4% | 7.2 / 6.2 / 11.4 |

GRPO v3 does not help the 9B: against its paired control it loses 1.2 points on test, 1.1 on validation and
6.7 on test_hard (exact 98 vs 114 of 175), with unchanged tool calls. The 9B base is already strong on v2;
SFT mainly cuts its tool calls. Likely causes to test: the pool is 55% tier 1, which the policy mostly
solves; lr 2e-5 may be too high for the 9B; the step-50 and step-100 checkpoints in the adapter repo's history
would show whether the drop grows with training. Cost about $175.

Artifacts: [`nielsr/qwen35-9b-url-agent-v3-sft-merged`](https://huggingface.co/nielsr/qwen35-9b-url-agent-v3-sft-merged),
[`-v3-grpo-lora`](https://huggingface.co/nielsr/qwen35-9b-url-agent-v3-grpo-lora),
[`-v3-grpo-merged`](https://huggingface.co/nielsr/qwen35-9b-url-agent-v3-grpo-merged),
[`-v3-eval`](https://huggingface.co/datasets/nielsr/qwen35-9b-url-agent-v3-eval) (episodes, sessions, probe, run
manifest).
