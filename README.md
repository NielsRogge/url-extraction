# Distilling a frontier agent into a small open model: paper URL extraction

Given an arXiv paper, an agent finds its official GitHub repository and project page. A frontier
open-weight **teacher** does this well but is expensive per call. This repository replaces it with a
small fine-tuned **student** that runs in the same agent harness:

1. **Traces.** The teacher (served through Hugging Face Inference Providers) solves training papers inside
   [OpenCode](https://opencode.ai). Only trajectories whose URLs match verified labels and that follow the
   research protocol are kept.
2. **SFT.** The student is fine-tuned on those trajectories (LoRA, loss on assistant turns only).
3. **GRPO.** The student keeps training in the real OpenCode harness with TRL's
   `HarnessAsyncGRPOTrainer` and OpenEnv's capture proxy, rewarded by the URL verifier.
4. **Evaluation.** Base, SFT, GRPO and teacher are compared in the same harness on held-out papers.

Teachers and students so far:

| Teacher | Student | Write-up |
|---|---|---|
| `zai-org/GLM-5.2` (`GLM-5.3` is now the default) | `Qwen/Qwen3.5-4B` | [docs/EXPERIMENT_glm52_qwen35-4b.md](docs/EXPERIMENT_glm52_qwen35-4b.md) |
| `moonshotai/Kimi-K3.1-Pro` | `Qwen/Qwen3.5-9B` | [docs/EXPERIMENT_kimi-k31_qwen35-9b.md](docs/EXPERIMENT_kimi-k31_qwen35-9b.md) |
| `thinkingmachines/Inkling-2-Small` | `thinkingmachines/Inkling-2-Mini` | [docs/EXPERIMENT_inkling-2-small_inkling-2-mini.md](docs/EXPERIMENT_inkling-2-small_inkling-2-mini.md) |

## Results

Field accuracy (fraction of verified URL fields correct) on the fully scoreable held-out papers, in
OpenCode with a 180 s budget and at most 15 tool calls. Test has 42 papers (84 fields), validation 50
(100 fields).

| Model | Test | Validation | Test tool calls |
|---|---:|---:|---:|
| Qwen3.5-4B base | 78.6% | 71.0% | 8.4 |
| Qwen3.5-4B SFT (873 GLM-5.2 traces) | 85.7% | 90.0% | 4.7 |
| Qwen3.5-4B SFT + GRPO | 84.5% | 93.0% | 3.7 |
| GLM-5.2 teacher | 83.3% | - | 4.7 |
| Qwen3.5-9B base | 75.0% | 88.0% | 6.8 |
| Qwen3.5-9B SFT (815 Kimi traces) | 85.7% | 93.0% | 3.8 |
| Qwen3.5-9B SFT + GRPO | 82.1% | 88.0% | 3.0 |
| Kimi-K3.1-Pro teacher | 88.1% | - | 7.5 |
| Inkling-2-Mini SFT + GRPO (Inkling-2-Small traces; teacher 85.7%) | 85.7% | 92.0% | 4.1 |

GRPO v3 on the harder [`paper-url-extraction-v2`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v2)
(with a `test_hard` split): +4.9 points on test_hard for the 4B with 35-41% fewer tool calls, but -6.7 for the
9B; details in the two experiment write-ups.

Takeaways:

- **SFT is what closes the gap.** Both students reach their teacher's accuracy with about half the tool
  calls of the Kimi teacher and the same as GLM, while the teachers cost about $0.03 (GLM-5.2) and $0.10
  (Kimi) per paper through Inference Providers.
- **GRPO mainly makes episodes shorter.** Its accuracy changes are within run-to-run noise, which is at
  least 5 points on splits this small.
- **The benchmark is mostly easy.** Most papers are solved by one HF API call. Harder papers, whose URLs
  are only in the PDF or on the project page, are being mined to make evaluation and the RL signal more
  informative ([docs/data.md](docs/data.md)).

All models, datasets, traces, evaluations and the demo Space are grouped in the
[Hub collection](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e).

## Try it

[`nielsr/url-agent-race`](https://huggingface.co/spaces/nielsr/url-agent-race) runs the GRPO-trained
Qwen3.5-4B (on the Space's own GPU) against the GLM-5.3 teacher on any arXiv paper, streaming both
agents' reasoning and tool calls side by side. Source in [`space/`](space/).

## Run the recipe

Teachers and students are pluggable: each model is a small profile in [`profiles/`](profiles/) (provider
pin, vLLM image and parsers, GPU flavors, LoRA targets, GRPO backend, ...), and one entry point runs
traces -> SFT -> merge -> (difficulty probe) -> GRPO -> eval -> Hub publishing as HF Jobs:

```bash
uv run pipeline.py --teacher kimi-k3.1-pro --student qwen3.5-9b --dataset nielsr/paper-url-extraction-v1
uv run pipeline.py --teacher glm-5.2 --student qwen3.5-4b --stages eval --splits test,test_hard --dry-run
```

Requirements: an `HF_TOKEN` that can call Inference Providers (billed to `huggingface`), run Jobs in the
`huggingface` namespace and write the target repos. Every stage also runs on its own (`--stages`), and
`--dry-run` prints each job. Profiles, stages, artifact names and how to add a model:
[docs/pipeline.md](docs/pipeline.md).

All stages share one harness package (`harness/`: prompt, answer schema, OpenCode episode runner and the
single scorer), so a teacher trace is accepted, an evaluation episode is scored and a GRPO rollout is
rewarded by the same code (`tests/test_parity.py`).

## Repository map

| Path | Purpose |
|---|---|
| `pipeline.py`, `profiles/` | Entry point and teacher/student profiles ([docs/pipeline.md](docs/pipeline.md)) |
| `harness/` | Task, scorer, OpenCode runner, profile loader, rescoring of published records |
| `generate_opencode_traces.py` | Teacher trajectories in OpenCode, verified against labels |
| `train_sft.py` | LoRA SFT on accepted trajectories |
| `train_grpo_harness.py` | GRPO in OpenCode via TRL `HarnessAsyncGRPOTrainer` + OpenEnv capture |
| `evaluate_vllm.py` | Paired base vs. adapter evaluation in OpenCode, served by vLLM |
| `merge_adapter.py` | Merge a LoRA into a standalone checkpoint |
| `trace_export.py` | Episodes as Pi sessions for the Hub agent-traces viewer ([docs/data.md](docs/data.md)) |
| `validate_labels.py`, `tier_labels.py`, `difficulty_probe.py`, `mine_papers.py` | Label validation, difficulty tiers, RL prompt selection, mining harder papers ([docs/data.md](docs/data.md)) |
| `space/`, `scripts/serve_space.py` | Side-by-side demo Space and its deployment |
| `url_extraction_agent/` | Production batch runner |
| `deploy_endpoint.py`, `run_endpoint_agent.py`, `prepare_canary.py`, `finalize_evaluation.py`, `convert_mlx.py` | Serving and production runs ([docs/deployment.md](docs/deployment.md)) |
| `train_opencode.py` | Legacy one-step GRPO smoke test ([docs/legacy-train-opencode.md](docs/legacy-train-opencode.md)) |

## Further reading

- [docs/pipeline.md](docs/pipeline.md): harness, profiles, pipeline stages, and the rescoring of every
  published evaluation with the unified scorer.
- [docs/EXPERIMENT_glm52_qwen35-4b.md](docs/EXPERIMENT_glm52_qwen35-4b.md): the full 4B history (SFT v1/v2,
  GRPO v1/v2, design notes on the OpenEnv capture proxy and the reward).
- [docs/EXPERIMENT_kimi-k31_qwen35-9b.md](docs/EXPERIMENT_kimi-k31_qwen35-9b.md): the 9B run, its cost and compute.
- [docs/data.md](docs/data.md): label semantics, validation, difficulty tiers, mining.
- [docs/deployment.md](docs/deployment.md): Inference Endpoint, production batches on HF Jobs, MLX.
