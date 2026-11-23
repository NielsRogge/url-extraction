---
title: Same Paper, Two Agents
emoji: 🏁
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
pinned: false
short_description: GRPO-tuned 4B OpenCode agent races its frontier teacher
---

# Same paper, two agents

A Qwen3.5-4B agent, distilled from GLM-5.2's OpenCode trajectories and refined with GRPO through
OpenEnv's capture proxy and TRL, races its teacher to find a paper's official GitHub repository and
project page. Both run in the unmodified [OpenCode](https://opencode.ai) harness with the same tools,
prompt and JSON answer schema. Each panel streams that agent's tool calls; the replay button replays
the last race.

- **Student**: `nielsr/qwen35-4b-url-agent-v3-grpo-merged`, served by vLLM on the managed Inference
  Endpoint `huggingface/qwen35-4b-url-agent-vllm` (one L4, scales to zero after 15 idle minutes;
  `STUDENT_URL`). The Space itself runs on `cpu-basic`. The first race after the endpoint has scaled
  to zero waits for a cold start of a few minutes. Without `STUDENT_URL`, `launch.py` serves
  `STUDENT_MODEL` with vLLM on the Space's own GPU instead (`scripts/serve_space.py --local-gpu`).
- **Teacher**: GLM-5.3 (`TEACHER_MODEL`) through Hugging Face Inference Providers.

## Configuration

| Name | Kind | Purpose |
|---|---|---|
| `HF_TOKEN` | secret | Calls the student endpoint and Inference Providers for the teacher (and, in local-GPU mode, downloads the student). Scope it to call Inference Endpoints in the `huggingface` org (`inference.endpoints.infer.write`) and to make Inference Providers calls. |
| `HF_BILL_TO` | variable | Optional organization to bill Inference Providers usage to. |
| `STUDENT_URL` | variable | Base URL of the Inference Endpoint serving the student (no `/v1`). Unset to serve locally. |
| `STUDENT_MODEL` | variable | Student checkpoint (served locally when `STUDENT_URL` is unset). |
| `STUDENT_LABEL` | variable | Student name shown in the UI. |
| `STUDENT_GPU_PER_HOUR` | variable | Student GPU price in USD/h for the cost readout (L4: 0.80). |
| `TEACHER_MODEL` | variable | Teacher model on Inference Providers. |

The agents execute shell commands and read untrusted web pages in this container, so the token is
moved into a one-time file by `start.sh` and removed from every process environment; vLLM and the
agents never receive it, and model calls for the student and the teacher go through an in-process proxy that adds it.
