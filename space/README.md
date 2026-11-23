---
title: Same Paper, Two Agents
emoji: 🏁
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
suggested_hardware: l4x1
pinned: false
short_description: GRPO-tuned 4B OpenCode agent races its frontier teacher
---

# Same paper, two agents

A Qwen3.5-4B agent, distilled from GLM-5.2's OpenCode trajectories (SFT v2) and refined with GRPO (v3)
through OpenEnv's capture proxy and TRL, races a frontier model, GLM-5.3, to find a paper's official
GitHub repository and project page. Both run in the unmodified [OpenCode](https://opencode.ai) harness with the same tools,
prompt and JSON answer schema. Each panel streams that agent's tool calls; the replay button replays
the last race.

- **Student**: served by vLLM on this Space's own GPU (`STUDENT_MODEL`, default
  `nielsr/qwen35-4b-url-agent-v3-grpo-merged`).
- **Teacher**: GLM-5.3 (`TEACHER_MODEL`, default `zai-org/GLM-5.3`) through Hugging Face Inference Providers.

## Configuration

| Name | Kind | Purpose |
|---|---|---|
| `HF_TOKEN` | secret | Downloads the private student model and calls Inference Providers for the teacher. Scope it to read that model and to make Inference Providers calls. |
| `HF_BILL_TO` | variable | Optional organization to bill Inference Providers usage to. |
| `STUDENT_MODEL` | variable | Student checkpoint to serve. |
| `STUDENT_LABEL` | variable | Student name shown in the race panel. |
| `TEACHER_MODEL` | variable | Teacher model on Inference Providers. |

The agents execute shell commands and read untrusted web pages in this container, so the token is
moved into a one-time file by `start.sh` and removed from every process environment; vLLM and the
agents never receive it, and model calls for the teacher go through an in-process proxy that adds it.
