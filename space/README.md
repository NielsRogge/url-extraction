---
title: Small Agent vs Frontier Teacher
emoji: 🏎️
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Fine-tuned 4B OpenCode agent races its teacher on paper URLs
---

# Paper URL agent: fine-tuned 4B vs. frontier teacher

Enter an arXiv paper and two agents race to find its official GitHub repository and project page,
both inside the unmodified [OpenCode](https://opencode.ai) harness with the same tools, system prompt
and JSON output schema:

- **Student**: Qwen3.5-4B fine-tuned on the teacher's verified OpenCode trajectories (and optionally
  refined with GRPO through OpenEnv's capture proxy and TRL), served from a single-A10G Inference Endpoint.
- **Teacher**: GLM-5.2 through Hugging Face Inference Providers.

Each column streams that agent's reasoning and tool calls (click a call to see its output), with
elapsed time, tool-call count, tokens and cost. Final URLs are checked against verified labels from
`nielsr/paper-url-extraction-v1` when available, otherwise against current HF paper metadata.

## Configuration

| Name | Kind | Purpose |
|---|---|---|
| `HF_TOKEN` | secret | Calls the student endpoint and Inference Providers. Use a fine-grained token limited to those two permissions. |
| `STUDENT_ENDPOINT_URL` | variable | Base URL of the student's vLLM Inference Endpoint (no `/v1`). |
| `HF_BILL_TO` | variable | Optional organization to bill Inference Providers usage to. |
| `STUDENT_LABEL`, `TEACHER_MODEL` | variables | Optional display name / teacher model override. |

The agents execute shell commands and read untrusted web pages, so they never receive a credential:
the token is removed from the process environment at startup and attached only by a local proxy that
OpenCode reaches at `127.0.0.1`. Because the agents still share the container, keep the token's scope
narrow.
