# Experiment: GLM-5.3 teacher -> Qwen3.5-4B student in OpenCode v2 (2.3.2)

The GLM -> Qwen3.5-4B recipe (traces -> SFT -> merge -> difficulty probe -> GRPO -> evaluation) rerun
end to end in **OpenCode 2.3.2** instead of 1.2.4, with GLM-5.3 as the teacher. Every earlier published
result ran in OpenCode 1.2.4. The reference is GRPO v3 of
[EXPERIMENT_glm52_qwen35-4b.md](EXPERIMENT_glm52_qwen35-4b.md#grpo-v3-dataset-v2-difficulty-probed-pool)
(GLM-5.2 traces from OpenCode 1.2.4 -> SFT v2 -> GRPO v3).

## Results

Evaluation in OpenCode 2.3.2 with the standard recipe (180 s per episode, at most 15 tool calls, 8
concurrent episodes, each on its own OpenCode server). Field accuracy over every paper with at least one
verified field (`--eval-include-partial`), and over the fully scoreable subset. Each student job also ran
its control in the same job: the SFT job runs base Qwen3.5-4B, the GRPO job the merged SFT checkpoint.

Headline (field accuracy, all papers), against the OpenCode 1.2.4 reference (GLM-5.2 traces -> SFT v2 ->
GRPO v3):

| Model | Test | Validation | Test_hard | Tool calls (test_hard) |
|---|---:|---:|---:|---:|
| **Qwen3.5-4B SFT + GRPO, OpenCode 2.3.2** | **87.0%** | **87.9%** | **71.7%** | 4.3 |
| Qwen3.5-4B merged SFT, OpenCode 2.3.2 | 60.5% | 52.5% | 41.6% | 8.7 |
| Qwen3.5-4B base, OpenCode 2.3.2 | 59.7% | 59.5% | 46.8% | 10.1 |
| GLM-5.3 teacher, OpenCode 2.3.2 | 88.9% | 93.0% | 69.9% | 8.6 |
| Qwen3.5-4B GRPO v3, OpenCode 1.2.4 | 89.7% | 93.4% | 78.1% | 4.4 |
| Qwen3.5-4B merged SFT v2, OpenCode 1.2.4 | 90.1% | 91.8% | 73.2% | 6.5 |
| GLM-5.2 teacher, OpenCode 1.2.4 | 90.1% | 91.4% | 81.8% | 7.3 |

Full table (`python -m harness.report table`; "SFT (LoRA)" is the unmerged adapter served by vLLM, run
in the same job as base):

| Model | Split | Field acc. (all) | Exact (all) | Field acc. (fully scoreable) | Exact (fully scoreable) | Tool calls |
|---|---|---:|---:|---:|---:|---:|
| GRPO | test | 87.0% (253) | 190/220 | 83.3% (66) | 25/33 | 3.1 |
| Merged SFT | test | 60.5% (253) | 135/220 | 54.5% (66) | 18/33 | 7.1 |
| GRPO | test_hard | 71.7% (269) | 106/175 | 77.7% (188) | 59/94 | 4.3 |
| Merged SFT | test_hard | 41.6% (269) | 66/175 | 45.2% (188) | 39/94 | 8.7 |
| GRPO | validation | 87.9% (257) | 184/213 | 89.8% (88) | 37/44 | 2.8 |
| Merged SFT | validation | 52.5% (257) | 107/213 | 59.1% (88) | 24/44 | 6.7 |
| SFT (LoRA) | test | 52.2% (253) | 111/220 | 60.6% (66) | 19/33 | 7.2 |
| Base | test | 59.7% (253) | 129/220 | 65.2% (66) | 21/33 | 9.1 |
| SFT (LoRA) | test_hard | 37.5% (269) | 58/175 | 41.5% (188) | 35/94 | 9.3 |
| Base | test_hard | 46.8% (269) | 69/175 | 52.1% (188) | 41/94 | 10.1 |
| SFT (LoRA) | validation | 57.6% (257) | 115/213 | 68.2% (88) | 27/44 | 6.6 |
| Base | validation | 59.5% (257) | 119/213 | 67.0% (88) | 25/44 | 9.1 |
| GLM-5.3 | test | 88.9% (253) | 196/220 | 86.4% (66) | 28/33 | 5.3 |
| GLM-5.3 | test_hard | 69.9% (269) | 109/175 | 74.5% (188) | 61/94 | 8.6 |
| GLM-5.3 | validation | 93.0% (257) | 196/213 | 93.2% (88) | 39/44 | 5.1 |

Takeaways:

- **GRPO gets close to the teacher in OpenCode 2.3.2; SFT alone does not.** The GRPO student is within 2 points
  of GLM-5.3 on test, 5 below on validation and 2 above on test_hard, with half the teacher's tool calls.
  It trails the 1.2.4 GRPO v3 by 3-6 points.
- **The SFT student is no better than base**, unlike in 1.2.4 (merged SFT v2: 90.1/91.8/73.2). Its main
  failure is ending an episode with prose instead of the JSON answer (65 of 213 validation episodes,
  "no valid structured output"), often a one-line plan ("Let me start by running `hf papers info`...")
  with no tool call. In 1.2.4 this could not happen: the `StructuredOutput` tool forced
  `tool_choice: required`, so every turn was a tool call and the episode only ended with the answer.
  OpenCode 2 sends no `tool_choice` and ends the loop on the first text-only turn. GRPO's reward removes
  the behaviour (6 such episodes on test). Likely fixes, not tried here: a harness retry that asks for the
  JSON answer once when the final message is not valid JSON (1.x already does this when a provider rejects
  structured output), or SFT on traces without the spill-file habit (below).
- **GLM-5.3 in 2.3.2 is weaker on test_hard than GLM-5.2 in 1.2.4** (69.9% vs 81.8%): it hits the
  15-call limit (50 of 175 papers in the first run) and opens the project page after it already found
  GitHub (45). The same habit was the top trace rejection reason (570 of 2,772 attempts).
- **Caveat: traces and probe ran on a shared server.** The traces and the difficulty probe were generated
  before episodes ran on their own servers (see "One server per episode" below). 834 of the 1,641
  accepted traces read OpenCode's spilled shell output files, 76 searched `/tmp` or `*.out` (and could
  have read another episode's files), 3 ran `rm`. Every accepted trace still passed label verification,
  but the SFT student learned to hunt for output files (7-9 tool calls per episode). All reported
  evaluations are from the rerun with isolated episodes; the first evaluation (shared server) was
  discarded.

### Run

| Stage | Details | Job(s) |
|---|---|---|
| Traces | GLM-5.3, 2,772 v2 train attempts, 1,641 accepted (59%), 4.6 tool calls per accepted trace, $116 | 6 shards on cpu-upgrade |
| SFT | 1 epoch, bf16 LoRA, 32k max length, full trajectories, tool schemas captured from 2.3.2; 103 steps, final loss ~0.40, token accuracy ~0.87; 4 h 44 min on one H200 | `6b0412671fd37e5ecf00282c` |
| Probe | merged SFT, 700 papers x 4 attempts, tier 7 excluded | `6b04566808b6bdfb5bd0e1d2` and 3 more |
| GRPO | lr 2e-5, 150 steps, tiers 0 and 2-6 weighted x2, 421 papers ({0: 32, 1: 275, 2: 82, 3: 96, 4: 18, 5: 32, 6: 32} after weighting); reward ~0.5-0.7 at the start, ~1.03-1.09 over the last steps; 1 h 04 min on 2x H200 | `6b04604308b6bdfb5bd0f786` |
| Evaluation | 9 jobs (rerun with isolated episodes) | orchestrator `6b048e16ca8b51043c3a1812` |

Cost: about $278 ($116 teacher traces, $39 for two teacher evaluations, about $122 of HF Jobs
compute including the discarded first evaluation). Artifacts (collection
[paper-url-agent-frontier-to-small-distillation](https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e)):
[`qwen35-4b-url-agent-oc2-traces`](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-oc2-traces),
[`-lora`](https://huggingface.co/nielsr/qwen35-4b-url-agent-oc2-lora),
[`-sft-merged`](https://huggingface.co/nielsr/qwen35-4b-url-agent-oc2-sft-merged),
[`-grpo-lora`](https://huggingface.co/nielsr/qwen35-4b-url-agent-oc2-grpo-lora),
[`-grpo-merged`](https://huggingface.co/nielsr/qwen35-4b-url-agent-oc2-grpo-merged),
[`-eval`](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-oc2-eval).

Reproduce:

```bash
python pipeline.py --teacher glm-5.3 --student qwen3.5-4b --dataset nielsr/paper-url-extraction-v2 \
  --opencode-version 2.3.2 --run-name qwen35-4b-url-agent-oc2 \
  --stages traces,sft,merge,probe,grpo,eval --probe-exclude-tiers 7 --probe-args '--sample 700' \
  --grpo-args '--learning-rate 2e-5 --max-steps 150 --exclude-tiers 7 --tier-weights 0:2,2:2,3:2,4:2,5:2,6:2' \
  --splits test,validation,test_hard --eval-variants sft,grpo,teacher --eval-include-partial --eval-limit 1000 \
  --orchestrate
```

## OpenCode 1.2.4 -> 2.3.2: what changed for this harness

"OpenCode v2" is the 2.x line. The latest release is v2.3.2 (tagged 2026-11-18). It is published on npm as
`@opencode/cli`; the old `opencode-ai` npm package stops at 1.19.6. The GitHub release tarball holds the
same binary as the npm package (identical sha256), and the npm package carries SLSA provenance. The
harness pins the sha256 of both versions (`harness.opencode.OPENCODE_SHA256`). Everything below was checked
against the v2.3.2 source and a running 2.3.2 server. What 1.2.4 and 2.3.2 actually send to the model was
captured with a fake OpenAI-compatible endpoint.

| | OpenCode 1.2.4 | OpenCode 2.3.2 |
|---|---|---|
| Version output | `1.2.4` | `opencode v2.3.2` |
| Server | `opencode serve`, no auth, `GET /global/health` | `opencode serve`. HTTP Basic auth is always on (`opencode:<password>`); with no `OPENCODE_PASSWORD` the server generates a password and prints it once. Health: `GET /api/info` |
| Routes | `/session`, `/session/{id}/message` (synchronous), `/session/{id}/abort` | Everything under `/api/`, with no 1.x compatibility layer. `POST /api/session` -> `{data:{id}}`; `POST /api/session/{id}/prompt` is asynchronous; wait with `POST /api/experimental/session/{id}/wait`; messages are paged (`?order=asc&limit=200`, `cursor.next`); `POST /api/session/{id}/interrupt` |
| Python SDK | `opencode-ai` 0.1.0a36 (generated from the 1.x OpenAPI spec) | None. The harness uses plain httpx (`OpenCodeV2Client`) |
| System prompt | `system` per request, **appended** to OpenCode's own 11.7k-character coding-agent prompt | No per-request `system`. A config agent's `system` **replaces** OpenCode's base prompt. OpenCode still appends a model and environment block (date, working directory, a hint about the `execute` tool); 2.7k characters in total |
| Tools | `tools: {name: bool}` per request. The model sees `bash`, `read`, `glob`, `grep`, `webfetch`, `StructuredOutput` | Permission rules `{action, resource, effect}`. The last matching rule wins; no match means "ask", which blocks a headless run. A tool is hidden only if the last rule matching it denies `*`. The model sees `shell`, `read`, `glob`, `grep`, `webfetch` |
| Tool schemas | `bash {command, timeout, workdir, description}`, `read {filePath, offset, limit}` | `shell {command, workdir, timeout, background}` (no `description`), `read {path, offset, limit}`, `grep` and `glob` gained options. `apply_patch` -> `patch`, `task` -> `subagent`; `codesearch` and `invalid` are gone; new tools: `execute` (Code Mode, on by default), MCP resources, image generation |
| Structured output | `format: {type: json_schema}` adds a `StructuredOutput` tool, forces `tool_choice: required`, and stores the answer in `info.structured` | **Removed.** The harness asks for a JSON-only final message (the schema is in the system prompt) and validates it with the same `URLFindings` model |
| Messages | `{info, parts}` with parts text/reasoning/tool/step-start/step-finish; tool `{tool, callID, state:{status, input, output, error}}` | Flat `{id, type, ...}`: types user/assistant/idle/...; one assistant message per model step, with `content: [text, reasoning, tool]`, `tokens`, `cost`, `finish`; tool `{name, id, state:{status, input, content:[{type:text}], error:{type, message}}}` |
| Extra model calls | A title-generation call per session, made with the session's model even when a title is given | None when a title is given |
| Config | `provider` with `npm`/`options` | `providers` with `package`/`settings`/`headers`, and `agents`. 1.x `provider` entries are migrated on read (the `X-HF-Bill-To` header and the HF token reach the router; verified) |
| Isolation | (none needed) | `OPENCODE_DISABLE_CLAUDE_CODE=1` (otherwise it loads `~/.claude` prompts and skills), `OPENCODE_DISABLE_MODELS_FETCH=1` (use the bundled models.dev catalog, which already has `huggingface/zai-org/GLM-5.3`), and per-workspace `XDG_*` directories (sessions live in SQLite) |

### Porting the harness

- **Version pin.** `--opencode-version` (default `1.2.4`) on `generate_opencode_traces.py`,
  `evaluate_vllm.py`, `difficulty_probe.py`, `train_grpo_harness.py`, `train_sft.py` and `pipeline.py`.
  Profiles may also pin it, and the pipeline refuses conflicting pins. `find_or_install_opencode` no longer
  uses whatever `opencode` is on PATH: it checks the binary's version and installs the pinned release
  otherwise. Installed binaries are checked against their sha256. The version is stored in every episode
  record (`opencode_version`), in the Pi session header, in the run manifest (`pipeline/run-*.json`), in
  collection notes, and in dataset cards (tag `opencode-<version>` and an "OpenCode" section). 1.2.4 runs
  are unchanged.
- **`harness/opencode_v2.py`.**
  - The `url-agent` config agent holds the v2 system prompt (`harness/system_prompt_v2.txt`) and
    permission rules that deny everything except `shell`, `read`, `grep`, `glob` and `webfetch` (plus
    `external_directory`, as in 1.x).
  - The same rules are repeated on each session. OpenCode appends its own `skill:browser` deny after the
    agent's rules, which otherwise leaves `skill` visible; GLM-5.3 called it in the first probe.
  - An episode is: create the session, post the prompt, wait until idle, then page through the messages.
    The 15-call watchdog and the episode timeout interrupt the session.
  - The messages are converted to the 1.x `{info, parts}` shape, so the scorer, the SFT trajectory, the
    GRPO reward and the Pi session export are shared. Tool names are kept (`shell`); the scorer already
    treated `shell` like `bash`.
- **One server per episode.** OpenCode 2 writes long shell output to files in its data directory
  (`<XDG_DATA_HOME>/opencode/shell/<project>/sh_*.out`) and tells the agent to `read`/`grep` them; about half
  of the accepted GLM-5.3 traces do this. The first evaluation ran 8 concurrent episodes on one server with one
  shared workspace (as 1.x did), so agents found and read other episodes' output files (`find /tmp -name
  "*.out"`), and one base-model episode ran `rm -rf .../opencode/shell`, which broke the tool calls of every
  episode running beside it (tool success fell to 45-60%). Each 2.x episode in the evaluator, the trace
  generator and the probe now starts its own server with its own workspace and data directory
  (`harness.opencode.isolated_episode_server`), killed with its process group when the episode ends. GRPO
  rollouts already did this. The shell still runs as the job's user, so an agent that deliberately searches
  `/tmp` can still see other episodes' workspaces; nothing is shared by default any more.
- **System prompt.** It is identical to 1.x except for "You may use Shell, ..." and the answer contract (a
  JSON-only final message with the schema). OpenCode's 1.x base prompt (11.7k characters of coding-agent
  instructions) is gone, so every turn's prompt is much shorter.
- **Credentials.** The server password is generated by OpenCode and read from its stdout, so it never
  enters the environment the agent's shell inherits. HF tokens are stripped as before (evaluation, probe
  and GRPO rollouts). Records are redacted by `record_line`.
- **SFT.** 1.2.4 students were trained with hand-written tool schemas (`train_sft.TOOLS`, kept for
  reproducibility). For 2.3.2, `train_sft.py --opencode-version 2.3.2` uses the exact tool definitions
  OpenCode sends to the model (`harness/opencode_tools/opencode-2.3.2.json`, captured from the server).
  It refuses traces from another OpenCode version.
- **GRPO.** The rollout session starts a 2.x server per rollout (password, config agent, no tokens) and
  runs the same v2 episode. `tool_choice` is no longer forced (2.x sends none), so
  `UnconstrainedToolChoice` is a no-op.
- **Tests.**
  - `tests/test_opencode_v2.py` converts every parity episode to 2.x messages and replays it through the
    evaluator's and the generator's `run_one` against a fake v2 server. The scores, and the GRPO reward of
    the same messages, must match the 1.x expectations.
  - Further tests cover message normalization (interrupted and failed calls), the permission rules, server
    config and environment, version parsing and pinning, the pipeline flags, the SFT tool schemas, the Pi
    export, and the evaluator's batch loop.
  - An opt-in test (`OPENCODE_V2_BINARY=...`) runs the real 2.3.2 binary against a fake model. It checks
    the tools and system prompt the model receives, that no title call is made, and that the answer
    parses.

Not ported: the production endpoint runner (`url_extraction_agent/opencode_endpoint.py`) and the demo Space
still run OpenCode 1.2.4.

### Smoke tests (before any paid job)

GLM-5.3 on the same 8 v2 train papers, run locally:

| OpenCode | Accepted | Tool calls | Seconds per paper | Cost per paper |
|---|---:|---:|---:|---:|
| 1.2.4 | 7/8 | 5.9 | 46 | $0.070 |
| 2.3.2 | 7/8 | 6.3 | 30 | $0.046 |

The cost is lower in 2.3.2 because the system prompt no longer carries OpenCode's coding-agent prompt.
Base Qwen3.5-4B in 2.3.2 (one H200 job, 6 test papers, no upload) completed every episode: 4/6
correct; misses were 3 skipped metadata checks and 2 tool-call limits, the base model's known
failure modes.
