# Experiment: GLM-5.3 teacher -> Qwen3.5-4B student in OpenCode v2 (2.3.2)

The GLM -> Qwen3.5-4B recipe (traces -> SFT -> merge -> difficulty probe -> GRPO -> evaluation) rerun
end to end in **OpenCode 2.3.2** instead of 1.2.4, with GLM-5.3 as the teacher. Every earlier published
result ran in OpenCode 1.2.4. The reference is GRPO v3 of
[EXPERIMENT_glm52_qwen35-4b.md](EXPERIMENT_glm52_qwen35-4b.md#grpo-v3-dataset-v2-difficulty-probed-pool)
(GLM-5.2 traces from OpenCode 1.2.4 -> SFT v2 -> GRPO v3).

<!-- RESULTS -->

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
