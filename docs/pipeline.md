# The general recipe: harness, profiles, pipeline

One codebase distils any teacher into any student. Model-specific settings live in small profiles, the
task and its scoring live in one package, and one entry point runs the stages on HF Jobs.

```
profiles/teachers/*.toml  profiles/students/*.toml      pipeline.py  (stages -> HF Jobs)
            \                   /                           |
             harness/profiles.py                            v
                                        generate_opencode_traces.py  train_sft.py  train_grpo_harness.py
                                        evaluate_vllm.py  merge_adapter.py  difficulty_probe.py
                                                            |
                                                            v
                              harness/  task.py (prompt, answer schema)  scoring.py (the one scorer)
                                        opencode.py (episode runner, trajectories)  report.py  hooks.py
```

## The harness package

| Module | Contents |
|---|---|
| `harness/task.py` | System prompt (`system_prompt.txt`), user prompt, `URLFindings` answer schema (strict `^https?://` URLs, quoted `null` normalized), `PaperCandidate`, `ToolEvent` |
| `harness/scoring.py` | `canonical_url`, `score_findings` (reference aliases; no answer = every verified field wrong), `derive_attempt_complete` (research protocol), `score_episode` (both, plus the tool-call limit), the GRPO reward (`env_reward`, `tool_efficiency_bonus`, shortcut guard) |
| `harness/opencode.py` | Supported OpenCode versions (1.2.4, 2.3.2) with pinned sha256 and version checks; OpenCode tools, permissions and request body (1.x); `run_one` (one episode with watchdog and structured-output fallback; dispatches to 2.x); `build_record`; `portable_trajectory` (keeps reasoning as `reasoning_content`); OpenCode install, server and client |
| `harness/opencode_v2.py` | OpenCode 2.x: `url-agent` config agent (system prompt, permission rules), `/api` episode loop, v2 -> 1.x message normalization (see [the OpenCode v2 experiment](EXPERIMENT_glm53_qwen35-4b_opencode2.md#opencode-124---232-what-changed-for-this-harness)) |
| `harness/report.py` | Evaluation summary (`summarize`, McNemar) and `rescore` of published records |
| `harness/profiles.py` | Profile schema and loader |
| `harness/hooks.py` | Resolves profile hooks given as `module:function` |

`evaluate_vllm.py`, `generate_opencode_traces.py`, `train_grpo_harness.py`, `difficulty_probe.py` and the
demo Space import it, so a teacher trace is accepted, an evaluation episode is scored and a GRPO rollout is
rewarded by the same code. `tests/test_parity.py` replays the same episodes through the evaluator's and the
generator's `run_one` (against a fake OpenCode server) and through the GRPO reward and checks they agree.

What changed when the copies were merged (the evaluator's copy was the reference, so published evaluation
numbers are unchanged; see [Validation](#validation)):

- **Teacher traces** used an older copy: no reference aliases, a pydantic `HttpUrl` answer schema, and an
  episode without an answer left its fields unscored instead of wrong. They now use the strict schema the
  students are evaluated with, and no-answer fields count as wrong (acceptance is unaffected: such an
  episode was never accepted).
- **GRPO reward** now treats reaching `--max-tool-calls` (15, the evaluation limit) as an incomplete
  episode, as evaluation does: with `--protocol-weight 0.5` such a rollout earns half credit. Before, only
  the research-protocol check applied (and the tool-efficiency bonus was already 0 at 15 calls). The shortcut
  guard (looking up the labels on the Hub scores 0) is the one intentional difference from evaluation.
- **Trajectories** keep the teacher's reasoning as `reasoning_content` (from PR #9). `train_sft.py` drops it
  unless the student profile sets `sft.keep_reasoning`, so Qwen students train exactly as before.

## Profiles

One TOML file per model. Fields left out take the defaults in `harness/profiles.py` (Qwen3.5 on H200), so
`profiles/students/qwen3.5-4b.toml` is three lines.

Teacher (`profiles/teachers/<name>.toml`):

| Field | Meaning |
|---|---|
| `model` | Hub id |
| `opencode_model` | OpenCode `provider/model`; a provider pin is part of the id (`...:deepinfra`) |
| `provider` | the pinned Inference Provider, for documentation |
| `[opencode_model_entry]` | inline OpenCode model declaration, needed for ids not in OpenCode's catalog (pinned ids) |
| `opencode_version` | optional pin of the OpenCode release (`1.2.4` or `2.3.2`) for runs with this teacher |
| `[traces]` | `concurrency`, `attempts_per_row`, `accept_partial`, `episode_timeout`, `max_tool_calls`, `shards`, `flavor`, `timeout`, `baseline_concurrency` (teacher evaluation baseline, default 4) |

Student (`profiles/students/<name>.toml`):

| Section | Fields |
|---|---|
| top level | `model`, `requires` (extra repo files its hooks need; a run refuses to start without them), `opencode_version` (optional pin) |
| `[serve]` | `image`, `vllm` (`pep723`: scripts install vLLM; `image`: use the image's `vllm`), `flavor`, `timeout`, `deps`, `tool_call_parser`, `reasoning_parser`, `enforce_eager`, `language_model_only`, `max_model_len`, `max_output_tokens`, `gpu_memory_utilization`, `vllm_extra_args` |
| `[lora]` | `rank`, `sft_target_modules` (comma list or regex), `grpo_target_modules`, `adapter_converter` (`module:function(src, dst)` making an adapter servable by vLLM) |
| `[sft]` | `image`, `flavor`, `timeout`, `deps` (empty: the script's PEP 723 deps), `model_class`, `train_patch` (module with `enable(chunk_size=...)`), `keep_reasoning`, `args` |
| `[merge]` | `method` (`peft` or `module:function(args, adapter_dir, token)`), `image`, `flavor` (default `a100-large`: a bf16 PEFT merge wants a GPU), `timeout`, `deps`, `replace_deps` |
| `[probe]` | `flavor` (default: `serve.flavor`), `timeout`, `shards`, `sample`, `k`, `concurrency`, `temperature`, `episode_timeout`, `args` |
| `[grpo]` | `image`, `flavor`, `timeout`, `backend` (`trl` or `simple`), `deps`, `args` (appended after the recipe defaults; the last flag wins) |

Current profiles:

| Profile | Notable settings |
|---|---|
| `glm-5.2` | `huggingface/zai-org/GLM-5.2`, automatic routing |
| `glm-5.3` | `huggingface/zai-org/GLM-5.3`, automatic routing (in OpenCode 2.3.2's bundled catalog) |
| `kimi-k3.1-pro` | `huggingface/moonshotai/Kimi-K3.1-Pro` (served by Together) |
| `inkling-2-small` | pinned to DeepInfra (`:deepinfra`; inline model entry with `temperature`, limits and `cost`, without which OpenCode records $0), 6 concurrent sessions for distillation, 4 for the test baseline |
| `qwen3.5-4b` | all defaults: PyPI vLLM, `qwen3_coder`, text-only LoRA r32, H200 SFT/eval, 2x H200 TRL GRPO with 32 rollouts |
| `qwen3.5-9b` | defaults, 16 concurrent GRPO rollouts |
| `inkling-2-mini` | `vllm/vllm-openai:inkling-2-mini` image on RTX PRO 6000 (H200 for SFT), `inkling` parsers, attention-only LoRA r64 renamed by `inkling_adapter`, `inkling_train_patch`, reasoning distilled, attention merge on `cpu-xl`, `simple` GRPO backend (8 rollouts, tiers 7 excluded and harder tiers repeated, 14 h for 100 steps at 390-490 s each); needs PR #9's modules |

Inkling-2-Mini notes: vLLM prefill on RTX PRO 6000 is slow (~1-2k tokens/s), so GRPO and the probe use 8
concurrent rollouts, not 16. Never pass `--dspark`: the bundled drafter has no kernels for RTX PRO 6000.
Evaluator environments next to the image's vLLM use uv-managed Python (`--python-preference
only-managed`); the image's system Python lacks `Python.h`, so Triton cannot build against it.

To add a model, copy the closest profile. If it needs code the scripts do not have (an adapter rename, a
merge for a quantized checkpoint, a training patch), put it in a module, name it in the profile as
`module:function` and list the file under `requires` so job staging uploads it.

## pipeline.py

```bash
uv run pipeline.py --teacher kimi-k3.1-pro --student qwen3.5-9b --dataset nielsr/paper-url-extraction-v1 \
  --grpo-ids scripts/rl-kimi-9b-ids.txt
```

| Stage | Jobs | Output |
|---|---|---|
| `traces` | `[traces].shards` CPU jobs, disjoint `--shard i/n` slices of train | `<run>-traces` (`raw/train-<i>.jsonl` + sessions) |
| `sft` | one job on `sft.flavor`; trains on the accepted records of `--traces-files` | `<run>-lora` |
| `merge` | one job on `merge.flavor`: SFT adapter into the student | `<run>-sft-merged` |
| `probe` (opt-in) | `probe.shards` jobs: `difficulty_probe.py` on the merged SFT policy, K attempts per paper under rollout conditions | `<run>-eval/difficulty/probe-sft/shard-<i>/` incl. `qualified-ids.txt` |
| `grpo` | GRPO from `<run>-sft-merged` (`--launch`) on `grpo.flavor`, then a merge job on `merge.flavor` | `<run>-grpo-lora`, `<run>-grpo-merged` |
| `eval` | per `--eval-variants` (`base`, `sft`, `grpo`, `teacher`), each over every `--splits` split | `<run>-eval` (`sft/`, `sft-grpo/`, `teacher/`, ... + sessions + summaries) |
| `publish` | local | models and datasets public, code repo private, everything in the collection |

- `--stages sft,grpo` runs a subset; with several stages each stage waits for its jobs. A single stage runs
  detached unless `--wait`. `--dry-run` prints each job's image, flavor and full command.
- Any dataset with the `nielsr/paper-url-extraction-v1` schema works (`--dataset`), and any split can be
  evaluated: `--splits test,validation,test_hard`. Evaluation takes the first `--eval-limit` (50) fully
  scoreable papers of each split.
- `--opencode-version` (`1.2.4`, the default, or `2.3.2`) pins the OpenCode release of every stage (traces,
  SFT tool schemas, probe, GRPO rollouts, evaluation); a profile may pin it instead. It is recorded in the run
  manifest, every episode record and the dataset cards. Scripts never use another `opencode` on PATH.
- `--eval-args` passes extra `evaluate_vllm.py` flags to the student evaluations. Write it with `=` when
  the value starts with `--`, e.g. `--eval-args=--answer-retry` (OpenCode 2.x: ask once for the JSON
  answer when an episode ends without one).
- `<run>` defaults to `<student>-url-agent-<teacher>` (plus `-oc2` in OpenCode 2.x) under `--namespace` (nielsr). Override single repos
  with `--traces-repo`, `--sft-adapter`, `--sft-merged`, `--grpo-adapter`, `--grpo-merged`, `--eval-repo`.
- Code is staged to the private `<run>-code` model repo under `code/<rev>/`: the stage scripts, the whole
  `harness/` package, `trace_export.py`, the student's `requires` files and the `--grpo-ids` pool.
  `--rev` reuses a staged snapshot.
- Jobs run in the `huggingface` namespace and bill Inference Providers to `huggingface`; traces and
  evaluations publish agent-trace sessions through `trace_export.publish_sessions`.
- Recipe defaults: GRPO `--protocol-weight 0.5 --num-generations 8 --max-steps 100 --save-steps 50
  --gradient-accumulation-steps 2 --max-tool-calls 15`; evaluation 180 s episodes, 15 tool calls,
  concurrency 8. The teacher baseline uses the same limits; teacher *trace generation* keeps its profile's
  20-call limit.
- GRPO prompt pool: `--grpo-ids FILE`, or `--grpo-ids probe` (the default when the probe stage runs) for
  the union of the probe's qualified IDs: papers the SFT policy solves sometimes but not always. With the
  probe, a FILE restricts the probed candidates instead.
- `--probe-exclude-tiers 7` probes only scoreable train papers outside those difficulty tiers
  (`python -m harness.select`). `--grpo-args '...'` appends trainer arguments, e.g.
  `--learning-rate 2e-5 --max-steps 150 --exclude-tiers 7 --tier-weights 0:2,2:2,3:2,4:2,5:2,6:2`.
- Starting from an existing SFT adapter: pass `--sft-adapter <repo>` and run from `merge` on.
- `--grpo-merge-only` merges an already pushed GRPO adapter (recovery after a failed merge or eval).

### Credentials

Agents run shell commands, and some (base models especially) run `env` or print `os.environ`; their tool
output is stored in the published records. Three safeguards, added after GRPO v3's evaluation published
the job's HF token in raw episode records:

- The evaluator and the difficulty probe start OpenCode without `HF_TOKEN` (and other API keys) in its
  environment; the student is served by local vLLM and needs none. The teacher (trace generation and
  teacher baselines) still needs the token to call Inference Providers, so its shell can read it.
- Every episode record is written through `harness.opencode.record_line`, which redacts `hf_`, `sk-` and
  `ghp_` tokens (Pi sessions were already redacted by `trace_export.py`).
- Datasets are created private; the `publish` stage scans every text file for credential-shaped strings
  and refuses to make a dataset public if it finds one (models are created public).

### Reproducing the existing pairs

The first runs predate the naming scheme, so they pass their repo names:

```bash
# GLM-5.2 -> Qwen3.5-4B (GRPO v2 from SFT v2)
uv run pipeline.py --teacher glm-5.2 --student qwen3.5-4b --run-name qwen35-4b-url-agent-v2 \
  --traces-repo nielsr/paper-url-opencode-traces --traces-files train-v2.jsonl \
  --sft-adapter nielsr/qwen35-4b-url-agent-lora-v2 --grpo-adapter nielsr/qwen35-4b-url-agent-grpo-v2-lora \
  --grpo-merged nielsr/qwen35-4b-url-agent-grpo-v2-merged --eval-repo nielsr/qwen35-4b-url-agent-eval-v2 \
  --grpo-ids scripts/rl-v2-ids.txt --stages sft,merge,grpo,eval

# Kimi-K3.1-Pro -> Qwen3.5-9B
uv run pipeline.py --teacher kimi-k3.1-pro --student qwen3.5-9b --run-name qwen35-9b-url-agent-kimi \
  --traces-repo nielsr/paper-url-opencode-traces-kimi --traces-files train-kimi-k31.jsonl \
  --sft-adapter nielsr/qwen35-9b-url-agent-lora-kimi --grpo-adapter nielsr/qwen35-9b-url-agent-grpo-lora-kimi \
  --grpo-merged nielsr/qwen35-9b-url-agent-grpo-merged-kimi --eval-repo nielsr/qwen35-9b-url-agent-eval \
  --grpo-ids scripts/rl-kimi-9b-ids.txt --stages sft,merge,grpo,eval

# Inkling-2-Small -> Inkling-2-Mini (once PR #9's modules are merged)
uv run pipeline.py --teacher inkling-2-small --student inkling-2-mini --run-name inkling-2-mini-url-agent \
  --dataset nielsr/paper-url-extraction-v1-validated \
  --traces-repo nielsr/inkling-2-mini-url-agent-traces --traces-files train-inkling.jsonl \
  --grpo-ids scripts/rl-inkling-qualified-ids.txt --stages grpo,eval
# or let the pipeline select the pool: --grpo-ids scripts/rl-inkling-ids.txt --stages probe,grpo,eval
```

`scripts/serve_space.py --model <merged repo>` deploys the demo Space (it bundles `harness/`).

### Rescoring

```bash
uv run --extra traces python -m harness.report rescore --repo nielsr/qwen35-9b-url-agent-eval
uv run --extra traces python -m harness.report rescore --repo <traces repo> --files teacher-test.jsonl --max-tool-calls 20
uv run --extra traces python -m harness.report rescore --repo <eval repo> --dataset nielsr/paper-url-extraction-v1-validated
```

Recomputes every record's verification and completeness with `score_episode` from its stored answer and
raw OpenCode messages, and prints published summary, stored records and rescored numbers side by side with
a reason for every per-episode difference. Pass the tool-call limit the episodes ran with.

## Validation

Every published episode file was rescored with the unified scorer (2026-11-21, labels
`nielsr/paper-url-extraction-v1` unless noted):

| Records | Published / documented | Rescored | Per-episode differences |
|---|---|---|---|
| `qwen35-4b-url-agent-eval-v2` `sft-v2/`, `sft1-grpo/` (test, validation; base and adapter) | as published | identical | none |
| `qwen35-4b-url-agent-eval` (v1 SFT; test, validation) | as published | identical | none |
| `qwen35-9b-url-agent-eval` `sft/`, `sft-grpo/` | as published | identical | none |
| `inkling-2-mini-url-agent-eval` `base/`, `sft/` | as published | identical | none |
| Kimi teacher test baseline (`teacher-test-kimi-k31.jsonl`) | 88.1% | 88.1% (84 fields), 34/42 exact | 2 no-answer episodes: their 4 fields were stored as unscored (stored records alone give 92.5% over 80 fields) |
| Inkling-2-Small teacher test, DeepInfra / default routing / Together / OpenRouter | 85.7% / 60.7% / - / - | 85.7% / 60.7% / 48.8% / 67.9% | 32 no-answer episodes stored as unscored, as above |
| Accepted train traces (GLM `train.jsonl`, `train-v2.jsonl`; Kimi; Inkling) | 102 / 873 / 815 / 1,215 | all still accepted, 100% field accuracy, all complete | none |

So no published number moves. Two kinds of drift existed only in stored records written by the old
generator copy, and both were already corrected in the published tables (which counted no-answer fields as
wrong): no-answer episodes left their fields unscored, and its scorer ignored reference aliases (the v1
dataset has no aliases, so this changed nothing yet). Rescoring needs the limit an episode ran with:
the Inkling teacher baselines ran with 15 tool calls; with the generator default of 20, four of them would
flip from incomplete to complete.

Not rescorable: the README's Qwen3.5-4B SFT + GRPO row (84.5% / 93.0%, GRPO v2) was evaluated with
`--output-repo ''` and its episodes were never published, and the GLM-5.2 teacher's 83.3% test baseline
has no published record file. Rescored against the validated labels
(`nielsr/paper-url-extraction-v1-validated`: aliases, contradicted labels unscored), `sft-v2/` test moves
from 78.6% / 85.7% (base / SFT, 84 fields) to 88.0% / 92.0% (75 fields), all from fields that lose their
label; no episode changes because of an alias.
