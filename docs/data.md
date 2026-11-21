# Data, labels and reward validation

## Benchmark

[`nielsr/paper-url-extraction-v1`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v1) asks
for the official GitHub repository and project page of an arXiv paper. The verifier scores a URL field
only when its label quality is `gold`, `silver` or `verified_null`. An `unknown` label is excluded from
the reward and is never treated as a negative target. Of the 4,000 training rows, 1,732 have at least one
scoreable target (1,628 GitHub and 450 project-page targets). Evaluations use the fully scoreable papers
of the validation (50) and test (42) splits unless `--include-partial` is passed.

## Validated labels and difficulty tiers

[`nielsr/paper-url-extraction-v1-validated`](https://huggingface.co/datasets/nielsr/paper-url-extraction-v1-validated)
is the benchmark after three checks, all reproducible from this repository:

- **`validate_labels.py`** re-checks every scoreable label against current HF paper metadata, resolves
  GitHub redirects (renamed repositories become accepted aliases), stops scoring contradicted fields, and
  runs oracle / floor / canary checks of the scorer (the analogue of `openenv validate`). Train: 1,454
  labels agree with metadata, 486 are label-only, 70 were dropped, 69 aliases were added.
- **`tier_labels.py`** ranks each paper by the hardest source an agent must reach:

  | Tier | Source | Train | Validation | Test |
  |---|---|---:|---:|---:|
  | 1 | HF paper API | 1,229 | 158 | 155 |
  | 2 | abstract | 144 | 17 | 15 |
  | 3 | PDF page 1 | 136 | 13 | 19 |
  | 4 | PDF page 2 | 37 | 4 | 7 |
  | 5 | GitHub only via the project page | 9 | 0 | 1 |
  | 6 | later in the paper | 67 | 7 | 9 |
  | 7 | not found anywhere (weak label, excluded) | 76 | 13 | 12 |
  | 0 | verified "no URL" | 12 | 1 | 2 |

  Most papers are tier 1, so the benchmark mostly measures whether the agent calls the API.
- **`difficulty_probe.py`** lets the starting policy attempt every training paper K times under the exact
  RL conditions and keeps the papers it solves only sometimes: GRPO learns nothing from papers that are
  always or never solved.

## Mining harder papers

`mine_papers.py` adds new papers from HF daily papers with per-tier quotas, to get far more tier 2-6 and
verified-null examples. It collects every URL from the abstract, the full PDF (text and link annotations)
and project-page HTML; an LLM (GLM-5.3 by default) may only choose among those URLs; chosen URLs must
resolve. HF metadata gives `gold` labels, adjudicated live URLs `silver`, and papers with no official URL
anywhere `verified_null`. Mined rows are uploaded under `mined/` in the validated dataset.


## Agent traces on the Hub

Trace and evaluation datasets store one episode record per JSONL line (raw OpenCode messages, a portable
chat trajectory, tool events and verification). `trace_export.py` also writes each episode as a
Pi-format session file under `sessions/`, so the Hub's
[agent-traces viewer](https://huggingface.co/docs/hub/en/agent-traces) shows the reasoning, tool calls
and results, and the final verification of every episode. `generate_opencode_traces.py` and
`evaluate_vllm.py` publish sessions automatically; `uv run trace_export.py --repo <dataset>` backfills an
existing dataset.
