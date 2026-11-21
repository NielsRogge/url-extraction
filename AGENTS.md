# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor, etc.) working in this repository.

## Branches and worktrees

- Give each feature request and non-hotfix bug fix an isolated worktree and a dedicated `<agent>/<short-feature-name>` branch, where `<agent>` is the name of the coding agent doing the work (for example `codex/…`, `claude/…`, `cursor/…`).
- If the agent supplies a detached-HEAD worktree, create the branch before the first commit.
- Do not develop independent features in the primary `main` checkout.

Example:

```bash
git fetch origin main
git worktree add -b claude/new-feature ../url-extraction-new-feature origin/main
```

## Hugging Face billing

Always bill Hugging Face services to the `huggingface` org, never to a personal account:

- **Inference Providers:** send the `X-HF-Bill-To: huggingface` header (for example `InferenceClient(bill_to="huggingface")`, or `HF_BILL_TO=huggingface` / `--hf-bill-to huggingface` for `generate_opencode_traces.py`).
- **Jobs:** pass `--namespace huggingface` to `hf jobs run` / `hf jobs uv run`.
- **Inference Endpoints:** create and manage them in the `huggingface` namespace (`--namespace huggingface`).

## Cleanup after release

- Once the task's merged commit is live in production and its smoke checks pass, remove the task's worktree and delete its local and remote branch.
- First confirm the worktree is clean and that `main` contains the branch's changes (for a squash merge, compare the branch diff against the merge commit's diff).
- Clean up only the task's own worktree and branch.

Example:

```bash
git -C ../url-extraction-new-feature status --porcelain   # must be empty
git worktree remove ../url-extraction-new-feature
git branch -D claude/new-feature
git push origin --delete claude/new-feature
```

## Hub artifacts

- Every artifact this project publishes on the Hugging Face Hub (models, adapters, datasets, trace and evaluation datasets, and Spaces) must be added to the project collection: https://huggingface.co/collections/nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e
- Add it when the repo is created, with a one-line note saying what it is (teacher, student, stage).
- Visibility: models and datasets are public. Spaces, code snapshots (repos or buckets holding job code) and this GitHub repository stay private.
- Before making a dataset public, scan it for credentials (`hf_`, `sk-`, `ghp_` tokens); `trace_export.py` redacts HF tokens in sessions.
- Agent traces must always be uploaded in a form the Hub's [agent-traces viewer](https://huggingface.co/docs/hub/en/agent-traces) can open: one Pi-format session per episode under `sessions/`, produced by `trace_export.py`. Never upload episode JSONL with full trajectories (teacher traces, evaluation episodes) on its own; slim summaries such as `difficulty_probe.py` attempts are exempt.
  - `generate_opencode_traces.py --output-repo ...` and `evaluate_vllm.py --output-repo ...` already publish the sessions; prefer them over manual uploads.
  - After any other upload (e.g. `hf upload`, bucket copies promoted to a dataset), run `uv run trace_export.py --repo <dataset>` (optionally `--files <path>`).
  - New scripts that publish episodes must call `trace_export.publish_sessions`, and HF Jobs code staging must include `trace_export.py`.
  - Check that the dataset page shows the traces in Data Studio before reporting the upload as done.

Example:

```python
from huggingface_hub import HfApi

HfApi().add_collection_item(
    "nielsr/paper-url-agent-frontier-to-small-distillation-6b01c8378db0ff55d524014e",
    item_id="nielsr/<repo>", item_type="model",  # or "dataset" / "space"
    note="Qwen3.5-4B GRPO v3 adapter (from SFT v2)", exists_ok=True,
)
```
