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
