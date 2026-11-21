# Deployment and production runs (Qwen3.5-4B SFT v1)

How the merged August SFT model was served (Inference Endpoint, local MLX) and evaluated on unseen papers
with resumable HF Jobs batches. The demo Space now serves the GRPO v2 model on its own GPU instead (see `space/`).

## Managed Inference Endpoint (recommended production runtime)

`deploy_endpoint.py` creates an authenticated, OpenAI-compatible vLLM
Inference Endpoint on one L4. It pins the current merged-model revision, caps
the context at the evaluated 65,536 tokens, and scales to zero after 15 idle
minutes. The script is idempotent and will report an existing endpoint rather
than creating a duplicate.

The `HF_TOKEN` used for deployment must grant read and write access to
Inference Endpoints in the target namespace.

```bash
uv run deploy_endpoint.py
hf endpoints describe qwen35-4b-url-agent-vllm --namespace huggingface
```

Use the stable URL returned by `hf endpoints describe` with the production
OpenCode runner. The token remains an environment variable; it is never
written to disk or included in output artifacts. The runner injects it only
into the short-lived OpenCode server process.

```bash
export HF_ENDPOINT_URL="https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud"
uv run --extra traces python run_endpoint_agent.py \
  --input candidates.json \
  --output attempts.json
```

For interactive OpenCode use, set `HF_ENDPOINT_BASE_URL` to the same URL with
`/v1` appended and select the checked-in provider configuration:

```bash
export HF_ENDPOINT_BASE_URL="https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud/v1"
export OPENCODE_CONFIG="$PWD/opencode.endpoint.example.json"
opencode
```

The Endpoint is the intended hosted runtime. OpenCode still owns and executes
the Bash, Read, Grep, Glob, and WebFetch tool loop; only model inference is
remote. Scale-to-zero saves idle GPU cost but introduces a cold start, so a
production caller should retry transient 502/503 responses with a bounded
backoff. The provider sends `X-Scale-Up-Timeout: 600`, allowing Hugging Face's
proxy to hold the first request while a scaled-to-zero replica starts.

The deployed Endpoint passed an end-to-end OpenCode smoke test on held-out
paper `2604.11004`: it executed two `hf papers` Bash calls, returned structured
output, and matched both expected URLs.

## Resumable production batches on HF Jobs

The original `nielsr/paper-url-extraction-v1` dataset is no longer an unseen
canary source: its train split supplied SFT data and its validation and test
splits supplied the paired evaluations. `prepare_canary.py` instead reads the
private `huggingface/community-science-llm-traces` URL-extraction shards,
parses the JSON object from each `output_raw` value, canonicalizes the
`arxiv_id`, and excludes IDs present in any original split. Because the trace
dataset contains repeated model calls, it accepts a field only when at least
three traces agree with at least 60% support. The trace outputs are pseudo-labels,
not gold annotations: `prepare_canary.py` enriches them with current HF paper
metadata, records the provenance for each field, and retains both URLs when the
trace consensus and HF metadata disagree. Evaluation also rejects GitHub URLs
outside `github.com` and project pages on GitHub, arXiv, or Hugging Face.

The current trace snapshot has 2,039 rows and 201 valid arXiv IDs. Six overlap
the original dataset, leaving 195 unseen papers; 168 have consensus labels for
both URL fields. A small canary can be balanced across papers with both URLs,
GitHub only, project page only, and neither:

```bash
uv run --extra traces python prepare_canary.py \
  --source-parquet data/step_1_url_extraction/*.parquet \
  --limit 20 \
  --output canary-v1.jsonl
hf buckets cp canary-v1.jsonl \
  hf://buckets/nielsr/jobs-artifacts/qwen35-4b-url-agent-production/canary-v1-input.jsonl
```

The production batch module runs OpenCode on CPU while the managed L4
Endpoint performs inference. It first wakes the Endpoint and waits up to 15
minutes for `/health`, keeping cold-start time outside every paper's episode
budget. It atomically checkpoints every paper to the mounted bucket, skips
existing results after a restart, and optionally
publishes the completed results and summary to a private Hub dataset. Build a
wheel and store it under a revision directory so Jobs install immutable code
without requiring access to the private GitHub repository:

```bash
uv build --wheel --out-dir /tmp/url-agent-wheel
hf buckets cp /tmp/url-agent-wheel/url_extraction-0.1.0-py3-none-any.whl \
  hf://buckets/nielsr/jobs-artifacts/qwen35-4b-url-agent-production/packages/REVISION/url_extraction-0.1.0-py3-none-any.whl
```

Run that pinned wheel on a CPU Job:

```bash
hf jobs run \
  --namespace huggingface \
  --flavor cpu-upgrade \
  --timeout 2h \
  --secrets HF_TOKEN \
  --env HF_ENDPOINT_URL=https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud \
  --volume hf://buckets/nielsr/jobs-artifacts:/artifacts \
  --detach \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv run \
  --with 'url-extraction[traces] @ file:///artifacts/qwen35-4b-url-agent-production/packages/REVISION/url_extraction-0.1.0-py3-none-any.whl' \
  python -m url_extraction_agent.batch \
  --input /artifacts/qwen35-4b-url-agent-production/canary-v1-input.jsonl \
  --output /artifacts/qwen35-4b-url-agent-production/canary-v1-results.jsonl \
  --output-repo nielsr/qwen35-4b-url-agent-production
```

Pass `--retry-incomplete` only when intentionally rerunning incomplete
episodes. Completed records remain deduplicated by arXiv ID.

After a full run, use the checked-in finalizer to verify that inputs and results
contain the same unique arXiv IDs and to recompute enriched, provenance-aware
scores without another inference call:

```bash
uv run --extra traces python finalize_evaluation.py \
  --input unseen-v2-input.jsonl \
  --results unseen-v2-first-pass-results.jsonl \
  --output-results unseen-v2-first-pass-final-results.jsonl \
  --output-summary unseen-v2-first-pass-summary.json
```

The balanced 20-paper canary produced structured findings for 19 papers and
completed the full research protocol for 15. After enriching the trace labels
with current HF metadata and enforcing URL-field semantics, the unchanged
predictions scored 16/20 exact matches, 16/20 GitHub fields, and 20/20 project
page fields. The original trace-only score was 8/20 exact because five GitHub
repositories and two project pages present in current HF metadata were labelled
null by the older teacher traces. This label audit is therefore required before
using these traces as an evaluation or reward signal.

The complete unseen evaluation contains 168 papers and preserves its one-shot
result before any retry. The immutable first pass and the separate operational
artifact are both published in
[`nielsr/qwen35-4b-url-agent-production`](https://huggingface.co/datasets/nielsr/qwen35-4b-url-agent-production):

| Metric | Immutable first pass | One operational retry |
| --- | ---: | ---: |
| Exact paper-level match | 134/168 (79.8%) | 138/168 (82.1%) |
| GitHub URL correct | 145/168 (86.3%) | 147/168 (87.5%) |
| Project-page URL correct | 149/168 (88.7%) | 153/168 (91.1%) |
| Structured output | 144/168 (85.7%) | 159/168 (94.6%) |
| Full research protocol | 86/168 (51.2%) | 123/168 (73.2%) |
| Error result | 24/168 (14.3%) | 9/168 (5.4%) |

The first pass ran in
[`6a75dc58da2af92a634eeb68`](https://huggingface.co/jobs/nielsr/6a75dc58da2af92a634eeb68).
After freezing that raw artifact, Job
[`6a7617bc3e1f34a7e32bd914`](https://huggingface.co/jobs/nielsr/6a7617bc3e1f34a7e32bd914)
retried each of the 82 protocol-incomplete episodes exactly once in a separate
file. The operational result is useful for production reliability, but it does
not replace the one-shot benchmark.

Label provenance remains important. On the operational artifact, GitHub
accuracy was 40/40 for HF-metadata labels, 61/65 where traces and HF metadata
agreed, 15/16 on trace/HF conflicts, and 31/47 for trace-only consensus labels.
Project-page accuracy was respectively 14/14, 60/61, 3/3, and 76/90. Retries
improved the high-confidence HF-backed groups while reducing accuracy on a few
trace-only pseudo-labels, reinforcing that these results should not be treated
as a gold benchmark without human adjudication.

## Optional Apple-silicon smoke runtime

MLX was used only to prove that the merged checkpoint could run locally
without CUDA. On an Apple-silicon Mac, convert it to a local 4-bit model and
serve its OpenAI-compatible endpoint:

```bash
uv run --extra local convert_mlx.py
uv run --extra local mlx_lm.server \
  --model artifacts/qwen35-4b-url-agent-mlx-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 1024
```

Point an OpenCode `@ai-sdk/openai-compatible` provider at
`http://127.0.0.1:8080/v1` and configure its model ID as `default_model`.
MLX is used for the final laptop runtime because vLLM requires CUDA; vLLM
remains the reproducible GPU serving backend for the HF Jobs evaluation.
The checked-in example can be selected without replacing an existing OpenCode
configuration:

```bash
export OPENCODE_CONFIG="$PWD/opencode.local.example.json"
opencode
```

The checked-in converter also corrects Qwen3.5's generated MLX stop-token
metadata so the server stops on `<|im_end|>` instead of continuing to its
per-turn token limit. A real local OpenCode smoke test on held-out paper
`2604.11004` completed in 29 seconds, used Bash plus structured output, and
matched both expected URLs.
