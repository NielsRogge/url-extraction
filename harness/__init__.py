"""Shared URL-agent harness: task definition, the one scorer, OpenCode episode runner, model profiles.

Imported by `generate_opencode_traces.py`, `evaluate_vllm.py`, `train_grpo_harness.py`, `difficulty_probe.py`
and `pipeline.py`. Job code staging copies this whole package (see `pipeline.STAGED_FILES`).
"""

from harness.scoring import (
    SCOREABLE_QUALITIES,
    canonical_url,
    correctness,
    derive_attempt_complete,
    fully_scoreable,
    partly_scoreable,
    rollout_reward,
    score_episode,
    score_findings,
    score_missing_findings,
)
from harness.task import SYSTEM_PROMPT, PaperCandidate, ToolEvent, URLFindings, candidate_from_row, user_prompt

__all__ = [
    "SCOREABLE_QUALITIES",
    "SYSTEM_PROMPT",
    "PaperCandidate",
    "ToolEvent",
    "URLFindings",
    "candidate_from_row",
    "canonical_url",
    "correctness",
    "derive_attempt_complete",
    "fully_scoreable",
    "partly_scoreable",
    "rollout_reward",
    "score_episode",
    "score_findings",
    "score_missing_findings",
    "user_prompt",
]
