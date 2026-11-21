"""Backward-compatible import path for the shared harness (prompt, URL scoring, research completeness)."""

from harness.opencode import json_text
from harness.scoring import canonical_url, derive_attempt_complete, score_findings
from harness.task import SYSTEM_PROMPT, findings_json_schema, user_prompt

__all__ = [
    "SYSTEM_PROMPT",
    "canonical_url",
    "derive_attempt_complete",
    "findings_json_schema",
    "json_text",
    "score_findings",
    "user_prompt",
]
