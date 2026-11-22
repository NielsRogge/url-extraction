"""Credential patterns: never written to records, scanned for before a dataset is made public."""

from __future__ import annotations

import re

# Credentials an agent can print from its environment (e.g. `env | grep HF`).
CREDENTIAL_RE = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,})")
CREDENTIAL_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN", "OPENAI_API_KEY", "GITHUB_TOKEN")


def redact_credentials(text: str) -> str:
    return CREDENTIAL_RE.sub(lambda match: match.group(0)[:3] + "***REDACTED***", text)
