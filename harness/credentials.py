"""Credential patterns: never written to records, scanned for before a dataset is made public."""

from __future__ import annotations

import re

# Credentials an agent can print from its environment (e.g. `env | grep HF`).
# `hf_` and `ghp_` tokens are matched anywhere: records are redacted after JSON serialisation, where a token
# printed at the start of a line becomes `\nhf_...` and a coloured one `\u001b[0mhf_...`, so the character before
# the token is often a letter. `sk-` keys must not follow a letter or digit (unless that ends a JSON escape such
# as `\n`), so words such as "task-..." are not matched.
_SK_START = r"(?:(?<![A-Za-z0-9])|(?<=\\[nrtbf])|(?<=\\u[0-9a-fA-F]{4}))"
CREDENTIAL_RE = re.compile(r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|" + _SK_START + r"sk-[A-Za-z0-9_-]{20,}")
CREDENTIAL_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN", "OPENAI_API_KEY", "GITHUB_TOKEN")


def redact_credentials(text: str) -> str:
    return CREDENTIAL_RE.sub(lambda match: match.group(0)[:3] + "***REDACTED***", text)
