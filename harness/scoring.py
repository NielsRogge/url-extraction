"""The one scorer: URL verification, research-protocol completeness, and the GRPO reward built from them.

Teacher-trace acceptance, evaluation records and the GRPO reward all call `score_episode`, so a rollout
earns reward for exactly what the evaluator counts as correct and complete.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from harness.task import PaperCandidate, ToolEvent, URLFindings, candidate_from_row

FIELDS = ("github", "project_page")
# Label qualities whose reference (a URL or a verified null) is trusted enough to score against.
SCOREABLE_QUALITIES = frozenset({"gold", "silver", "verified_null"})

# GRPO reward shaping (see `rollout_reward`).
W_TOOL_EFFICIENCY = 0.1
TOOL_BUDGET = 15.0


# --------------------------------------------------------------------------------------------------
# Rows and labels
# --------------------------------------------------------------------------------------------------


def scoreable_fields(row: dict[str, Any]) -> list[str]:
    """Requested fields (`missing_*`) whose label is verified."""
    return [
        name for name in FIELDS if bool(row[f"missing_{name}"]) and row.get(f"{name}_label_quality") in SCOREABLE_QUALITIES
    ]


def fully_scoreable(row: dict[str, Any]) -> bool:
    """Every requested field has a verified label (the evaluation population)."""
    requested = [name for name in FIELDS if bool(row[f"missing_{name}"])]
    return bool(requested) and len(scoreable_fields(row)) == len(requested)


def partly_scoreable(row: dict[str, Any]) -> bool:
    """At least one requested field has a verified label; unverified fields are never scored."""
    return bool(scoreable_fields(row))


# --------------------------------------------------------------------------------------------------
# URL verification
# --------------------------------------------------------------------------------------------------


def canonical_url(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"https://{text}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    host = parsed.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    if host == "github.com":
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            path = "/" + "/".join(parts[:2])
    return urlunparse(("https", host, path.lower(), "", "", ""))


def score_findings(row: dict[str, Any], findings: Any | None) -> dict[str, Any]:
    """Score verified requested fields; with no answer, every such field counts as wrong.

    `findings` is any object with `github_url` / `project_page_url` attributes (or None). Validated datasets
    may list other accepted URLs for the same target in `reference_<field>_aliases` (e.g. a renamed GitHub
    repository's redirect).
    """
    fields: dict[str, bool | None] = {name: None for name in FIELDS}
    for name in scoreable_fields(row):
        if findings is None:
            fields[name] = False
            continue
        predicted = canonical_url(getattr(findings, f"{name}_url"))
        accepted = {canonical_url(row.get(f"reference_{name}_url"))}
        accepted |= {canonical_url(alias) for alias in row.get(f"reference_{name}_aliases") or []}
        fields[name] = predicted in accepted
    scored = [value for value in fields.values() if value is not None]
    return {
        "fields": fields,
        "num_scoreable": len(scored),
        "all_scoreable_correct": findings is not None and bool(scored) and all(scored),
    }


def score_missing_findings(row: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible name for scoring an episode without an answer."""
    return score_findings(row, None)


def correctness(verification: dict[str, Any]) -> float:
    """Fraction of verified fields that are correct (0 when none are scoreable)."""
    scored = [value for value in (verification.get("fields") or {}).values() if value is not None]
    return sum(bool(value) for value in scored) / len(scored) if scored else 0.0


# --------------------------------------------------------------------------------------------------
# Research protocol
# --------------------------------------------------------------------------------------------------


def _is_shell(event: ToolEvent) -> bool:
    return event.tool_name.lower() in {"bash", "shell"}


def _is_webfetch(event: ToolEvent) -> bool:
    return event.tool_name.lower() in {"webfetch", "web_fetch"}


def _successful_bash(events: list[ToolEvent], fragment: str) -> bool:
    fragment = fragment.lower()
    return any(
        event.success and _is_shell(event) and fragment in str(event.tool_input.get("command", "")).lower()
        for event in events
    )


def _masks_hf_exit_status(event: ToolEvent) -> bool:
    if not _is_shell(event):
        return False
    command = str(event.tool_input.get("command", "")).strip().lower()
    return command.startswith(("hf papers info ", "hf papers read ")) and any(
        operator in command for operator in ("|", ";", "&&")
    )


RESEARCH_HOSTS = {"api.github.com", "github.com", "huggingface.co", "arxiv.org", "google.com", "paperswithcode.com"}


def _is_external_page_fetch(event: ToolEvent) -> bool:
    if _is_webfetch(event):
        url = str(event.tool_input.get("url", ""))
    elif _is_shell(event):
        command = str(event.tool_input.get("command", ""))
        if "curl" not in command.lower():
            return False
        match = re.search(r"https?://[^\s\"']+", command)
        url = match.group(0) if match else ""
    else:
        return False
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return bool(host) and host not in RESEARCH_HOSTS


def _successful_arxiv_fetch(events: list[ToolEvent], arxiv_id: str, paths: tuple[str, ...]) -> bool:
    for event in events:
        url = str(event.tool_input.get("url", "")).lower()
        if (
            event.success
            and _is_webfetch(event)
            and "arxiv.org" in url
            and arxiv_id.lower() in url
            and any(path in url for path in paths)
        ):
            return True
    return False


def _successful_page_fetch(events: list[ToolEvent], url: str) -> bool:
    wanted = url.rstrip("/").lower()
    for event in events:
        if not event.success:
            continue
        if _is_webfetch(event) and str(event.tool_input.get("url", "")).rstrip("/").lower() == wanted:
            return True
        if _is_shell(event):
            command = str(event.tool_input.get("command", "")).lower()
            if "curl" in command and wanted in command:
                return True
    return False


def derive_attempt_complete(
    candidate: PaperCandidate, findings: Any, events: list[ToolEvent]
) -> tuple[bool, str | None]:
    """Whether the episode followed the research protocol in the system prompt (judged from tool events)."""
    github_discovered = False
    for event in events:
        if _masks_hf_exit_status(event):
            return False, "an hf papers command masked its exit status"
        if github_discovered and _is_external_page_fetch(event):
            return False, "project-page content was fetched after GitHub discovery"
        if event.success and "github.com/" in str(event.response or "").lower():
            github_discovered = True

    metadata_checked = _successful_bash(events, f"hf papers info {candidate.arxiv_id}") or _successful_arxiv_fetch(
        events, candidate.arxiv_id, ("/abs/",)
    )
    if not metadata_checked:
        return False, "paper metadata was not successfully inspected"

    unresolved = (candidate.missing_github and findings.github_url is None) or (
        candidate.missing_project_page and findings.project_page_url is None
    )
    paper_checked = _successful_bash(events, f"hf papers read {candidate.arxiv_id}") or _successful_arxiv_fetch(
        events, candidate.arxiv_id, ("/html/", "/pdf/")
    )
    if unresolved and not paper_checked:
        return False, "paper content was not successfully inspected"

    if candidate.missing_github and findings.github_url is None:
        project_url = findings.project_page_url or candidate.known_project_page_url
        if project_url and not _successful_page_fetch(events, str(project_url)):
            return False, "project page was not inspected for a GitHub link"
    return True, None


# --------------------------------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------------------------------


def score_episode(
    row: dict[str, Any],
    findings: URLFindings | None,
    events: list[ToolEvent],
    *,
    max_tool_calls: int | None,
) -> dict[str, Any]:
    """Verification and protocol completeness of one finished episode.

    Returns `verification` (per-field correctness), `attempt_complete`, and `incomplete_reason`. An episode
    that used `max_tool_calls` or more tool calls is incomplete (it was cut off by the harness watchdog).
    """
    if findings is None:
        complete, reason = False, "no valid structured output"
    else:
        complete, reason = derive_attempt_complete(candidate_from_row(row), findings, events)
    if max_tool_calls is not None and len(events) >= max_tool_calls:
        complete, reason = False, f"tool-call limit reached ({max_tool_calls})"
    return {
        "verification": score_findings(row, findings),
        "attempt_complete": complete,
        "incomplete_reason": reason,
    }


# Rule-based shortcut guard: the labels live in a public Hub dataset, so a policy rewarded for correct URLs
# could learn to look them up instead of researching. Any authored tool input matching these patterns scores
# 0 during RL. Matched against what the agent writes, never against tool outputs.
SHORTCUT_PATTERNS = (
    re.compile(r"paper-url-extraction|pwc-url-extraction|\bnielsr/|community-science-llm-traces", re.IGNORECASE),
    re.compile(r"\bhf\s+download\b|load_dataset|snapshot_download|hf_hub_download"),
    re.compile(r"\bHF_TOKEN\b|\bhf\s+auth\b|huggingface-cli\s+login|/\.cache/huggingface/token"),
)


def shortcut_violation(events: list[Any]) -> str | None:
    for event in events:
        authored = json.dumps(getattr(event, "tool_input", {}) or {})
        for pattern in SHORTCUT_PATTERNS:
            if pattern.search(authored):
                return f"{event.tool_name}: {pattern.pattern[:40]}"
    return None


def env_reward(score: dict[str, Any], events: list[ToolEvent], *, protocol_weight: float) -> float:
    """GRPO environment reward from a `score_episode` result.

    Fraction of verified fields correct; when the protocol was not followed (including hitting the tool-call
    limit), `protocol_weight` of that credit is withheld. Shortcut attempts score 0.
    """
    if shortcut_violation(events):
        return 0.0
    reward = correctness(score["verification"])
    if protocol_weight and not score["attempt_complete"]:
        reward *= 1.0 - protocol_weight
    return reward


def tool_efficiency_bonus(reward: float, tool_calls: int | None) -> float:
    """`W_TOOL_EFFICIENCY * (1 - tool_calls / TOOL_BUDGET)` on top of a fully correct, complete answer."""
    if reward < 1.0 or tool_calls is None:
        return reward
    return reward + W_TOOL_EFFICIENCY * max(0.0, min(1.0, 1.0 - tool_calls / TOOL_BUDGET))


def rollout_reward(
    row: dict[str, Any],
    findings: URLFindings | None,
    events: list[ToolEvent],
    *,
    protocol_weight: float,
    max_tool_calls: int | None,
) -> float:
    """The full GRPO reward of one rollout (environment reward plus the tool-efficiency bonus)."""
    score = score_episode(row, findings, events, max_tool_calls=max_tool_calls)
    return tool_efficiency_bonus(env_reward(score, events, protocol_weight=protocol_weight), len(events))
