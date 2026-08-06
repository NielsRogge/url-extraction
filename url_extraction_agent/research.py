"""Harness-independent prompt, URL scoring, and research-completeness helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from url_extraction_agent.schema import PaperCandidate, ToolEvent, URLFindings

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system_prompt.txt").read_text().strip()


def user_prompt(candidate: PaperCandidate) -> str:
    requested: list[str] = []
    if candidate.missing_github:
        requested.append("GitHub repository")
    if candidate.missing_project_page:
        requested.append("project page")
    return "\n".join(
        [
            f"Paper ID: {candidate.arxiv_id}",
            f"Title: {candidate.title}",
            f"Abstract: {candidate.abstract}",
            f"Find only: {', '.join(requested)}",
            f"Known GitHub URL: {candidate.known_github_url or 'none'}",
            f"Known project page URL: {candidate.known_project_page_url or 'none'}",
        ]
    )


def findings_json_schema() -> dict[str, Any]:
    """Schema sent to OpenCode's structured-output tool."""
    return URLFindings.model_json_schema(mode="serialization")


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


def score_findings(row: dict[str, Any], findings: URLFindings) -> dict[str, Any]:
    """Score only gold/silver URLs and explicitly verified nulls."""
    field_scores: dict[str, bool | None] = {}
    for name in ("github", "project_page"):
        if not bool(row[f"missing_{name}"]):
            field_scores[name] = None
            continue
        quality = row.get(f"{name}_label_quality")
        if quality not in {"gold", "silver", "verified_null"}:
            field_scores[name] = None
            continue
        predicted = canonical_url(getattr(findings, f"{name}_url"))
        expected = canonical_url(row.get(f"reference_{name}_url"))
        field_scores[name] = predicted == expected
    scoreable = [value for value in field_scores.values() if value is not None]
    return {
        "fields": field_scores,
        "num_scoreable": len(scoreable),
        "all_scoreable_correct": bool(scoreable) and all(scoreable),
    }


def _successful_bash(events: list[ToolEvent], fragment: str) -> bool:
    fragment = fragment.lower()
    return any(
        event.success
        and event.tool_name.lower() in {"bash", "shell"}
        and fragment in str(event.tool_input.get("command", "")).lower()
        for event in events
    )


def _masks_hf_exit_status(event: ToolEvent) -> bool:
    if event.tool_name.lower() not in {"bash", "shell"}:
        return False
    command = str(event.tool_input.get("command", "")).strip().lower()
    if not command.startswith(("hf papers info ", "hf papers read ")):
        return False
    return any(operator in command for operator in ("|", ";", "&&"))


def _is_external_page_fetch(event: ToolEvent) -> bool:
    if event.tool_name.lower() in {"webfetch", "web_fetch"}:
        url = str(event.tool_input.get("url", ""))
    elif event.tool_name.lower() in {"bash", "shell"}:
        command = str(event.tool_input.get("command", ""))
        if "curl" not in command.lower():
            return False
        match = re.search(r"https?://[^\s\"']+", command)
        url = match.group(0) if match else ""
    else:
        return False
    host = urlparse(url).netloc.lower().removeprefix("www.")
    research_hosts = {"api.github.com", "github.com", "huggingface.co", "arxiv.org", "google.com", "paperswithcode.com"}
    return bool(host) and host not in research_hosts


def _successful_arxiv_fetch(events: list[ToolEvent], arxiv_id: str, paths: tuple[str, ...]) -> bool:
    return any(
        event.success
        and event.tool_name.lower() in {"webfetch", "web_fetch"}
        and "arxiv.org" in str(event.tool_input.get("url", "")).lower()
        and arxiv_id.lower() in str(event.tool_input.get("url", "")).lower()
        and any(path in str(event.tool_input.get("url", "")).lower() for path in paths)
        for event in events
    )


def _successful_page_fetch(events: list[ToolEvent], url: str) -> bool:
    wanted = url.rstrip("/").lower()
    for event in events:
        if not event.success:
            continue
        if event.tool_name.lower() in {"webfetch", "web_fetch"} and (
            str(event.tool_input.get("url", "")).rstrip("/").lower() == wanted
        ):
            return True
        if event.tool_name.lower() in {"bash", "shell"}:
            command = str(event.tool_input.get("command", "")).lower()
            if "curl" in command and wanted in command:
                return True
    return False


def derive_attempt_complete(
    candidate: PaperCandidate,
    findings: URLFindings,
    events: list[ToolEvent],
) -> tuple[bool, str | None]:
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


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
