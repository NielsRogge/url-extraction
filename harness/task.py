"""The URL-research task: system prompt, user prompt, and the answer and episode schemas.

Every stage (teacher traces, evaluation, GRPO rollouts, the demo) sends exactly this prompt and this
structured-output schema, so a change here changes all of them together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, HttpUrl, StringConstraints, field_validator

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text().strip()

# JSON Schema's `format: uri` is only an annotation for several guided decoders. A pattern is enforceable
# (e.g. by vLLM), which keeps prose such as "None.OnlyGitHub..." out of the URL fields.
StrictHttpUrl = Annotated[str, StringConstraints(pattern=r"^https?://[^\s]+$")]


class URLFindings(BaseModel):
    """The agent's final answer, submitted through OpenCode's structured-output tool."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    github_url: StrictHttpUrl | None
    project_page_url: StrictHttpUrl | None

    @field_validator("github_url", "project_page_url", mode="before")
    @classmethod
    def normalize_null_url(cls, value: Any) -> Any:
        # Small models occasionally quote JSON null or prefix it with a newline. Only exact null-like tokens
        # are normalized; prose and malformed URLs still fail validation and count as no answer.
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"", "null", "none"}:
                return None
        return value


class PaperCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: int
    arxiv_id: str
    title: str
    abstract: str
    missing_github: bool
    missing_project_page: bool
    known_github_url: HttpUrl | None = None
    known_project_page_url: HttpUrl | None = None
    priority: int = 0
    priority_source: str = "dataset"


class ToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_input: dict[str, Any]
    success: bool
    error: str | None = None
    response: str | None = None


def candidate_from_row(row: dict[str, Any]) -> PaperCandidate:
    return PaperCandidate(
        paper_id=int(row["paper_id"]),
        arxiv_id=str(row["arxiv_id"]),
        title=str(row["title"]),
        abstract=str(row["abstract"]),
        missing_github=bool(row["missing_github"]),
        missing_project_page=bool(row["missing_project_page"]),
        known_github_url=row.get("known_github_url"),
        known_project_page_url=row.get("known_project_page_url"),
        priority=int(row.get("priority", 0)),
        priority_source=str(row.get("priority_source", "dataset")),
    )


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
    return URLFindings.model_json_schema()
