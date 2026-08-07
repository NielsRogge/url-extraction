from pathlib import Path

from url_extraction_agent.batch import canonical_url, result_record, summarize, write_jsonl_atomic
from url_extraction_agent.schema import AgentAttempt, URLFindings


def test_result_scoring_canonicalizes_urls():
    source = {
        "reference_github_url": "https://github.com/Example/Repo",
        "reference_project_page_url": None,
        "github_label_quality": "consensus",
        "project_page_label_quality": "consensus",
        "trace_consensus": {},
    }
    attempt = AgentAttempt(
        paper_id=-1,
        arxiv_id="2501.00001",
        attempt_complete=True,
        result_subtype="success",
        findings=URLFindings(
            reasoning="Official metadata",
            github_url="http://www.github.com/example/repo/tree/main",
            project_page_url=None,
        ),
    )
    row = result_record(source, attempt)
    assert row["field_scores"] == {"github": True, "project_page": True}
    assert row["exact_match"]


def test_field_scoring_rejects_urls_in_the_wrong_role():
    source = {
        "reference_github_url": None,
        "reference_project_page_url": None,
        "github_label_quality": "trace_consensus",
        "project_page_label_quality": "trace_consensus",
        "trace_consensus": {},
    }
    attempt = AgentAttempt(
        paper_id=-1,
        arxiv_id="2501.00001",
        attempt_complete=True,
        findings=URLFindings(
            reasoning="No official URLs.",
            github_url="https://example.com/not-github",
            project_page_url="https://huggingface.co/example",
        ),
    )
    row = result_record(source, attempt)
    assert row["field_scores"] == {"github": True, "project_page": True}
    assert canonical_url("https://github.com/org/repo", "project_page") is None


def test_conflicting_official_references_accept_either_url():
    source = {
        "reference_github_url": "https://github.com/new/repo",
        "reference_github_urls": ["https://github.com/new/repo", "https://github.com/old/repo"],
        "reference_project_page_url": None,
        "reference_project_page_urls": [None],
        "github_label_quality": "trace_hf_conflict",
        "project_page_label_quality": "trace_consensus",
        "trace_consensus": {},
    }
    attempt = AgentAttempt(
        paper_id=-1,
        arxiv_id="2501.00001",
        attempt_complete=True,
        findings=URLFindings(
            reasoning="Official repository.",
            github_url="https://github.com/old/repo",
            project_page_url=None,
        ),
    )
    assert result_record(source, attempt)["exact_match"]


def test_atomic_checkpoint_and_summary(tmp_path: Path):
    rows = [
        {
            "arxiv_id": "2501.00001",
            "attempt": {"attempt_complete": True, "failure_reason": None},
            "field_scores": {"github": True, "project_page": False},
            "exact_match": False,
        },
        {
            "arxiv_id": "2501.00002",
            "attempt": {"attempt_complete": False, "failure_reason": "timeout"},
            "field_scores": {"github": True, "project_page": True},
            "exact_match": True,
        },
    ]
    output = tmp_path / "nested" / "results.jsonl"
    write_jsonl_atomic(output, rows)
    assert len(output.read_text().splitlines()) == 2
    assert not list(output.parent.glob("tmp*"))
    assert summarize(rows) == {
        "episodes": 2,
        "structured_episodes": 0,
        "complete_episodes": 1,
        "exact_match_episodes": 1,
        "complete_exact_match_episodes": 0,
        "fields": {
            "github": {"correct": 2, "total": 2, "complete_correct": 1, "complete_total": 1},
            "project_page": {"correct": 1, "total": 2, "complete_correct": 0, "complete_total": 1},
        },
        "failure_reasons": {"timeout": 1},
    }
