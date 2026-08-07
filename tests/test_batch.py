from pathlib import Path

from url_extraction_agent.batch import result_record, summarize, write_jsonl_atomic
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
        "complete_episodes": 1,
        "exact_match_episodes": 1,
        "fields": {
            "github": {"correct": 2, "total": 2},
            "project_page": {"correct": 1, "total": 2},
        },
        "failure_reasons": {"timeout": 1},
    }
