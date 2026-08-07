import pytest

from finalize_evaluation import finalize


def source(arxiv_id: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "reference_github_url": "https://github.com/example/repo",
        "reference_github_urls": ["https://github.com/example/repo"],
        "reference_project_page_url": None,
        "reference_project_page_urls": [None],
        "github_label_quality": "hf_metadata",
        "project_page_label_quality": "trace_consensus",
        "reference_sources": {"github": "hf_paper_metadata", "project_page": "trace_consensus"},
        "trace_consensus": {},
    }


def result(arxiv_id: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "evaluated_at": "2026-08-07T12:00:00+00:00",
        "attempt": {
            "paper_id": -1,
            "arxiv_id": arxiv_id,
            "findings": {
                "reasoning": "Official metadata.",
                "github_url": "https://github.com/example/repo",
                "project_page_url": None,
            },
            "attempt_complete": True,
            "failure_reason": None,
            "result_subtype": "success",
            "cost_usd": 0.0,
            "tool_events": [],
        },
    }


def test_finalize_rescores_and_preserves_timestamp():
    rows, summary = finalize([source("2501.00001")], [result("2501.00001")])
    assert rows[0]["exact_match"]
    assert rows[0]["evaluated_at"] == "2026-08-07T12:00:00+00:00"
    assert rows[0]["label_quality"]["github"] == "hf_metadata"
    assert summary["input_records"] == 1
    assert summary["unique_arxiv_ids"] == 1
    assert summary["label_quality"]["github"]["hf_metadata"]["correct"] == 1


def test_finalize_rejects_duplicate_or_incomplete_result_sets():
    with pytest.raises(ValueError, match="duplicate arxiv_id"):
        finalize([source("2501.00001")], [result("2501.00001"), result("2501.00001")])
    with pytest.raises(ValueError, match="ID mismatch"):
        finalize([source("2501.00001"), source("2501.00002")], [result("2501.00001")])
