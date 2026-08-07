import json

from evaluate_opencode import load_completed_ids, select_rows, summarize


def test_load_completed_ids_ignores_incomplete_lines(tmp_path):
    output = tmp_path / "episodes.jsonl"
    output.write_text('{"paper_id": 7}\nnot-json\n{}\n')

    assert load_completed_ids(output) == {7}


def test_select_rows_excludes_training_completed_and_unscoreable_rows():
    rows = [
        {"paper_id": 1, "missing_github": True, "github_label_quality": "gold", "missing_project_page": False},
        {"paper_id": 2, "missing_github": True, "github_label_quality": "gold", "missing_project_page": False},
        {"paper_id": 3, "missing_github": True, "github_label_quality": "unknown", "missing_project_page": False},
        {"paper_id": 4, "missing_github": False, "missing_project_page": True, "project_page_label_quality": "silver"},
    ]

    assert [row["paper_id"] for row in select_rows(rows, {1}, {2}, 0, 10)] == [4]


def test_summarize_reports_accuracy_behavior_and_training_overlap():
    records = [
        {
            "paper_id": 10,
            "verification": {"fields": {"github": True, "project_page": None}, "all_scoreable_correct": True},
            "tool_events": [{"success": False}, {"success": True}],
            "findings": {"github_url": "https://github.com/example/repo"},
            "attempt_complete": True,
        },
        {
            "paper_id": 11,
            "verification": {"fields": {"github": False, "project_page": None}, "all_scoreable_correct": False},
            "tool_events": [{"success": True}],
            "findings": None,
            "failure_reason": "no valid structured output",
            "attempt_complete": False,
        },
    ]

    summary = summarize(records, split="validation", model="local/url-agent", training_ids={10})

    assert json.loads(json.dumps(summary))["url_accuracy"] == {
        "correct": 1,
        "scoreable_fields": 2,
        "accuracy": 0.5,
        "all_scoreable_correct_episodes": 1,
    }
    assert summary["training_overlap_ids"] == [10]
    assert summary["agent_behavior"]["recovery_rate"] == 1.0
