from evaluate_vllm import URLFindings, score_missing_findings, summarize


def test_output_schema_requires_null_or_an_absolute_http_url():
    schema = URLFindings.model_json_schema()

    assert set(schema["required"]) == {"reasoning", "github_url", "project_page_url"}
    url_variant = schema["properties"]["project_page_url"]["anyOf"][0]
    assert url_variant["pattern"] == r"^https?://[^\s]+$"


def test_output_parser_normalizes_only_exact_null_tokens():
    findings = URLFindings(reasoning="No project page found", github_url=None, project_page_url="\nnull")

    assert findings.project_page_url is None

    findings = URLFindings(
        reasoning="Found a project page", github_url=None, project_page_url="\nhttps://example.com/paper/\n"
    )
    assert findings.project_page_url == "https://example.com/paper/"


def test_missing_answer_counts_requested_scoreable_fields_as_wrong():
    score = score_missing_findings(
        {
            "missing_github": True,
            "missing_project_page": True,
            "github_label_quality": "gold",
            "project_page_label_quality": "unknown",
        }
    )

    assert score == {
        "fields": {"github": False, "project_page": None},
        "num_scoreable": 1,
        "all_scoreable_correct": False,
    }


def test_summary_keeps_failed_episodes_in_accuracy_denominator():
    records = [
        {
            "evaluation_model": "base",
            "verification": {"fields": {"github": False}, "all_scoreable_correct": False},
            "tool_events": [],
            "attempt_complete": False,
            "failure_reason": "no answer",
        },
        {
            "evaluation_model": "adapter",
            "verification": {"fields": {"github": True}, "all_scoreable_correct": True},
            "tool_events": [],
            "attempt_complete": True,
            "failure_reason": None,
        },
    ]

    summary = summarize(records)

    assert summary["models"]["base"]["field_accuracy"] == 0.0
    assert summary["models"]["adapter"]["field_accuracy"] == 1.0
    assert summary["comparison"]["field_accuracy_delta"] == 1.0
