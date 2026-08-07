from run_endpoint_agent import attempt_from_episode, endpoint_config, unscored_row
from url_extraction_agent.schema import PaperCandidate


def candidate() -> PaperCandidate:
    return PaperCandidate(
        paper_id=7,
        arxiv_id="2601.00001",
        title="A paper",
        abstract="An abstract",
        missing_github=True,
        missing_project_page=True,
        priority=1,
        priority_source="test",
    )


def test_endpoint_config_keeps_token_out_of_config():
    config = endpoint_config("https://endpoint.example/", "repo/model")
    options = config["provider"]["hf-endpoint"]["options"]
    assert options == {
        "baseURL": "https://endpoint.example/v1",
        "apiKey": "{env:HF_TOKEN}",
        "headers": {"X-Scale-Up-Timeout": "600"},
    }


def test_unscored_row_never_turns_unknown_labels_into_negatives():
    row = unscored_row(candidate())
    assert row["github_label_quality"] == "unknown"
    assert row["project_page_label_quality"] == "unknown"


def test_attempt_from_episode_preserves_findings_and_tool_health():
    paper = candidate()
    attempt = attempt_from_episode(
        paper,
        {
            "findings": {
                "reasoning": "Official metadata.",
                "github_url": "https://github.com/example/repo",
                "project_page_url": None,
            },
            "attempt_complete": True,
            "failure_reason": None,
            "tool_events": [
                {
                    "tool_name": "bash",
                    "tool_input": {"command": "hf papers info 2601.00001 --format json"},
                    "success": True,
                }
            ],
        },
    )
    assert attempt.attempt_complete
    assert str(attempt.findings.github_url) == "https://github.com/example/repo"
    assert attempt.tool_events[0].success
