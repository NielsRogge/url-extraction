from run_endpoint_agent import attempt_from_episode, endpoint_config, unscored_row
from url_extraction_agent.opencode_endpoint import StructuredFindings, wait_for_endpoint
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


def test_endpoint_warmup_retries_before_episode_clock(monkeypatch):
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.is_success = status_code == 200

        def raise_for_status(self):
            raise AssertionError("unexpected authentication failure")

    class Client:
        def __init__(self, **kwargs):
            self.responses = iter([Response(503), Response(200)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, *args, **kwargs):
            return next(self.responses)

    monotonic = iter([0.0, 0.0, 0.0, 1.0])
    monkeypatch.setattr("url_extraction_agent.opencode_endpoint.httpx.Client", Client)
    monkeypatch.setattr("url_extraction_agent.opencode_endpoint.time.monotonic", lambda: next(monotonic))
    monkeypatch.setattr("url_extraction_agent.opencode_endpoint.time.sleep", lambda _: None)
    wait_for_endpoint("https://endpoint.example", "secret", 10)


def test_structured_findings_normalizes_urls_in_the_wrong_role():
    findings = StructuredFindings.model_validate(
        {
            "reasoning": "Only unsupported pages were found.",
            "github_url": "https://example.com/not-github",
            "project_page_url": "https://github.com/example/repo",
        }
    )
    assert findings.github_url is None
    assert findings.project_page_url is None
