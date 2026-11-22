import json
import os
import unittest

from generate_opencode_traces import configure_huggingface_billing, findings_from_text, portable_trajectory
from url_extraction_agent.research import canonical_url, derive_attempt_complete, score_findings
from harness.task import PaperCandidate, ToolEvent, URLFindings


def test_configure_huggingface_billing(monkeypatch):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"provider":{"huggingface":{"options":{"timeout":60}}}}')

    configure_huggingface_billing(("huggingface", "zai-org/GLM-5.2"), "huggingface")

    config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["huggingface"]["options"]["timeout"] == 60
    assert config["provider"]["huggingface"]["options"]["headers"]["X-HF-Bill-To"] == "huggingface"


class ResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = PaperCandidate(
            paper_id=1,
            arxiv_id="2601.22060",
            title="Vision-DeepResearch",
            abstract="abstract",
            missing_github=True,
            missing_project_page=True,
            priority=0,
            priority_source="test",
        )
        self.findings = URLFindings(
            reasoning="trusted HF metadata",
            github_url="https://github.com/Osilly/Vision-DeepResearch",
            project_page_url="https://osilly.github.io/Vision-DeepResearch/",
        )

    def test_canonical_github_url(self) -> None:
        self.assertEqual(
            canonical_url("http://WWW.GitHub.com/Osilly/Vision-DeepResearch/tree/main/"),
            "https://github.com/osilly/vision-deepresearch",
        )

    def test_unknown_is_not_scored(self) -> None:
        score = score_findings(
            {
                "missing_github": True,
                "missing_project_page": True,
                "github_label_quality": "silver",
                "project_page_label_quality": "unknown",
                "reference_github_url": "https://github.com/Osilly/Vision-DeepResearch",
                "reference_project_page_url": None,
            },
            self.findings,
        )
        self.assertEqual(score["fields"], {"github": True, "project_page": None})

    def test_rejects_project_fetch_after_github_discovery(self) -> None:
        events = [
            ToolEvent(
                tool_name="bash",
                tool_input={"command": "hf papers info 2601.22060 --format json"},
                success=True,
                response='{"github_repo":"https://github.com/Osilly/Vision-DeepResearch"}',
            ),
            ToolEvent(
                tool_name="webfetch",
                tool_input={"url": "https://osilly.github.io/Vision-DeepResearch/"},
                success=True,
                response="project",
            ),
        ]
        complete, reason = derive_attempt_complete(self.candidate, self.findings, events)
        self.assertFalse(complete)
        self.assertIn("after GitHub discovery", reason or "")

    def test_fenced_json_is_parsed_without_duplicate_final(self) -> None:
        text = "```json\n" + self.findings.model_dump_json() + "\n```"
        self.assertEqual(findings_from_text(text), self.findings)
        messages = [
            {
                "info": {"role": "assistant"},
                "parts": [{"type": "text", "text": text}],
            }
        ]
        trajectory = portable_trajectory(messages, self.findings)
        self.assertEqual(len([message for message in trajectory if message["role"] == "assistant"]), 1)

    def test_structured_output_call_is_the_terminal_answer(self) -> None:
        messages = [
            {
                "info": {"role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "id": "part-1",
                        "callID": "call-1",
                        "tool": "StructuredOutput",
                        "state": {
                            "status": "completed",
                            "input": self.findings.model_dump(mode="json"),
                            "output": "Structured output submitted successfully",
                        },
                    }
                ],
            }
        ]
        trajectory = portable_trajectory(messages, self.findings)
        assistant_messages = [message for message in trajectory if message["role"] == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0]["tool_calls"][0]["function"]["name"], "StructuredOutput")


if __name__ == "__main__":
    unittest.main()
