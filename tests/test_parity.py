"""The evaluator, the teacher-trace generator and the GRPO reward must score the same episode identically.

Each episode is replayed through the real `run_one` of `evaluate_vllm.py` and `generate_opencode_traces.py`
against a fake OpenCode server, and through `train_grpo_harness.episode_reward` / `url_reward`. The GRPO
reward is then checked against the value implied by the evaluator's record.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import evaluate_vllm
import generate_opencode_traces
import train_grpo_harness
from harness.opencode import parse_findings
from harness.scoring import TOOL_BUDGET, W_TOOL_EFFICIENCY, correctness

ARXIV = "2601.22060"
GITHUB = "https://github.com/Osilly/Vision-DeepResearch"
PAGE = "https://osilly.github.io/Vision-DeepResearch/"
MAX_TOOL_CALLS = 15
PROTOCOL_WEIGHT = 0.5


def row(**overrides: Any) -> dict[str, Any]:
    base = {
        "paper_id": 1,
        "arxiv_id": ARXIV,
        "title": "Vision-DeepResearch",
        "abstract": "abstract",
        "missing_github": True,
        "missing_project_page": True,
        "known_github_url": None,
        "known_project_page_url": None,
        "github_label_quality": "gold",
        "project_page_label_quality": "silver",
        "reference_github_url": GITHUB,
        "reference_project_page_url": PAGE,
    }
    return {**base, **overrides}


def tool(name: str, tool_input: dict[str, Any], output: str = "", status: str = "completed") -> dict[str, Any]:
    call = f"call-{abs(hash((name, json.dumps(tool_input)))) % 10**8}"
    return {"type": "tool", "id": call, "callID": call, "tool": name,
            "state": {"status": status, "input": tool_input, "output": output}}  # fmt: skip


INFO = tool("bash", {"command": f"hf papers info {ARXIV} --format json"}, f'{{"github_repo": "{GITHUB}"}}')
READ = tool("bash", {"command": f"hf papers read {ARXIV}"}, "paper text")


def answer(github: str | None, page: str | None) -> dict[str, Any]:
    return {"reasoning": "trusted HF metadata", "github_url": github, "project_page_url": page}


def episode(parts: list[dict[str, Any]], structured: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OpenCode messages and the prompt response for one episode."""
    if structured is not None:
        parts = [*parts, tool("StructuredOutput", structured, "Structured output submitted successfully")]
    messages = [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "prompt"}]},
        {"info": {"role": "assistant"}, "parts": parts},
    ]
    return messages, {"info": {"structured": structured} if structured is not None else {}}


EPISODES = {
    "correct_and_complete": (row(), *episode([INFO], answer(GITHUB, PAGE))),
    "verified_null_page": (
        row(project_page_label_quality="verified_null", reference_project_page_url=None),
        *episode([INFO, READ], answer(GITHUB, None)),
    ),
    "wrong_github": (row(), *episode([INFO], answer("https://github.com/someone/fork", PAGE))),
    "no_answer": (row(), *episode([INFO, {"type": "text", "text": "I could not decide."}], None)),
    "json_in_final_text": (
        row(),
        *episode([INFO, {"type": "text", "text": "```json\n" + json.dumps(answer(GITHUB, PAGE)) + "\n```"}], None),
    ),
    "quoted_null_is_null": (
        row(project_page_label_quality="verified_null", reference_project_page_url=None),
        *episode([INFO, READ], answer(GITHUB, "null")),
    ),
    "alias_accepted": (
        row(reference_github_url="https://github.com/osilly/new-name", reference_github_aliases=[GITHUB]),
        *episode([INFO], answer(GITHUB, PAGE)),
    ),
    "metadata_skipped": (row(), *episode([READ], answer(GITHUB, PAGE))),
    "page_fetched_after_github": (
        row(),
        *episode([INFO, tool("webfetch", {"url": PAGE}, "project page")], answer(GITHUB, PAGE)),
    ),
    "tool_call_limit": (
        row(),
        *episode([INFO] + [tool("bash", {"command": f"echo {i}"}, "ok") for i in range(MAX_TOOL_CALLS)], answer(GITHUB, PAGE)),
    ),
    "only_github_requested": (
        row(missing_project_page=False, known_project_page_url=PAGE),
        *episode([INFO], answer(GITHUB, None)),
    ),
    "unverified_label_unscored": (
        row(project_page_label_quality="unknown"),
        *episode([INFO], answer(GITHUB, "https://example.org/guess")),
    ),
}


class FakeOpenCode:
    """Replays one finished episode through the endpoints `run_one` calls."""

    def __init__(self, messages: list[dict[str, Any]], response: dict[str, Any]) -> None:
        self.messages, self.response = messages, response

    async def post(self, path: str, *, cast_to: Any, body: dict[str, Any] | None = None) -> httpx.Response:
        if path == "/session":
            payload: Any = {"id": "ses_test"}
        elif path.endswith("/message"):
            payload = self.response
        else:
            payload = True
        return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://opencode" + path))

    async def get(self, path: str, *, cast_to: Any) -> httpx.Response:
        return httpx.Response(200, json=self.messages, request=httpx.Request("GET", "http://opencode" + path))


def run(module: Any, case: str) -> dict[str, Any]:
    data, messages, response = EPISODES[case]
    return asyncio.run(
        module.run_one(
            FakeOpenCode(messages, response), data, ("provider", "model"), episode_timeout=30, max_tool_calls=MAX_TOOL_CALLS
        )
    )


def grpo_reward(case: str) -> float:
    data, messages, response = EPISODES[case]
    findings, _ = parse_findings(response, messages)  # what OpenCodeCaptureSession._run stores
    env = train_grpo_harness.episode_reward(
        data, findings, messages, protocol_weight=PROTOCOL_WEIGHT, max_tool_calls=MAX_TOOL_CALLS
    )
    tool_calls = len(train_grpo_harness.tool_events(messages))
    return train_grpo_harness.url_reward(SimpleNamespace(env_reward=env, tool_call_count=tool_calls))


def reward_from_record(record: dict[str, Any]) -> float:
    reward = correctness(record["verification"]) * (1 - PROTOCOL_WEIGHT * (not record["attempt_complete"]))
    if reward >= 1.0:
        reward += W_TOOL_EFFICIENCY * max(0.0, 1 - len(record["tool_events"]) / TOOL_BUDGET)
    return reward


@pytest.mark.parametrize("case", sorted(EPISODES))
def test_evaluator_generator_and_grpo_reward_agree(case: str) -> None:
    evaluation, trace = run(evaluate_vllm, case), run(generate_opencode_traces, case)
    for key in ("findings", "verification", "attempt_complete", "failure_reason", "tool_events", "trajectory"):
        assert evaluation[key] == trace[key], key
    # The generator only additionally requires every requested label to be verified before accepting a trace.
    assert trace["accepted"] == (evaluation["accepted"] and trace["all_requested_scoreable"])
    assert grpo_reward(case) == pytest.approx(reward_from_record(evaluation))


EXPECTED = {  # (fields, attempt_complete)
    "correct_and_complete": ({"github": True, "project_page": True}, True),
    "verified_null_page": ({"github": True, "project_page": True}, True),
    "wrong_github": ({"github": False, "project_page": True}, True),
    "no_answer": ({"github": False, "project_page": False}, False),
    "json_in_final_text": ({"github": True, "project_page": True}, True),
    "quoted_null_is_null": ({"github": True, "project_page": True}, True),
    "alias_accepted": ({"github": True, "project_page": True}, True),
    "metadata_skipped": ({"github": True, "project_page": True}, False),
    "page_fetched_after_github": ({"github": True, "project_page": True}, False),
    "tool_call_limit": ({"github": True, "project_page": True}, False),
    "only_github_requested": ({"github": True, "project_page": None}, True),
    "unverified_label_unscored": ({"github": True, "project_page": None}, True),
}


@pytest.mark.parametrize("case", sorted(EPISODES))
def test_expected_scores(case: str) -> None:
    record = run(evaluate_vllm, case)
    fields, complete = EXPECTED[case]
    assert record["verification"]["fields"] == fields
    assert record["attempt_complete"] is complete


def test_grpo_reward_values() -> None:
    assert grpo_reward("correct_and_complete") == pytest.approx(1.0 + W_TOOL_EFFICIENCY * (1 - 2 / TOOL_BUDGET))
    assert grpo_reward("wrong_github") == pytest.approx(0.5)
    assert grpo_reward("no_answer") == 0.0
    assert grpo_reward("metadata_skipped") == pytest.approx(0.5)  # protocol penalty, no bonus
    assert grpo_reward("tool_call_limit") == pytest.approx(0.5)  # the evaluator's limit is a protocol failure too


def test_shortcut_guard_is_the_only_intentional_difference() -> None:
    data, messages, response = EPISODES["correct_and_complete"]
    messages = json.loads(json.dumps(messages))
    messages[1]["parts"].insert(0, tool("bash", {"command": "hf download nielsr/paper-url-extraction-v1"}, "ok"))
    findings, _ = parse_findings(response, messages)
    reward = train_grpo_harness.episode_reward(
        data, findings, messages, protocol_weight=PROTOCOL_WEIGHT, max_tool_calls=MAX_TOOL_CALLS
    )
    assert reward == 0.0
    record = asyncio.run(
        evaluate_vllm.run_one(FakeOpenCode(messages, response), data, ("p", "m"), episode_timeout=30, max_tool_calls=15)
    )
    assert record["verification"]["all_scoreable_correct"]  # evaluation scores the answer; RL refuses to reward it


def test_records_with_lone_surrogates_are_written(tmp_path):
    """Tool output can hold lone surrogates (math symbols from papers); both writers must still emit JSONL."""
    from harness.opencode import record_line

    record = {"tool_events": [{"response": "x \ud835 y"}], "arxiv_id": ARXIV}
    path = tmp_path / "out.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record_line(record))
    assert json.loads(path.read_text(encoding="utf-8"))["tool_events"][0]["response"] == "x \ud835 y"


def test_evaluator_restarts_opencode_after_a_harness_failure(tmp_path, monkeypatch):
    """An OpenCode server whose shell tool cannot spawn must not be scored: restart and redo those papers."""
    from contextlib import asynccontextmanager

    data, good_messages, response = EPISODES["correct_and_complete"]
    broken_messages = json.loads(json.dumps(good_messages))
    broken_messages[1]["parts"][0]["state"]["output"] = "Error: ENOENT: no such file or directory, posix_spawn '/usr/bin/bash'"
    servers: list[str] = []

    @asynccontextmanager
    async def fake_server(command, base_url=None, strip_credentials=False, version=None):
        assert strip_credentials  # the agent's shell must not see the job's HF token
        assert version == "1.2.4"
        servers.append(command)
        yield "http://opencode"

    class FakeClientFactory:
        def __init__(self, server, **kwargs):
            # The first server is broken; every later one works.
            self.client = FakeOpenCode(broken_messages if len(servers) == 1 else good_messages, response)

        async def __aenter__(self):
            return self.client

        async def __aexit__(self, *exc):
            return False

    rows = [dict(data, paper_id=i, arxiv_id=f"{ARXIV}{i}") for i in range(3)]
    monkeypatch.setattr(evaluate_vllm, "opencode_server", fake_server)
    monkeypatch.setattr(evaluate_vllm, "opencode_client", FakeClientFactory)
    monkeypatch.setattr(evaluate_vllm, "load_dataset", lambda *a, **k: rows)
    args = evaluate_vllm.argparse.Namespace(
        include_partial=False, dataset="d", split="test", paper_ids=None, start=0, limit=10, adapter="",
        adapter_only=False, base_model="m", opencode_binary="opencode", request_timeout=30, concurrency=1,
        episode_timeout=30, max_tool_calls=15, max_harness_restarts=3, opencode_version="1.2.4",
    )  # fmt: skip
    output = tmp_path / "eval.jsonl"
    asyncio.run(evaluate_vllm.evaluate(args, output))
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(servers) == 2 and len(records) == 3
    assert not any("posix_spawn" in json.dumps(record) for record in records)


def test_records_never_contain_credentials():
    from harness.opencode import record_line

    token = "hf_" + "A1b2C3d4" * 4 + "xyzw1"
    line = record_line({"tool_events": [{"response": f"HF_TOKEN={token}", "tool_input": {"command": f"curl -H 'Bearer {token}'"}}]})
    assert token not in line and "hf_***REDACTED***" in line


@pytest.mark.parametrize(
    "output",
    [
        "line one\n{token}",  # a token printed at the start of a line: serialised as `\nhf_...`
        "a\t{token}",
        "a\r\n{token}",
        "\x1b[0m{token}",  # after a terminal escape, serialised as `\u001b[0m...` (here: `m` before the token)
        " {token}",  # a Unicode line break, written as the escape ` `
        "{token}",
    ],
)
def test_tokens_after_json_escapes_are_redacted(output):
    from harness.opencode import record_line

    tokens = ["hf_" + "A1b2C3d4" * 4, "ghp_" + "Z1x2C3v4" * 4]
    if not output.startswith("\x1b"):  # `sk-` right after a letter is indistinguishable from words like "task-"
        tokens.append("sk-" + "Q9w8E7r6" * 4)
    for token in tokens:
        line = record_line({"tool_events": [{"response": output.format(token=token)}]})
        assert token not in line, (output, token[:3])


def test_redaction_keeps_ordinary_words():
    from harness.credentials import redact_credentials

    text = "task-" + "abcdefgh" * 4 + " graph_" + "A1b2C3d4" * 4
    assert redact_credentials(text) == text  # `sk-` inside "task-..." is not a key
