"""OpenCode 2.x support: the v2 episode runner must score every parity episode exactly like 1.x.

The parity episodes of `test_parity.py` are converted to OpenCode 2 messages (flat messages, `shell` instead of
`bash`, the answer as a JSON-only final message instead of `StructuredOutput`) and replayed through the real
`run_one` of the evaluator and the trace generator against a fake v2 server (httpx MockTransport).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any

import httpx
import pytest

import evaluate_vllm
import generate_opencode_traces
import pipeline
import train_grpo_harness
from harness import opencode, opencode_v2
from harness.opencode import OpenCodeServer, OpenCodeV2Client, parse_findings
from harness.task import system_prompt_v2
from tests.test_parity import EPISODES, EXPECTED, MAX_TOOL_CALLS, PROTOCOL_WEIGHT, reward_from_record

V2 = "2.3.2"


def to_v2(messages: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    """A 1.x parity episode as the messages an OpenCode 2 server would return."""
    result: list[dict[str, Any]] = []
    for number, message in enumerate(messages):
        role = message["info"]["role"]
        if role == "user":
            text = "".join(part.get("text", "") for part in message["parts"])
            result.append({"id": f"msg_u{number}", "type": "user", "time": {"created": 1}, "text": text})
            continue
        content: list[dict[str, Any]] = []
        for part in message["parts"]:
            if part["type"] == "tool":
                if part["tool"] == "StructuredOutput":
                    continue
                state = part["state"]
                v2_state: dict[str, Any] = {"status": state["status"], "input": state["input"]}
                if state["status"] == "completed":
                    v2_state["content"] = [{"type": "text", "text": state.get("output", "")}]
                else:
                    v2_state["error"] = {"type": "tool.execution", "message": state.get("error", "")}
                name = "shell" if part["tool"] == "bash" else part["tool"]
                content.append({"type": "tool", "id": part["callID"], "name": name, "state": v2_state, "time": {}})
            else:
                content.append({"type": part["type"], "text": part.get("text", "")})
        structured = (response.get("info") or {}).get("structured")
        if structured is not None:  # 2.x has no structured output: the answer is a JSON-only final message
            content.append({"type": "text", "text": json.dumps(structured)})
        result.append(
            {"id": f"msg_a{number}", "type": "assistant", "time": {"created": 2, "completed": 3}, "agent": "url-agent",
             "model": {"id": "model", "providerID": "provider"}, "content": content, "finish": "stop",
             "cost": 0.001, "tokens": {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}}}
        )  # fmt: skip
    result.append({"id": "msg_idle", "type": "idle", "time": {"created": 4}, "outcome": "succeeded"})
    return result


class FakeOpenCodeV2:
    """The /api routes `harness.opencode_v2` calls, replaying one finished episode."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.requests: list[tuple[str, str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        assert request.headers["authorization"].startswith("Basic ")
        path = request.url.path
        if path == "/api/session":
            return httpx.Response(200, json={"data": {"id": "ses_v2"}})
        if path.endswith("/prompt"):
            return httpx.Response(200, json={"data": {"id": "msg_u0", "type": "user"}})
        if path.endswith("/wait"):
            return httpx.Response(204)
        if path.endswith("/message"):
            return httpx.Response(200, json={"data": self.messages, "cursor": {"previous": None, "next": None}})
        if path == "/api/info":
            return httpx.Response(200, json={"version": V2})
        if path.endswith("/interrupt"):
            return httpx.Response(200, json={"interrupted": True})
        return httpx.Response(404, json={"message": path})


def client_for(fake: FakeOpenCodeV2) -> OpenCodeV2Client:
    server = OpenCodeServer("http://opencode", V2, "secret", "/tmp/workspace")
    return OpenCodeV2Client(server, timeout=30, transport=httpx.MockTransport(fake))


def run_v2(module: Any, case: str) -> tuple[dict[str, Any], FakeOpenCodeV2]:
    data, messages, response = EPISODES[case]
    fake = FakeOpenCodeV2(to_v2(messages, response))

    async def go() -> dict[str, Any]:
        async with client_for(fake) as client:
            return await module.run_one(client, data, ("provider", "model"), episode_timeout=30,
                                        max_tool_calls=MAX_TOOL_CALLS)  # fmt: skip

    return asyncio.run(go()), fake


@pytest.mark.parametrize("case", sorted(EPISODES))
def test_v2_scores_like_v1(case: str) -> None:
    evaluation, _ = run_v2(evaluate_vllm, case)
    trace, _ = run_v2(generate_opencode_traces, case)
    fields, complete = EXPECTED[case]
    assert evaluation["verification"]["fields"] == fields
    assert evaluation["attempt_complete"] is complete
    assert evaluation["opencode_version"] == V2
    for key in ("findings", "verification", "attempt_complete", "failure_reason", "tool_events", "trajectory"):
        assert evaluation[key] == trace[key], key
    # The GRPO reward of the same (normalized) messages matches the evaluator's record.
    data, messages, response = EPISODES[case]
    normalized = opencode_v2.normalize_messages(to_v2(messages, response))
    findings, _ = parse_findings({}, normalized)
    env = train_grpo_harness.episode_reward(data, findings, normalized, protocol_weight=PROTOCOL_WEIGHT,
                                            max_tool_calls=MAX_TOOL_CALLS)  # fmt: skip
    tools = len(train_grpo_harness.tool_events(normalized))
    reward = train_grpo_harness.url_reward(type("R", (), {"env_reward": env, "tool_call_count": tools})())
    assert reward == pytest.approx(reward_from_record(evaluation))


def test_v2_requests() -> None:
    record, fake = run_v2(evaluate_vllm, "correct_and_complete")
    create = next(body for method, path, body in fake.requests if path == "/api/session")
    assert create["agent"] == opencode_v2.AGENT_ID
    assert create["model"] == {"providerID": "provider", "id": "model"}
    assert create["location"] == {"directory": "/tmp/workspace"}
    assert create["permissions"] == opencode_v2.PERMISSIONS  # repeated on the session: the session's rules win
    prompt = next(body for method, path, body in fake.requests if path.endswith("/prompt"))
    assert set(prompt) == {"text"} and prompt["text"].startswith("Paper ID: ")
    # The trajectory carries the v2 system prompt, and the JSON final message is the answer (nothing appended).
    trajectory = record["trajectory"]
    assert trajectory[0] == {"role": "system", "content": system_prompt_v2()}
    assert json.loads(trajectory[-1]["content"])["github_url"] == record["findings"]["github_url"]
    assert [event["tool_name"] for event in record["tool_events"]] == ["shell"]


def test_evaluator_runs_a_v2_batch(tmp_path, monkeypatch) -> None:
    """The evaluator's whole batch loop (incl. its end-of-batch health check) on a 2.x server."""
    from contextlib import asynccontextmanager

    data, messages, response = EPISODES["correct_and_complete"]
    fake = FakeOpenCodeV2(to_v2(messages, response))

    @asynccontextmanager
    async def fake_server(command, base_url=None, strip_credentials=False, version=None):
        assert strip_credentials and version == V2
        yield OpenCodeServer("http://opencode", V2, "secret", "/tmp/workspace")

    monkeypatch.setattr(evaluate_vllm, "opencode_server", fake_server)
    monkeypatch.setattr(evaluate_vllm, "opencode_client", lambda server, **kw: client_for(fake))
    monkeypatch.setattr(evaluate_vllm, "load_dataset", lambda *a, **k: [dict(data, paper_id=i) for i in range(2)])
    args = evaluate_vllm.argparse.Namespace(
        include_partial=False, dataset="d", split="test", paper_ids=None, start=0, limit=10, adapter="",
        adapter_only=False, base_model="m", opencode_binary="opencode", request_timeout=30, concurrency=1,
        episode_timeout=30, max_tool_calls=15, max_harness_restarts=3, opencode_version=V2,
    )  # fmt: skip
    output = tmp_path / "eval.jsonl"
    asyncio.run(evaluate_vllm.evaluate(args, output))
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2 and all(r["opencode_version"] == V2 and r["accepted"] for r in records)
    assert ("GET", "/api/info", None) in fake.requests


def test_each_episode_gets_its_own_server(monkeypatch) -> None:
    """Episodes on a server started by `opencode_server` run on their own server (own workspace and data dir):
    on a shared one, agents read each other's spilled shell output and one `rm -rf` broke the whole batch."""
    from contextlib import asynccontextmanager

    data, messages, response = EPISODES["correct_and_complete"]
    fakes: list[FakeOpenCodeV2] = []
    started: list[OpenCodeServer] = []
    stopped: list[str] = []

    class MockClient(OpenCodeV2Client):
        def __init__(self, server: OpenCodeServer, **kwargs: Any) -> None:
            fakes.append(FakeOpenCodeV2(to_v2(messages, response)))
            super().__init__(server, transport=httpx.MockTransport(fakes[-1]), **kwargs)

    @asynccontextmanager
    async def episode_server(server: OpenCodeServer, **kwargs: Any):
        assert server.executable == "/bin/opencode" and server.strip_credentials
        episode = OpenCodeServer(f"http://episode-{len(started)}", V2, "pw", f"/tmp/episode-{len(started)}")
        started.append(episode)
        try:
            yield episode
        finally:
            stopped.append(episode.url)

    monkeypatch.setattr(opencode, "OpenCodeV2Client", MockClient)
    monkeypatch.setattr(opencode, "isolated_episode_server", episode_server)
    shared = OpenCodeServer("http://shared", V2, "secret", "/tmp/shared", executable="/bin/opencode",
                            strip_credentials=True)  # fmt: skip

    async def go() -> list[dict[str, Any]]:
        async with MockClient(shared, timeout=30) as client:
            return await asyncio.gather(*(
                evaluate_vllm.run_one(client, data, ("provider", "model"), episode_timeout=30, max_tool_calls=15)
                for _ in range(3)
            ))  # fmt: skip

    records = asyncio.run(go())
    assert all(record["accepted"] for record in records)
    assert len(started) == 3 and len({s.workspace for s in started}) == 3 and len(stopped) == 3
    assert fakes[0].requests == []  # nothing ran on the shared server
    for fake, server in zip(fakes[1:], started):
        create = next(body for method, path, body in fake.requests if path == "/api/session")
        assert create["location"] == {"directory": server.workspace}


def test_opencode_server_records_how_to_start_episode_servers(monkeypatch) -> None:
    monkeypatch.setattr(opencode, "require_opencode_version", lambda executable, version: version)
    monkeypatch.setattr(opencode, "start_opencode", lambda executable, version, workspace, env, **kw: (
        type("P", (), {"pid": 0})(), OpenCodeServer("http://x", version, "pw", workspace)))  # fmt: skip
    monkeypatch.setattr(opencode, "stop_process", lambda process, **kw: None)

    async def go(version: str) -> OpenCodeServer:
        async with opencode.opencode_server("/bin/opencode", strip_credentials=True, version=version) as server:
            return server

    v2 = asyncio.run(go(V2))
    assert v2.executable == "/bin/opencode" and v2.strip_credentials
    assert asyncio.run(go("1.2.4")).executable is None


@pytest.mark.parametrize("retry", [False, True])
def test_answer_retry(retry: bool) -> None:
    """2.x ends an episode on a text-only turn; with answer_retry the harness asks once for the JSON answer."""
    data, messages, response = EPISODES["correct_and_complete"]
    full = to_v2(messages, response)
    answer = full[-2]  # the assistant message with the JSON answer (before idle)
    prose = {**answer, "id": "msg_prose", "content": [{"type": "text", "text": "Let me start with hf papers info."}]}
    first = [*full[:-2], prose, {"id": "msg_idle1", "type": "idle", "outcome": "succeeded"}]
    nudge = {"id": "msg_nudge", "type": "user", "text": opencode_v2.ANSWER_RETRY}
    second = [*first, nudge, {**answer, "id": "msg_answer"}, {"id": "msg_idle2", "type": "idle", "outcome": "succeeded"}]
    fake = FakeOpenCodeV2(first)
    prompts: list[str] = []
    original = fake.__call__

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/prompt"):
            prompts.append(json.loads(request.content)["text"])
            if len(prompts) == 2:
                fake.messages = second
        return original(request)

    async def go() -> dict[str, Any]:
        server = OpenCodeServer("http://opencode", V2, "secret", "/tmp/workspace")
        async with OpenCodeV2Client(server, timeout=30, transport=httpx.MockTransport(handler)) as client:
            return await evaluate_vllm.run_one(client, data, ("provider", "model"), episode_timeout=30,
                                               max_tool_calls=MAX_TOOL_CALLS, answer_retry=retry)  # fmt: skip

    record = asyncio.run(go())
    if retry:
        assert prompts[1] == opencode_v2.ANSWER_RETRY and len(prompts) == 2
        assert record["accepted"] and record["answer_retry"] is True
    else:
        assert len(prompts) == 1 and record["findings"] is None and "answer_retry" not in record


def test_answer_retry_is_v2_only() -> None:
    with pytest.raises(ValueError):
        asyncio.run(opencode.run_one(object(), {}, ("p", "m"), episode_timeout=1, max_tool_calls=1, answer_retry=True))


def test_pipeline_passes_eval_args() -> None:
    args = pipeline.parse_args(["--teacher", "glm-5.3", "--student", "qwen3.5-4b", "--opencode-version", V2,
                                "--eval-variants", "sft", "--splits", "test", "--eval-args=--answer-retry"])  # fmt: skip
    jobs = pipeline.eval_jobs(pipeline.make_run(args), "rev", args)
    assert len(jobs) == 1 and "--answer-retry" in jobs[0].command


def test_permissions_hide_everything_but_the_research_tools() -> None:
    def hidden(action: str, rules: list[dict[str, str]]) -> bool:  # OpenCode's `whollyDisabled`
        import fnmatch

        rule = next((r for r in reversed(rules) if fnmatch.fnmatchcase(action, r["action"])), None)
        return bool(rule) and rule["resource"] == "*" and rule["effect"] == "deny"

    # OpenCode appends its own rules after the agent's (e.g. skill:browser deny); session rules come last.
    agent = [{"action": "*", "resource": "*", "effect": "allow"}, *opencode_v2.PERMISSIONS,
             {"action": "skill", "resource": "browser", "effect": "deny"}]  # fmt: skip
    effective = agent + opencode_v2.PERMISSIONS
    for tool in ("shell", "read", "grep", "glob", "webfetch"):
        assert not hidden(tool, effective)
    for tool in ("skill", "execute", "question", "edit", "write", "patch", "subagent", "websearch", "todowrite"):
        assert hidden(tool, effective)
    assert not hidden("skill", agent)  # why the session repeats the rules


def test_normalize_messages_interrupted_and_failed() -> None:
    raw = [
        {"id": "u", "type": "user", "time": {"created": 1}, "payload": {"text": "hi"}},
        {"id": "a", "type": "assistant", "time": {"created": 2}, "model": {"id": "m", "providerID": "p"},
         "content": [{"type": "reasoning", "text": "think"},
                     {"type": "tool", "id": "c1", "name": "shell", "state": {"status": "running", "input": {"command": "sleep 99"}}},
                     {"type": "tool", "id": "c2", "name": "read", "state": {"status": "streaming", "input": '{"path": "x'}},
                     {"type": "tool", "id": "c3", "name": "webfetch",
                      "state": {"status": "error", "input": {"url": "u"}, "error": {"type": "tool.execution", "message": "boom"}}}]},
        {"id": "i", "type": "idle", "time": {"created": 3}, "outcome": "failed", "error": {"type": "provider", "message": "rate limited"}},
    ]  # fmt: skip
    user, assistant = opencode_v2.normalize_messages(raw)
    assert user["parts"] == [{"type": "text", "text": "hi"}]
    states = [part["state"] for part in assistant["parts"] if part["type"] == "tool"]
    assert [state["status"] for state in states] == ["error", "error", "error"]
    assert states[0]["error"] == "tool call running" and states[2]["error"] == "boom"
    assert states[1]["input"] == {"raw": '{"path": "x'}
    assert assistant["info"]["error"]["data"]["message"] == "rate limited"
    assert opencode.provider_errors(opencode_v2.normalize_messages(raw)) == ["rate limited"]


def test_server_config_adds_the_agent_and_keeps_providers() -> None:
    v1 = {"$schema": "https://opencode.ai/config.json", "share": "disabled", "small_model": "capture/policy",
          "provider": {"local-vllm": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://x/v1"}}}}  # fmt: skip
    config = opencode_v2.server_config(json.dumps(v1))
    assert config["provider"] == v1["provider"]  # migrated by OpenCode 2 itself
    assert "$schema" not in config and "share" not in config and "small_model" not in config
    agent = config["agents"][opencode_v2.AGENT_ID]
    assert agent["system"] == system_prompt_v2() and agent["permissions"] == opencode_v2.PERMISSIONS
    assert "JSON Schema" in agent["system"] and "Shell, Read" in agent["system"]


def test_server_env_keeps_secrets_out() -> None:
    base = {"PATH": "/bin", "HF_TOKEN": "hf_x", "OPENCODE_PASSWORD": "pw", "OPENCODE_CONFIG_CONTENT": "{}"}
    env = opencode.server_env(V2, base, workspace="/w", strip_credentials=True)
    assert "HF_TOKEN" not in env and "OPENCODE_PASSWORD" not in env
    assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1" and env["XDG_DATA_HOME"] == "/w/.opencode-data"
    assert opencode_v2.AGENT_ID in json.loads(env["OPENCODE_CONFIG_CONTENT"])["agents"]
    v1 = opencode.server_env("1.2.4", base, strip_credentials=False)
    assert v1 == base  # 1.x servers start exactly as before


def test_version_parsing_and_pinning(monkeypatch) -> None:
    assert opencode.parse_opencode_version("1.2.4\n") == "1.2.4"
    assert opencode.parse_opencode_version("opencode v2.3.2\n") == "2.3.2"
    assert opencode.parse_opencode_version("") is None
    assert opencode.opencode_api("1.2.4") == 1 and opencode.opencode_api("2.3.2") == 2
    with pytest.raises(ValueError):
        opencode.check_supported("1.18.14")
    installed: list[str] = []
    monkeypatch.setattr(opencode.shutil, "which", lambda command: "/usr/bin/opencode")
    monkeypatch.setattr(opencode, "binary_version", lambda path: "1.2.4")
    monkeypatch.setattr(opencode, "install_opencode", lambda dest, version: installed.append(version) or dest)
    assert opencode.find_or_install_opencode("opencode", "1.2.4") == "/usr/bin/opencode"
    assert opencode.find_or_install_opencode("opencode", "2.3.2").endswith("/opencode")  # PATH 1.2.4 is not used
    assert installed == ["2.3.2"]


def test_pipeline_pins_the_version_everywhere() -> None:
    assert pipeline.SUPPORTED_OPENCODE_VERSIONS == opencode.SUPPORTED_OPENCODE_VERSIONS
    assert pipeline.DEFAULT_OPENCODE_VERSION == opencode.OPENCODE_VERSION
    argv = ["--teacher", "glm-5.3", "--student", "qwen3.5-4b", "--opencode-version", V2, "--dry-run",
            "--stages", "traces,sft,probe,grpo,eval", "--eval-variants", "base,sft,grpo,teacher"]  # fmt: skip
    args = pipeline.parse_args(argv)
    run = pipeline.make_run(args)
    assert run.name == "qwen35-4b-url-agent-glm-53-oc2" and run.opencode_version == V2
    jobs = [job for stage in args.stages for job in pipeline.build_jobs(stage, run, "rev", args)]
    for job in jobs:
        if "merge" not in job.name:
            assert f"--opencode-version {V2}" in job.command, job.name
    default = pipeline.make_run(pipeline.parse_args(["--teacher", "glm-5.2", "--student", "qwen3.5-4b"]))
    assert default.opencode_version == "1.2.4" and default.name == "qwen35-4b-url-agent-glm-52"


def test_sft_uses_the_captured_v2_tool_schemas() -> None:
    train_sft = pytest.importorskip("train_sft")
    tools = train_sft.tools_for(V2)
    assert [tool["function"]["name"] for tool in tools] == ["glob", "grep", "read", "shell", "webfetch"]
    read = next(tool for tool in tools if tool["function"]["name"] == "read")
    assert read["function"]["parameters"]["required"] == ["path"]
    assert train_sft.tools_for("1.2.4") is train_sft.TOOLS
    train_sft.check_trace_version([{"opencode_version": V2}], V2)
    with pytest.raises(ValueError):
        train_sft.check_trace_version([{"paper_id": 1}], V2)  # a 1.2.4 trace (no version field) in a v2 run


def test_trace_export_renders_v2_records() -> None:
    import trace_export

    record, _ = run_v2(evaluate_vllm, "correct_and_complete")
    lines = trace_export.record_to_pi_session(record)
    calls = [c for line in lines if line.get("type") == "message" for c in line["message"].get("content") or []
             if isinstance(c, dict) and c.get("type") == "toolCall"]  # fmt: skip
    assert [call["name"] for call in calls] == ["shell"]
    assert trace_export.opencode_versions([record, {"paper_id": 1}]) == {V2, "1.2.4"}


# --------------------------------------------------------------------------------------------------
# Optional: the real OpenCode 2.3.2 binary against a fake model (set OPENCODE_V2_BINARY to run it)
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("OPENCODE_V2_BINARY"), reason="set OPENCODE_V2_BINARY to an OpenCode 2.3.2 binary")
def test_real_opencode_v2_against_a_fake_model(tmp_path, monkeypatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests: list[dict[str, Any]] = []
    answer = json.dumps({"reasoning": "x", "github_url": None, "project_page_url": None})

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            requests.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m"}
            for chunk in (
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ):  # fmt: skip
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = {"provider": {"local-vllm": {"npm": "@ai-sdk/openai-compatible", "options": {
        "baseURL": f"http://127.0.0.1:{server.server_address[1]}/v1", "apiKey": "EMPTY"},
        "models": {"policy": {"name": "policy", "limit": {"context": 65536, "output": 1024}}}}}}  # fmt: skip
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(config))
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak")
    binary = shutil.which(os.environ["OPENCODE_V2_BINARY"]) or os.environ["OPENCODE_V2_BINARY"]
    data = EPISODES["correct_and_complete"][0]

    async def go() -> dict[str, Any]:
        async with opencode.opencode_server(binary, strip_credentials=True, version=V2) as oc:
            assert oc.password and oc.api == 2
            async with opencode.opencode_client(oc, timeout=60) as client:
                return await evaluate_vllm.run_one(client, data, ("local-vllm", "policy"), episode_timeout=60,
                                                   max_tool_calls=15)  # fmt: skip

    record = asyncio.run(go())
    server.shutdown()
    assert record["findings"] == json.loads(answer) and record["opencode_version"] == V2
    assert len(requests) == 1  # no title-generation call
    request = requests[0]
    assert sorted(tool["function"]["name"] for tool in request["tools"]) == ["glob", "grep", "read", "shell", "webfetch"]
    assert request["messages"][0]["content"].startswith(system_prompt_v2())
    assert "tool_choice" not in request or request["tool_choice"] == "auto"
