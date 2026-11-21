import json

from trace_export import record_to_pi_session, session_filename, write_sessions

RECORD = {
    "paper_id": 7,
    "arxiv_id": "2301.08243",
    "model": "huggingface/moonshotai/Kimi-K3.1-Pro",
    "session_id": "ses_1",
    "attempt": 1,
    "trajectory": [{"role": "system", "content": "You find URLs."}],
    "findings": {"github_url": "https://github.com/a/b", "project_page_url": None},
    "verification": {"fields": {"github": True}, "all_scoreable_correct": True},
    "opencode_messages": [
        {
            "info": {"role": "user", "time": {"created": 1_700_000_000_000}},
            "parts": [{"type": "text", "text": "Paper ID: 2301.08243"}],
        },
        {
            "info": {
                "role": "assistant",
                "time": {"created": 1_700_000_001_000},
                "modelID": "moonshotai/Kimi-K3.1-Pro",
                "providerID": "huggingface",
                "path": {"cwd": "/tmp/w"},
                "tokens": {"total": 10, "input": 8, "output": 2},
            },
            "parts": [
                {"type": "step-start"},
                {"type": "reasoning", "text": "Look up metadata first."},
                {
                    "type": "tool",
                    "callID": "c1",
                    "tool": "bash",
                    "state": {"status": "completed", "input": {"command": "env | grep hf_abcdefghijklmnopqrstuvwxyz"}, "output": "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123"},
                },
                {"type": "tool", "callID": "c2", "tool": "webfetch", "state": {"status": "error", "input": {"url": "x"}, "error": "404"}},
            ],
        },
    ],
}


def test_pi_session_structure():
    lines = record_to_pi_session(RECORD)
    header, *entries = lines
    assert header["type"] == "session" and header["version"] == 3 and header["harness"] == "opencode"
    assert header["id"] == "ses_1" and header["cwd"] == "/tmp/w"
    messages = [e["message"] for e in entries if e["type"] == "message"]
    roles = [m["role"] for m in messages]
    assert roles == ["custom", "user", "assistant", "toolResult", "toolResult", "custom"]
    assistant = messages[2]
    assert [c["type"] for c in assistant["content"]] == ["thinking", "toolCall", "toolCall"]
    assert isinstance(assistant["content"][1]["arguments"], dict)
    assert messages[3]["toolCallId"] == "c1" and messages[4]["isError"] is True
    # Entries form a single parent chain.
    ids = [e["id"] for e in entries]
    assert [e["parentId"] for e in entries] == [None, *ids[:-1]]


def test_tokens_redacted_and_filename(tmp_path):
    text = json.dumps(record_to_pi_session(RECORD))
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in text
    assert session_filename(RECORD) == "2301.08243_Kimi-K3.1-Pro_a1.jsonl"
    assert write_sessions([RECORD, {"arxiv_id": "x"}], tmp_path) == 1
    lines = (tmp_path / "2301.08243_Kimi-K3.1-Pro_a1.jsonl").read_text().splitlines()
    assert all(json.loads(line) for line in lines)
