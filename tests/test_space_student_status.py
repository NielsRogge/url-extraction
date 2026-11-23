import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "space"))

from student_status import ENDPOINTS_API, StudentStatus

TOKEN = "hf_test_secret"
ENDPOINT_URL = "https://abc123.us-east-1.aws.endpoints.huggingface.cloud"


def api(state: str | None, seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        host = request.url.host
        if host.endswith("endpoints.huggingface.cloud") and host != "api.endpoints.huggingface.cloud":
            raise AssertionError("the endpoint itself must never be called (it would scale up)")
        if host == "huggingface.co":
            return httpx.Response(200, json={"name": "nielsr", "orgs": [{"name": "huggingface"}]})
        path = request.url.path
        if path == "/v2/endpoint/nielsr":
            return httpx.Response(200, json={"items": []})
        if path == "/v2/endpoint/huggingface":
            return httpx.Response(200, json={"items": [{"name": "url-agent", "status": {"url": ENDPOINT_URL + "/"}}]})
        if path == "/v2/endpoint/huggingface/url-agent":
            return httpx.Response(200, json={"name": "url-agent", "status": {"state": state, "url": ENDPOINT_URL}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("running", "ready"), ("scaledToZero", "asleep"), ("initializing", "starting"), ("failed", "unavailable")],
)
def test_endpoint_state_comes_from_endpoints_api(raw, expected):
    seen: list[httpx.Request] = []
    status = StudentStatus(ENDPOINT_URL, TOKEN, transport=api(raw, seen))
    result = asyncio.run(status.get())
    assert result["state"] == expected
    assert status.endpoint == "huggingface/url-agent"
    assert TOKEN not in str(result)
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in seen)
    assert seen[-1].url == f"{ENDPOINTS_API}/huggingface/url-agent"


def test_configured_endpoint_skips_discovery_and_result_is_cached():
    seen: list[httpx.Request] = []
    status = StudentStatus(ENDPOINT_URL, TOKEN, endpoint="huggingface/url-agent", transport=api("running", seen))
    assert asyncio.run(status.get())["state"] == "ready"
    assert asyncio.run(status.get())["state"] == "ready"
    assert [str(r.url) for r in seen] == [f"{ENDPOINTS_API}/huggingface/url-agent"]


def test_endpoint_without_token_or_api_access_is_unknown():
    assert asyncio.run(StudentStatus(ENDPOINT_URL, None).get())["state"] == "unknown"
    forbidden = httpx.MockTransport(lambda request: httpx.Response(403))
    status = StudentStatus(ENDPOINT_URL, TOKEN, endpoint="huggingface/url-agent", transport=forbidden)
    assert asyncio.run(status.get())["state"] == "unknown"


def test_local_vllm_is_probed_without_credentials():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200 if request.url.path == "/health" else 404)

    status = StudentStatus("http://127.0.0.1:8000", TOKEN, transport=httpx.MockTransport(handler))
    assert asyncio.run(status.get())["state"] == "ready"
    assert "authorization" not in seen[0].headers

    down = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    assert asyncio.run(StudentStatus("http://127.0.0.1:8000", None, transport=down).get())["state"] == "unavailable"
