"""Is the student model up? Answered without waking a scaled-to-zero Inference Endpoint.

Two deployments:
- Local vLLM on this Space's GPU (`STUDENT_URL` on 127.0.0.1/localhost): probe `/health`, no credentials.
- A Hugging Face Inference Endpoint (any other `STUDENT_URL`): ask the Inference Endpoints management
  API for the endpoint's state. Any authenticated request to the endpoint itself would trigger a
  scale-up, so the endpoint is never called here.

The token is only ever sent to huggingface.co and the Endpoints API, and it never appears in the result.
"""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any
from urllib.parse import urlparse

import httpx

ENDPOINTS_API = "https://api.endpoints.huggingface.cloud/v2/endpoint"
WHOAMI_API = "https://huggingface.co/api/whoami-v2"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

# Inference Endpoints `status.state` -> what the page shows.
ENDPOINT_STATES = {
    "running": "ready",
    "scaledToZero": "asleep",
    "pending": "starting",
    "initializing": "starting",
    "updating": "starting",
}
# Anything else ("paused", "failed", "updateFailed", ...) is "unavailable"; "unknown" means we could not ask.
STATES = ("ready", "asleep", "starting", "unavailable", "unknown")


def is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in LOOPBACK


def _same_url(a: str | None, b: str) -> bool:
    return bool(a) and a.rstrip("/").lower() == b.rstrip("/").lower()


async def _find_endpoint(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> str | None:
    """`namespace/name` of the endpoint serving `url`, searched in the token owner's namespaces."""
    me = (await client.get(WHOAMI_API, headers=headers)).raise_for_status().json()
    namespaces = [me.get("name")] + [org.get("name") for org in me.get("orgs") or []]
    for namespace in filter(None, namespaces):
        response = await client.get(f"{ENDPOINTS_API}/{namespace}", headers=headers)
        if response.status_code != 200:
            continue
        data = response.json()
        items = data.get("items", []) if isinstance(data, dict) else data
        for item in items:
            if _same_url((item.get("status") or {}).get("url"), url):
                return f"{namespace}/{item['name']}"
    return None


class StudentStatus:
    """Cached student-state lookups for `/api/status` (page loads must not hammer the Endpoints API)."""

    def __init__(self, url: str, token: str | None, endpoint: str | None = None, ttl: float = 15.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:  # fmt: skip
        self.url = url.rstrip("/")
        self._token = token
        self.endpoint = endpoint or None  # "namespace/name"; discovered from the URL when not configured
        self.ttl = ttl
        self._transport = transport
        self._cached: tuple[float, dict[str, Any]] | None = None

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def _local(self) -> dict[str, Any]:
        with suppress(Exception):
            async with self._client(3) as client:
                if (await client.get(f"{self.url}/health")).status_code == 200:
                    return {"state": "ready", "detail": "local vLLM"}
        return {"state": "unavailable", "detail": "local vLLM not responding"}

    async def _remote(self) -> dict[str, Any]:
        if not self._token:
            return {"state": "unknown", "detail": "no token to query the Inference Endpoints API"}
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with self._client(8) as client:
                if not self.endpoint:
                    self.endpoint = await _find_endpoint(client, self.url, headers)
                if not self.endpoint:
                    return {"state": "unknown", "detail": "no Inference Endpoint with this URL is visible to the token"}
                response = await client.get(f"{ENDPOINTS_API}/{self.endpoint}", headers=headers)
                if response.status_code != 200:
                    return {"state": "unknown", "detail": f"Inference Endpoints API returned {response.status_code}"}
                raw = (response.json().get("status") or {}).get("state") or ""
        except Exception as exc:  # noqa: BLE001 - report, don't fail the page; the message never holds the token
            return {"state": "unknown", "detail": f"Inference Endpoints API unreachable ({type(exc).__name__})"}
        return {"state": ENDPOINT_STATES.get(raw, "unavailable"), "detail": f"endpoint {raw or 'state missing'}"}

    async def get(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._cached and now - self._cached[0] < self.ttl:
            return self._cached[1]
        result = await (self._local() if is_local(self.url) else self._remote())
        self._cached = (now, result)
        return result
