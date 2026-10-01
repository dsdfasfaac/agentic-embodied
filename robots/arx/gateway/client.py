# Copyright (c) 2026 Zetta Contributors
"""Public client reconciles uncertain transport outcomes by the same request ID."""

from __future__ import annotations

import time

import httpx


class ArxGatewayClient:
    def __init__(
        self, base_url, episode_id, capability, *, timeout_s=30.0, transport=None
    ):
        self.http = httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": "Bearer " + capability},
        )
        self.prefix = f"/v1/episodes/{episode_id}"

    def _get(self, path, **kwargs):
        response = self.http.get(self.prefix + path, **kwargs)
        response.raise_for_status()
        return response.json()

    def observation(self):
        return self._get("/observation")

    def catalog(self):
        return self._get("/catalog")

    def events(self, after=0, wait_s=0):
        return self._get("/events", params={"after": after, "wait_s": wait_s})["events"]

    def image(self, content_id):
        response = self.http.get(self.prefix + "/images/" + content_id)
        response.raise_for_status()
        return response.content

    def status(self, request_id):
        return self._get("/operations/" + request_id)

    def submit(self, request):
        try:
            response = self.http.post(
                self.prefix + "/operations", json=request.model_dump()
            )
            response.raise_for_status()
            return response.json()
        except httpx.TransportError:
            # A failed status query is deliberately propagated; callers retain
            # the original request and must not invent a new motion identity.
            return self.status(request.request_id)

    def wait(self, request_id, *, timeout_s, poll_s=0.1):
        deadline = time.monotonic() + timeout_s
        while True:
            result = self.status(request_id)
            if result["status"] not in {"accepted", "running"}:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"query the same request ID: {request_id}")
            time.sleep(min(poll_s, max(0, deadline - time.monotonic())))

    def cancel(self, request_id):
        response = self.http.post(self.prefix + "/operations/" + request_id + "/cancel")
        response.raise_for_status()
        return response.json()

    def close(self):
        self.http.close()
