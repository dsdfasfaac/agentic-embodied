# Copyright (c) 2026 Zetta Contributors
"""Authenticated episode HTTP surface; no environment or planner access."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from .contracts import GatewayError, ToolRequest


def create_app(worker, *, agent_capability: str, harness_capability: str):
    if (
        len(agent_capability) < 32
        or len(harness_capability) < 32
        or agent_capability == harness_capability
    ):
        raise ValueError("distinct strong capabilities required")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    prefix = f"/v1/episodes/{worker.episode_id}"

    @app.middleware("http")
    async def authenticate(request, call_next):
        secret = (
            harness_capability
            if request.url.path.startswith("/admin/")
            else agent_capability
        )
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + secret
        ):
            return JSONResponse({"error": {"code": "UNAUTHORIZED"}}, status_code=401)
        if not (
            request.url.path.startswith(prefix + "/")
            or request.url.path.startswith("/admin/")
        ):
            return JSONResponse({"error": {"code": "NOT_FOUND"}}, status_code=404)
        return await call_next(request)

    @app.exception_handler(GatewayError)
    async def gateway_error(request, exc):
        return JSONResponse(
            {
                "error": {"code": exc.code, "public_message": str(exc)},
                "write_certainty": "none",
            },
            status_code=409,
        )

    async def body(request):
        data = bytearray()
        try:

            async def read_body():
                async for chunk in request.stream():
                    data.extend(chunk)
                    if len(data) > 65536:
                        raise HTTPException(413, "request too large")

            await asyncio.wait_for(read_body(), timeout=10)
            return json.loads(
                data,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
                object_pairs_hook=unique_object,
            )
        except (ValueError, TimeoutError):
            raise HTTPException(422, "invalid JSON or body timeout") from None

    @app.get(prefix + "/catalog")
    def catalog():
        if worker.catalog is None:
            raise HTTPException(503, "episode initializing")
        return worker.catalog

    @app.get(prefix + "/observation")
    def observation():
        snapshot = worker.snapshot()
        if snapshot is None:
            raise HTTPException(503, "episode initializing")
        return snapshot

    @app.get(prefix + "/events")
    async def events(after: int = 0, wait_s: float = 0):
        if after < 0 or not 0 <= wait_s <= 30:
            raise HTTPException(422, "invalid event window")
        deadline = time.monotonic() + wait_s
        while True:
            result = worker.events(after)
            if result or time.monotonic() >= deadline or worker.closed:
                return {"events": result}
            await asyncio.sleep(0.05)

    @app.get(prefix + "/images/{content_id}")
    def image(content_id: str):
        if not re.fullmatch(r"rgb-[0-9a-f]{64}", content_id):
            raise HTTPException(404)
        # Publication is proven by the public observation stream, not path existence.
        registered = False
        with worker.lock:
            rows = worker.journal.db.execute(
                "SELECT payload FROM records WHERE kind='ObservationPublished' AND public=1"
            ).fetchall()
        for row in rows:
            if any(
                ref["content_id"] == content_id
                for ref in json.loads(row[0])["cameras"].values()
            ):
                registered = True
                break
        if not registered:
            raise HTTPException(404)
        return FileResponse(
            worker.output / "public" / "images" / (content_id + ".png"),
            media_type="image/png",
        )

    @app.post(prefix + "/operations")
    async def operation(request: Request):
        payload = await body(request)
        try:
            operation = ToolRequest.model_validate(payload)
        except ValidationError:
            with worker.lock:
                worker.journal.record("invalid_envelope", {"code": "VALIDATION_ERROR"})
            return JSONResponse(
                {"error": {"code": "VALIDATION_ERROR"}, "write_certainty": "none"},
                status_code=422,
            )
        result = worker.submit(operation)
        status_code = 202 if result["status"] in {"accepted", "running"} else 200
        return JSONResponse(
            result,
            status_code=status_code,
            headers={"Location": prefix + "/operations/" + operation.request_id},
        )

    @app.get(prefix + "/operations/{request_id}")
    def status(request_id: str):
        result = worker.status(request_id)
        if result is None:
            raise HTTPException(404)
        return result

    @app.post(prefix + "/operations/{request_id}/cancel")
    def cancel(request_id: str):
        return worker.cancel(request_id)

    @app.post("/admin/heartbeat")
    def heartbeat():
        worker.heartbeat()
        return {"renewed": True}

    @app.post("/admin/decisions")
    async def decision(request: Request):
        payload = await body(request)
        try:
            if set(payload) != {"request", "source", "evidence"}:
                raise ValueError("unknown decision fields")
            operation = ToolRequest.model_validate(payload["request"])
            worker.register_decision(
                operation, source=payload["source"], evidence=payload["evidence"]
            )
        except (ValidationError, ValueError):
            raise HTTPException(422, "invalid decision") from None
        return {"registered": True}

    @app.post("/admin/stop")
    def stop():
        worker.close()
        return {"closed": True}

    return app


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value
