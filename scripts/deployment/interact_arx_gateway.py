#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Harness console: choose schema-defined tools over the gateway HTTP API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from robots.arx.gateway.client import ArxGatewayClient
from robots.arx.gateway.episode import AgentDecision, EpisodeDriver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    credentials = json.loads(args.capabilities.read_text())
    public = ArxGatewayClient(
        args.url, credentials["episode_id"], credentials["agent_capability"]
    )
    admin = httpx.Client(
        base_url=args.url,
        headers={"Authorization": "Bearer " + credentials["harness_capability"]},
        timeout=30.0,
    )

    def post(path, payload=None):
        response = admin.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def decide(event):
        print(json.dumps(event, indent=2), flush=True)
        with args.log.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(
            'Enter {"tool":"arx.hold","arguments":{"steps":15}}; one call per line.',
            flush=True,
        )
        value = json.loads(input("tool> "))
        return AgentDecision(**value)

    def register(request, *, source, evidence):
        # Full catalog is immutable and available separately; keep admission body bounded.
        post(
            "/admin/decisions",
            {
                "request": request.model_dump(),
                "source": source,
                "evidence": {
                    "snapshot": evidence["snapshot"],
                    "trigger": evidence["trigger"],
                },
            },
        )

    try:
        deadline = time.monotonic() + 60
        while True:
            post("/admin/heartbeat")
            try:
                public.catalog()
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 503 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)
        driver = EpisodeDriver(
            public,
            decide=decide,
            register_decision=register,
            heartbeat=lambda: post("/admin/heartbeat"),
            close_attempt=lambda: post("/admin/stop"),
            heartbeat_interval_s=10.0,
            operation_timeout_s=360.0,
        )
        final = driver.run()
        with args.log.open("a") as stream:
            stream.write(json.dumps({"final": final}) + "\n")
        print(json.dumps(final, indent=2))
    finally:
        public.close()
        admin.close()


if __name__ == "__main__":
    main()
