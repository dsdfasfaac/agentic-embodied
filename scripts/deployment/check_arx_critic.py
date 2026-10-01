#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Validate a sealed critic with real isolated code and optionally real MuJoCo RGB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from robots.arx.critics import ArxCriticRegistry, WorkerLimits
from robots.arx.gateway.contracts import RuntimeLimits, ToolRequest
from robots.arx.gateway.journal import Journal
from robots.arx.gateway.public import ImageStore
from robots.arx.gateway.session_core import ArxSessionCore
from robots.arx.gateway.tools import default_registry
from robots.arx.gateway.zeva import ZevaPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    limits = WorkerLimits(
        python=sys.executable,
        max_history=60,
        max_evaluation_ms=5000,
        startup_timeout_s=15.0,
        memory_bytes=2147483648,
        cpu_seconds=60,
        scratch_bytes=1048576,
        image_width=320,
        image_height=240,
        max_message_bytes=2097152,
    )
    registry = ArxCriticRegistry(limits=limits)
    package = registry.register_package(args.package)
    critic = registry.freeze()
    if not args.scene:
        store = ImageStore(args.output / "images")
        results = []
        try:
            for step, red in enumerate([0, 0, 255, 255, 0]):
                images = {
                    k: np.zeros((240, 320, 3), np.uint8)
                    for k in ("front_rgb", "left_rgb", "right_rgb")
                }
                images["front_rgb"][:, :, 0] = red
                refs, images = store.publish(images)
                obs = {
                    "episode_nonce": "replay",
                    "observation_id": f"obs-{step}",
                    "step_index": step,
                    "simulation_time_s": step / 15,
                    "cameras": refs,
                    "lifecycle": "reset" if step == 0 else "nominal",
                    "event_sequence": step,
                }
                if step == 0:
                    critic.reset(obs, images)
                else:
                    results.append(critic.observe(obs, images).model_dump())
        finally:
            critic.close()
        (args.output / "replay.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {"fired_steps": [r["step_index"] for r in results if r["events"]]}
            )
        )
        return
    from robots.arx.gateway.backend import DirectBackend
    from robots.arx.gateway.recording import export_public_recording

    backend = DirectBackend(
        scene=args.scene,
        mapping=args.scene / "mapping.json",
        task=args.scene / "task.yaml",
        seed=17,
    )
    core = None

    # Same deterministic offline-test targets as run_arx_camera_chunk.py.
    def predict(observation):
        targets = np.repeat(observation.state[None], 32, axis=0)
        phase = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        for channel, delta in (
            (7, 0.45),
            (8, -0.40),
            (9, 0.35),
            (10, 0.30),
            (11, -0.25),
            (12, 0.30),
            (13, 0.60),
        ):
            targets[:, channel] += phase * delta
        return targets

    zeva = ZevaPlanner(predict, lambda: core.policy_observation(), execution_steps=16)
    journal = Journal(args.output / "journal.sqlite3")
    core = ArxSessionCore(
        episode_id="critic-smoke",
        backend=backend,
        registry=default_registry(zeva=zeva),
        journal=journal,
        output=args.output,
        limits=RuntimeLimits(
            max_steps=32,
            max_decisions=10,
            max_recoveries=1,
            operation_timeout_s=60.0,
            critic_timeout_s=5.0,
            idle_agent_timeout_s=60.0,
            lease_timeout_s=60.0,
            shutdown_timeout_s=5.0,
        ),
        critic=critic,
        bindings=package.bindings,
        package_sha256=package.sha256,
    )
    results = []
    try:
        core.reset()
        for index, (tool, arguments) in enumerate(
            [
                ("arx.zeva", {"max_chunks": 1}),
                ("arx.finish", {"reason": "Critic functional smoke completed"}),
            ]
        ):
            request = ToolRequest(
                request_id=f"test-{index}",
                decision_ref=f"decision-{index}",
                observation_id=core.current["observation_id"],
                control_epoch=core.epoch,
                tool=tool,
                arguments=arguments,
            )
            journal.register_decision(
                request, source="runner", evidence=core.snapshot()
            )
            results.append(
                {
                    "request": request.model_dump(),
                    "result": core.execute(request),
                    "snapshot": core.snapshot(),
                }
            )
    finally:
        core.close()
        export_public_recording(journal, args.output)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        journal.close()
    print(json.dumps(results[0]["result"]))


if __name__ == "__main__":
    main()
