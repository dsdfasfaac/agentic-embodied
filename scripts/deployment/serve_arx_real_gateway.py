#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Serve one ARX real-robot baseline episode through the existing gateway."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.contracts import load_model_contract, load_task_manifest
from robots.arx.gateway.contracts import RuntimeLimits
from robots.arx.gateway.real_config import (
    build_real_backend,
    load_real_hardware_config,
    validate_real_hardware_config,
)


@dataclass
class RealCoreFactory:
    hardware_config: str
    hardware_sha256: str
    task: str
    model_contract: str
    output: str
    episode_id: str
    limits: dict
    zeva_host: str
    zeva_port: int
    kinematics_calibration: str | None

    def __call__(self, cancelled, phase_changed):
        from robots.arx.gateway.journal import Journal
        from robots.arx.gateway.motion import Calibration, CommandKinematics
        from robots.arx.gateway.session_core import ArxSessionCore, BaselineMonitor
        from robots.arx.gateway.tools import (
            EefPlanner, PolicyGripperPlanner, default_registry,
        )
        from robots.arx.gateway.zeva import CosmosPredictor, ZevaPlanner

        config = load_real_hardware_config(
            Path(self.hardware_config), self.hardware_sha256
        )
        task_path, model_path = Path(self.task), Path(self.model_contract)
        validate_real_hardware_config(config, task_path, model_path)
        task, model = load_task_manifest(task_path), load_model_contract(model_path)
        limits = RuntimeLimits.model_validate(self.limits)
        backend = build_real_backend(
            config=config, task_path=task_path, model_path=model_path
        )
        try:
            core = None
            predictor = CosmosPredictor(
                host=self.zeva_host, port=self.zeva_port,
                contract=model, task=task,
                timeout_s=limits.operation_timeout_s,
            )
            zeva = ZevaPlanner(
                predictor, lambda: core.policy_observation(),
                execution_steps=task.execution_steps,
                action_horizon=model.action_horizon,
            )
            eef = None
            if self.kinematics_calibration is not None:
                calibration = Calibration.model_validate_json(
                    Path(self.kinematics_calibration).read_text()
                )
                eef = EefPlanner(CommandKinematics(calibration))
            registry = default_registry(
                zeva=zeva,
                gripper=PolicyGripperPlanner(
                    closed_policy=config.right_gripper_closed_policy,
                    open_policy=config.right_gripper_open_policy,
                ),
                eef=eef,
            )
            core = ArxSessionCore(
                episode_id=self.episode_id,
                backend=backend,
                registry=registry,
                journal=Journal(Path(self.output) / "journal.sqlite3"),
                output=Path(self.output),
                limits=limits,
                critic=BaselineMonitor(),
                cancel_requested=cancelled,
                phase_changed=phase_changed,
                privileged=False,
            )
            return core
        except BaseException:
            backend.close()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hardware-config", "task", "model-contract"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-hardware-sha256", required=True)
    parser.add_argument("--kinematics-calibration", type=Path)
    parser.add_argument("--zeva-host", default="127.0.0.1")
    parser.add_argument("--zeva-port", type=int, default=5581)
    parser.add_argument("--listen-host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8091)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()

    config = load_real_hardware_config(
        args.hardware_config, args.expected_hardware_sha256
    )
    validate_real_hardware_config(config, args.task, args.model_contract)
    if args.check_config:
        print(json.dumps({
            "valid": True, "schema_version": config.schema_version,
            "arm_transport": config.arm_transport,
            "camera_serials": {camera.name: camera.serial for camera in config.cameras},
            "control_hz": config.timing.control_hz,
            "hardware_opened": False,
        }, sort_keys=True))
        return
    if args.runtime_config is None or args.output is None:
        parser.error("--runtime-config and --output are required to serve hardware")
    task = load_task_manifest(args.task)
    limits = RuntimeLimits.model_validate_json(args.runtime_config.read_text())
    if limits.max_steps > task.max_steps:
        parser.error("gateway max_steps exceeds frozen task limit")
    args.output.mkdir(parents=True, exist_ok=False)
    os.chmod(args.output, 0o700)
    episode_id = uuid.uuid4().hex
    agent, harness = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    credentials = args.output / "capabilities.json"
    with credentials.open("x") as stream:
        os.chmod(credentials, 0o600)
        json.dump({
            "episode_id": episode_id,
            "agent_capability": agent,
            "harness_capability": harness,
        }, stream)
    factory = RealCoreFactory(
        str(args.hardware_config), args.expected_hardware_sha256,
        str(args.task), str(args.model_contract), str(args.output),
        episode_id, limits.model_dump(),
        args.zeva_host, args.zeva_port,
        str(args.kinematics_calibration) if args.kinematics_calibration else None,
    )
    import uvicorn
    from robots.arx.gateway.service import create_app
    from robots.arx.gateway.worker import EpisodeWorker

    worker = EpisodeWorker(
        factory=factory, output=args.output, episode_id=episode_id, limits=limits
    )
    try:
        uvicorn.run(
            create_app(worker, agent_capability=agent, harness_capability=harness),
            host=args.listen_host, port=args.listen_port, access_log=False,
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
