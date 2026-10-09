#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Serve one ARX real-robot episode through the existing gateway."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.contracts import load_task_manifest
from robots.arx.gateway.contracts import RuntimeLimits
from robots.arx.gateway.real_config import (
    load_real_hardware_config,
    validate_real_hardware_config,
)


from robots.arx.gateway.real_factory import RealCoreFactory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hardware-config", "task", "model-contract"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-hardware-sha256", required=True)
    parser.add_argument("--kinematics-calibration", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--tool-catalog", type=Path)
    parser.add_argument("--real-input-contract", type=Path)
    parser.add_argument("--expected-real-input-sha256")
    parser.add_argument("--feature-provider", type=Path)
    parser.add_argument("--expected-feature-provider-sha256")
    parser.add_argument("--grasp-config", type=Path)
    parser.add_argument("--expected-grasp-config-sha256")
    parser.add_argument("--zeva-host", default="127.0.0.1")
    parser.add_argument("--zeva-port", type=int, default=5581)
    parser.add_argument(
        "--listen-host", choices=("127.0.0.1", "::1"), default="127.0.0.1"
    )
    parser.add_argument("--listen-port", type=int, default=8091)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()

    config = load_real_hardware_config(
        args.hardware_config, args.expected_hardware_sha256
    )
    validate_real_hardware_config(config, args.task, args.model_contract)
    if args.check_config:
        if args.bundle:
            if args.runtime_config is None:
                parser.error("bundle --check-config requires --runtime-config")
            limits = RuntimeLimits.model_validate_json(args.runtime_config.read_text())
            report = RealCoreFactory(
                str(args.hardware_config),
                args.expected_hardware_sha256,
                str(args.task),
                str(args.model_contract),
                "/tmp",
                "preflight",
                limits.model_dump(),
                args.zeva_host,
                args.zeva_port,
                (
                    str(args.kinematics_calibration)
                    if args.kinematics_calibration
                    else None
                ),
                str(args.bundle),
                str(args.tool_catalog) if args.tool_catalog else None,
                str(args.real_input_contract) if args.real_input_contract else None,
                args.expected_real_input_sha256,
                str(args.feature_provider) if args.feature_provider else None,
                args.expected_feature_provider_sha256,
                True,
                str(args.grasp_config) if args.grasp_config else None,
                args.expected_grasp_config_sha256,
            )(lambda: False, lambda _: None)
            print(json.dumps(dict(report, hardware_opened=False), sort_keys=True))
            return
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": config.schema_version,
                    "arm_transport": config.arm_transport,
                    "camera_serials": {
                        camera.name: camera.serial for camera in config.cameras
                    },
                    "control_hz": config.timing.control_hz,
                    "hardware_opened": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.runtime_config is None or args.output is None:
        parser.error("--runtime-config and --output are required to serve hardware")
    task = load_task_manifest(args.task)
    limits = RuntimeLimits.model_validate_json(args.runtime_config.read_text())
    if limits.max_steps > task.max_steps:
        parser.error("gateway max_steps exceeds frozen task limit")
    from scripts.deployment.prepare_arx_trial_storage import admit_gateway_output
    admit_gateway_output(args.output)
    os.chmod(args.output, 0o700)
    episode_id = uuid.uuid4().hex
    agent, harness = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    credentials = args.output / "capabilities.json"
    with credentials.open("x") as stream:
        os.chmod(credentials, 0o600)
        json.dump(
            {
                "episode_id": episode_id,
                "agent_capability": agent,
                "harness_capability": harness,
            },
            stream,
        )
    factory = RealCoreFactory(
        str(args.hardware_config),
        args.expected_hardware_sha256,
        str(args.task),
        str(args.model_contract),
        str(args.output),
        episode_id,
        limits.model_dump(),
        args.zeva_host,
        args.zeva_port,
        str(args.kinematics_calibration) if args.kinematics_calibration else None,
        str(args.bundle) if args.bundle else None,
        str(args.tool_catalog) if args.tool_catalog else None,
        str(args.real_input_contract) if args.real_input_contract else None,
        args.expected_real_input_sha256,
        str(args.feature_provider) if args.feature_provider else None,
        args.expected_feature_provider_sha256,
        grasp_config=str(args.grasp_config) if args.grasp_config else None,
        expected_grasp_config_sha256=args.expected_grasp_config_sha256,
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
            host=args.listen_host,
            port=args.listen_port,
            access_log=False,
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
