#!/usr/bin/env python3
"""Read one synchronized dodo ARX/RGB-D observation without publishing motion.

The ROS2 status controllers must already be running. The adapter constructs
command publishers but this script never calls ArmDevice.send or Backend.step.
It checks the frozen task start state and computes the bundle's two features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.contracts import load_task_manifest
from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
from robots.arx.gateway.real_config import (
    build_real_backend, load_real_hardware_config,
)


def audit(hardware_path: Path, hardware_sha: str, task_path: Path,
          model_path: Path) -> dict:
    config = load_real_hardware_config(hardware_path, hardware_sha)
    task = load_task_manifest(task_path)
    provider = PickTubeRgbdProvider()
    provider.validate_hardware(config)
    backend = build_real_backend(config=config, task_path=task_path,
                                 model_path=model_path)
    try:
        policy, evidence, feature_frames, _ = backend._observe()
        features = provider.observe(
            {"hardware": evidence}, {**policy.images, **feature_frames},
        )
        error = np.asarray(policy.state, dtype=float) - np.asarray(task.start_state)
        tolerance = np.asarray(config.timing.position_tolerance)
        return {
            "schema_version": "arx.live.observation.audit.v1",
            "status": "observed",
            "robot_commands_sent": False,
            "measured_state": policy.state.tolist(),
            "task_start_error": error.tolist(),
            "task_start_eligible": bool(np.all(np.abs(error) <= tolerance)),
            "features": features,
            "sensor_age_ms": evidence["sensor_age_ms"],
            "sensor_skew_ms": evidence["sensor_skew_ms"],
            "device_health": evidence["device_health"],
            "right_controller_ee_xyz_m": evidence.get("right_tcp_xyz_m"),
        }
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--hardware-config", type=Path, default=root /
                        "robots/arx/manifests/real/dodo_picktube_hardware.json")
    parser.add_argument("--hardware-sha256", required=True)
    parser.add_argument("--task", type=Path, default=root /
                        "robots/arx/manifests/pickup_test_tube.yaml")
    parser.add_argument("--model-contract", type=Path, default=root /
                        "robots/arx/manifests/task7_model_a.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.hardware_config, args.hardware_sha256,
                       args.task, args.model_contract)
    except Exception as exc:
        report = {
            "schema_version": "arx.live.observation.audit.v1",
            "status": "unavailable", "reason": str(exc),
            "robot_commands_sent": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "observed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
