#!/usr/bin/env python3
"""Build a dodo PickTube hardware config from pinned cameras and raw feedback.

Joint bounds are a conservative envelope of executed PickTube recordings,
not the robot's mechanical hard stops. The real gateway additionally checks
the live task start state and every command against these bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.gateway.motion import Calibration
from robots.arx.gateway.real_config import (
    RealHardwareConfig, validate_real_hardware_config,
)
from zetta.evolution.jsonio import file_sha256


def freeze(raw_root: Path, output: Path, provenance_path: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    calibration_path = root / "robots/arx/manifests/real/dodo_right_controller_ee_fk.json"
    task_path = root / "robots/arx/manifests/pickup_test_tube.yaml"
    model_path = root / "robots/arx/manifests/task7_model_a.yaml"
    calibration = Calibration.model_validate_json(calibration_path.read_text())
    files = sorted(raw_root.glob("*/data.npz"))
    if len(files) < 50:
        raise ValueError("50 recorded PickTube episodes required for task envelope")
    samples = []
    for path in files:
        states = np.load(path)["observation.state"]
        if states.ndim != 2 or states.shape[1] != 54 or not np.isfinite(states).all():
            raise ValueError(f"invalid raw feedback: {path}")
        samples.append(states[:, :14])
    states = np.concatenate(samples)
    left_low = np.min(states[:, :6], axis=0) - 0.05
    left_high = np.max(states[:, :6], axis=0) + 0.05
    right_low = [link.limits[0] for link in calibration.links]
    right_high = [link.limits[1] for link in calibration.links]
    camera_settings = []
    for name, serial, depth in (
        ("front_rgb", "260422272500", True),
        ("left_rgb", "260422271945", False),
        ("right_rgb", "260422275847", False),
    ):
        path = root / f"robots/arx/manifests/real/dodo_{name}_d405_intrinsics.json"
        camera_settings.append({
            "name": name, "serial": serial,
            "calibration_id": f"arx-task7-{name.removesuffix('_rgb')}-v1",
            "calibration_file": str(path), "calibration_sha256": file_sha256(path),
            "width": 320, "height": 240,
            "capture_width": 640, "capture_height": 480, "capture_fps": 15,
            "depth_enabled": depth,
        })
    def arm(port: str, low, high, *, native_max: float = 0.0):
        return {
            "can_port": port, "arm_type": 2,
            "joint_min_rad": [float(x) for x in low],
            "joint_max_rad": [float(x) for x in high],
            "gripper_native_min": -3.45, "gripper_native_max": native_max,
            "gripper_policy_scale": 1.0, "gripper_policy_offset": 0.0,
        }
    payload = {
        "schema_version": "arx.real.hardware.v1",
        "arm_transport": "arx_ros2", "camera_transport": "realsense",
        "left": arm("can1", left_low, left_high),
        "right": arm("can3", right_low, right_high, native_max=0.95),
        "command_left": False, "command_right": True,
        "cameras": camera_settings,
        "timing": {
            "control_hz": 15.0, "max_sensor_skew_ms": 100.0,
            "max_sensor_age_ms": 150.0, "observation_timeout_s": 5.0,
            "arrival_timeout_s": 3.0, "feedback_poll_s": 0.02,
            "position_tolerance": [0.05] * 6 + [0.1] + [0.05] * 6 + [0.1],
        },
        "right_gripper_closed_policy": 0.0,
        "right_gripper_open_policy": -3.4,
        "ros2_topics": {
            "left_status": "/arm_slave_l_status",
            "right_status": "/arm_slave_r_status",
            "left_command": "/arm_master_l_status",
            "right_command": "/arm_master_r_status",
        },
    }
    config = RealHardwareConfig.model_validate_json(json.dumps(payload))
    validate_real_hardware_config(config, task_path, model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(config.model_dump_json(indent=2) + "\n")
    provenance = {
        "schema_version": "arx.real.hardware.provenance.v1",
        "hardware_config_sha256": file_sha256(output),
        "right_fk_sha256": file_sha256(calibration_path),
        "task_manifest_sha256": file_sha256(task_path),
        "model_contract_sha256": file_sha256(model_path),
        "raw_episode_count": len(files),
        "raw_frame_count": len(states),
        "raw_data_sha256": {path.parent.name: file_sha256(path) for path in files},
        "joint_limit_scope": "recorded PickTube controller feedback envelope plus 0.05 rad; not mechanical hard stops",
        "gripper_scope": "feedback remains controller native; PickTube right actions add +0.9 native for firmer grip as frozen in the task manifest; right command range extends to +0.95",
        "motion_tested": False,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"hardware_config": str(output),
            "hardware_config_sha256": provenance["hardware_config_sha256"],
            "provenance": str(provenance_path),
            "raw_frames": len(states)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/home/dodo/chenfu/data/raw/PickTube"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.raw_root, args.output, args.provenance), sort_keys=True))


if __name__ == "__main__":
    main()
