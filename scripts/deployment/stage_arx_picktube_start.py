#!/usr/bin/env python3
"""Move both empty ARX grippers and joints toward the frozen PickTube start.

This is a bounded, measured staging command, not a VLA policy. With
--execute-steps 0 it only reads status and writes a plan. Use one step first;
the next invocation replans from the newly measured state. No command is
sent if status, units, bounds, or timing are invalid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.contracts import load_task_manifest
from robots.arx.gateway.arx_ros2_device import ArxRos2Device, Ros2Topics
from robots.arx.gateway.real_config import load_real_hardware_config


def _read_fresh(device: ArxRos2Device, deadline: float):
    last_error = "no joint feedback"
    while time.monotonic() < deadline:
        try:
            return device.read()
        except TimeoutError as exc:
            last_error = str(exc)
            time.sleep(0.01)
    raise TimeoutError(last_error)


def stage(hardware_path: Path, hardware_sha: str, task_path: Path,
          execute_steps: int, output: Path, max_joint_step_rad: float = 0.015) -> dict:
    if not math.isfinite(max_joint_step_rad) or not 0 < max_joint_step_rad <= 0.035:
        raise ValueError("joint staging step must be finite and in (0, 0.035] rad")
    config = load_real_hardware_config(hardware_path, hardware_sha)
    task = load_task_manifest(task_path)
    if config.arm_transport != "arx_ros2" or config.timing.control_hz != 15:
        raise ValueError("staging requires the frozen dodo ROS2 15 Hz hardware")
    device = ArxRos2Device.from_ros2(
        topics=Ros2Topics(**config.ros2_topics),
        left_calibration=config.left.calibration(),
        right_calibration=config.right.calibration(),
        command_left=True, command_right=True,
        max_status_age_ms=150, max_pair_skew_ms=50,
        keepalive_hz=60, command_lease_s=1.5,
    )
    report = {
        "schema_version": "arx.picktube.start.stage.v1",
        "hardware_sha256": hardware_sha,
        "task": task.name,
        "requested_execute_steps": execute_steps,
        "joint_step_limit_rad": max_joint_step_rad,
        "command_log": [],
        "status": "initializing",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        initial = _read_fresh(device, time.monotonic() + 5)
        current = np.asarray(initial.positions, dtype=np.float64)
        goal = np.asarray(task.start_state, dtype=np.float64).copy()
        goal[[6, 13]] += np.asarray(task.control.gripper_command_offsets)
        for side, calibration, chunk in (
            ("left", config.left.calibration(), goal[:7]),
            ("right", config.right.calibration(), goal[7:]),
        ):
            device._native_target(chunk, calibration)
        delta = goal - current
        arm_steps = np.abs(delta[[i for i in range(14) if i not in (6, 13)]]) / max_joint_step_rad
        gripper_steps = np.abs(delta[[6, 13]]) / 0.07
        planned = max(1, math.ceil(float(max(np.max(arm_steps), np.max(gripper_steps)))))
        if planned > 150:
            raise ValueError(f"start-state staging requires {planned} steps, exceeds 150")
        report.update({
            "initial_state": current.tolist(), "goal_state": goal.tolist(),
            "model_start_state": list(task.start_state),
            "gripper_command_offsets": list(task.control.gripper_command_offsets),
            "planned_steps": planned,
            "max_joint_step_rad": float(np.max(np.abs(delta[[i for i in range(14) if i not in (6, 13)]])) / planned),
            "max_gripper_step": float(np.max(np.abs(delta[[6, 13]])) / planned),
            "status": "planned",
        })
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if execute_steps == 0:
            return report
        start_tolerance = np.asarray(config.timing.position_tolerance)
        # Use the same measured arrival tolerance as the deployed backend.
        # Progress checks below still reject unchanged feedback, and every
        # subsequent target is bounded from the latest measurement.
        tolerance = start_tolerance
        for index in range(1, execute_steps + 1):
            if np.all(np.abs(current - goal) <= start_tolerance):
                break
            # Replan from measured feedback each time. A fixed interpolation
            # accumulates gripper tracking lag even when every step passes.
            step = np.clip(goal - current, -max_joint_step_rad, max_joint_step_rad)
            step[[6, 13]] = np.clip((goal - current)[[6, 13]], -0.07, 0.07)
            target = current + step
            active_joints = [axis for axis in range(14) if axis not in (6, 13)
                             and abs(goal[axis] - current[axis]) > start_tolerance[axis]
                             and abs(step[axis]) >= 0.01]
            command_id = "stage-" + uuid.uuid4().hex
            receipt = device.send(target.astype(np.float32), command_id)
            deadline = time.monotonic() + 2.0
            last = None
            while time.monotonic() < deadline:
                sample = _read_fresh(device, deadline)
                if sample.monotonic_ns <= receipt.sent_monotonic_ns:
                    time.sleep(0.01)
                    continue
                last = sample
                tracking_ok = np.all(np.abs(sample.positions - target) <= tolerance)
                grip_progress = all(
                    abs(sample.positions[grip] - goal[grip]) <= start_tolerance[grip]
                    or abs(sample.positions[grip] - current[grip]) >= 0.015
                    for grip in (6, 13) if abs(step[grip]) >= 0.05
                )
                joint_progress = all(
                    abs(sample.positions[axis] - goal[axis]) <= start_tolerance[axis]
                    or abs(sample.positions[axis] - current[axis]) >= 0.002
                    for axis in active_joints
                )
                if tracking_ok and grip_progress and joint_progress:
                    break
                time.sleep(0.02)
            arrived = last is not None and bool(
                np.all(np.abs(last.positions - target) <= tolerance)
                and all(abs(last.positions[grip] - goal[grip]) <= start_tolerance[grip]
                        or abs(last.positions[grip] - current[grip]) >= 0.015
                        for grip in (6, 13) if abs(step[grip]) >= 0.05)
                and all(abs(last.positions[axis] - goal[axis]) <= start_tolerance[axis]
                        or abs(last.positions[axis] - current[axis]) >= 0.002
                        for axis in active_joints)
            )
            report["command_log"].append({
                "index": index, "target": target.tolist(),
                "receipt_status": receipt.status,
                "sent_monotonic_ns": receipt.sent_monotonic_ns,
                "measured_state": None if last is None else last.positions.tolist(),
                "measured_monotonic_ns": None if last is None else last.monotonic_ns,
                "arrival_verified": arrived,
                "max_tracking_error": None if last is None else float(
                    np.max(np.abs(last.positions - target))),
                "gripper_progress": None if last is None else [
                    float(abs(last.positions[grip] - current[grip])) for grip in (6, 13)
                ],
            })
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            if not arrived:
                raise TimeoutError(f"stage step {index} did not reach its bounded target")
            current = np.asarray(last.positions, dtype=np.float64)
        final = _read_fresh(device, time.monotonic() + 1)
        report["final_state"] = final.positions.tolist()
        report["status"] = "complete" if np.all(
            np.abs(final.positions - goal) <= start_tolerance
        ) else "partial"
        return report
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        device.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--hardware-config", type=Path, default=root /
                        "robots/arx/manifests/real/dodo_picktube_hardware.json")
    parser.add_argument("--hardware-sha256", required=True)
    parser.add_argument("--task", type=Path, default=root /
                        "robots/arx/manifests/pickup_test_tube.yaml")
    parser.add_argument("--execute-steps", type=int, default=0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.015,
                        help="Measured staging increment; at most the deployed 0.035 rad VLA limit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.execute_steps <= 150:
        parser.error("execute-steps must be 0..150")
    try:
        result = stage(args.hardware_config, args.hardware_sha256,
                       args.task, args.execute_steps, args.output, args.max_joint_step_rad)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc),
                          "report": str(args.output)}, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps({"status": result["status"],
                      "planned_steps": result["planned_steps"],
                      "executed_steps": len(result["command_log"]),
                      "report": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
