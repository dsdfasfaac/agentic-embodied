#!/usr/bin/env python3
"""Read-only one-shot PickTube RGB-D/TCP distance probe on dodo.

Subscribe to right-arm status and open only the front D405. No arm command
publisher is created. Put the pink tube in the fixed camera view first.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from arx5_arm_msg.msg import RobotStatus

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.picktube_rgbd_provider import (
    FRONT_INTRINSICS_SHA256, FRONT_SERIAL, PickTubeRgbdProvider,
)


def _check_profile(profile, expected):
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    for key in ("fx", "fy", "ppx", "ppy"):
        if not math.isclose(float(getattr(intrinsics, key)), expected[key], abs_tol=1e-3):
            raise ValueError(f"live front D405 RGB intrinsic differs: {key}")
    if not np.allclose(intrinsics.coeffs, expected["coeffs"], atol=1e-5, rtol=0):
        raise ValueError("live front D405 distortion coefficients differ")
    if str(intrinsics.model).split(".")[-1] != expected["distortion_model"]:
        raise ValueError("live front D405 distortion model differs")


def probe(timeout_s: float = 5.0, max_skew_ms: float = 100.0) -> dict:
    root = Path(__file__).resolve().parents[2]
    calibration_file = root / "robots/arx/manifests/real/dodo_front_rgb_d405_intrinsics.json"
    observer = PickTubeRgbdProvider()
    observer.validate_hardware(SimpleNamespace(
        cameras=[SimpleNamespace(
            name="front_rgb", serial=FRONT_SERIAL, depth_enabled=True,
            capture_width=640, capture_height=480, width=320, height=240,
            calibration_sha256=FRONT_INTRINSICS_SHA256,
            calibration_file=calibration_file,
        )],
        right_gripper_closed_policy=0.0, right_gripper_open_policy=-3.4,
    ))
    expected = json.loads(calibration_file.read_text())["camera"]["intrinsics"]
    rclpy.init()
    node = rclpy.create_node("zetta_picktube_distance_probe")
    latest = {}

    def on_status(message):
        stamp = time.monotonic_ns()
        pose = np.asarray(message.end_pos, dtype=np.float64)
        joints = np.asarray(message.joint_pos, dtype=np.float64)
        if pose.shape == (6,) and joints.shape == (7,) and np.isfinite(pose).all() and np.isfinite(joints).all():
            latest["sample"] = (pose[:3].copy(), stamp)

    node.create_subscription(RobotStatus, "/arm_slave_r_status", on_status, 10)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(FRONT_SERIAL)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    started = False
    try:
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            return {"status": "unavailable", "reason": f"front D405 cannot open: {exc}",
                    "front_serial": FRONT_SERIAL}
        started = True
        _check_profile(profile, expected)
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("live front D405 depth scale is invalid")
        align = rs.align(rs.stream.color)
        deadline = time.monotonic() + timeout_s
        last_error = "no synchronized front RGB-D/right TCP sample"
        while time.monotonic() < deadline:
            frames = align.process(pipeline.wait_for_frames(1000))
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            stamp = time.monotonic_ns()
            if not color or not depth:
                last_error = "front RGB-D frame is incomplete"
                continue
            sample = latest.get("sample")
            tcp, state_stamp = sample if sample is not None else (None, None)
            if state_stamp is None or abs(stamp - state_stamp) / 1e6 > max_skew_ms:
                last_error = "front RGB-D and right TCP are not synchronized"
                continue
            rgb = cv2.resize(np.asarray(color.get_data()), (320, 240),
                             interpolation=cv2.INTER_AREA)
            raw = np.asarray(depth.get_data())
            if raw.dtype != np.uint16 or raw.shape != (480, 640):
                raise ValueError("front D405 did not return z16 depth")
            depth_mm = cv2.resize(
                np.rint(raw.astype(np.float32) * (1000 * scale)).astype(np.uint16),
                (320, 240), interpolation=cv2.INTER_NEAREST,
            )
            try:
                distance = observer.measure_distance(rgb, depth_mm, tcp)
            except ValueError as exc:
                last_error = str(exc)
                continue
            return {
                "status": "measured", "distance_m": distance,
                "target_point": "pink_label_centre", "tcp_frame": "right_arm_local_base",
                "front_serial": FRONT_SERIAL, "depth_scale_m": scale,
                "sensor_skew_ms": abs(stamp - state_stamp) / 1e6,
                "host_monotonic_ns": stamp,
            }
        return {"status": "unavailable", "reason": last_error, "front_serial": FRONT_SERIAL}
    finally:
        if started:
            pipeline.stop()
        rclpy.shutdown()
        spin_thread.join(timeout=1)
        node.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--max-skew-ms", type=float, default=100.0)
    args = parser.parse_args()
    if not (0 < args.timeout_s <= 60 and 0 < args.max_skew_ms <= 1000):
        parser.error("timeout and skew must be positive and bounded")
    result = probe(args.timeout_s, args.max_skew_ms)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "measured":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
