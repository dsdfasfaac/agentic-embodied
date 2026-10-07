#!/usr/bin/env python3
"""Read-only dodo D405 identity, synchronization, and PickTube depth audit.

Opens only the three RealSense cameras. It does not import ROS, open CAN, or
create a robot command publisher. A TCP distance requires separate fresh arm
feedback and is deliberately not inferred from camera data alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.picktube_rgbd_provider import (
    CALIBRATION_PATH, CALIBRATION_SHA256, PickTubeRgbdProvider,
)
from robots.arx.gateway.real_backend import CameraIdentity
from robots.arx.gateway.real_camera import RealSenseCameraSource, RealSenseCameraSpec


CAMERAS = (
    ("front_rgb", "260422272500", True),
    ("left_rgb", "260422271945", False),
    ("right_rgb", "260422275847", False),
)


def audit(samples: int, image_dir: Path | None) -> dict:
    root = Path(__file__).resolve().parents[2] / "robots/arx/manifests/real"
    specs = []
    identities = []
    for name, serial, depth in CAMERAS:
        calibration = root / f"dodo_{name}_d405_intrinsics.json"
        sha = hashlib.sha256(calibration.read_bytes()).hexdigest()
        identity = CameraIdentity(
            name, serial, f"arx-task7-{name.removesuffix('_rgb')}-v1",
            sha, 320, 240,
        )
        specs.append(RealSenseCameraSpec(
            identity, serial, calibration, 640, 480, 15, depth,
        ))
        identities.append({
            "name": name, "serial": serial, "rgb_intrinsics_sha256": sha,
            "capture": "640x480@15", "policy_rgb": "320x240", "aligned_depth": depth,
        })
    provider = PickTubeRgbdProvider()
    if hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest() != CALIBRATION_SHA256:
        raise ValueError("front extrinsic SHA-256 changed")
    result = {
        "schema_version": "arx.live.camera.audit.v1",
        "clock": "host_monotonic_ns",
        "cameras": identities,
        "front_extrinsic_sha256": CALIBRATION_SHA256,
        "observations": [],
        "robot_status_used": False,
        "robot_commands_sent": False,
    }
    source = RealSenseCameraSource(tuple(specs), max_skew_ms=100)
    last_stamps = {name: -1 for name, _, _ in CAMERAS}
    try:
        for index in range(samples):
            deadline = time.monotonic() + 5
            while True:
                frames = source.capture(max(0.001, deadline - time.monotonic()))
                stamps = {name: frame.monotonic_ns for name, frame in frames.items()}
                if all(stamps[name] > last_stamps[name] for name in last_stamps):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("fresh synchronized camera trio unavailable")
                time.sleep(0.01)
            last_stamps = stamps
            now = time.monotonic_ns()
            front = frames["front_rgb"]
            depth = np.asarray(front.depth_mm)
            observation = {
                "index": index,
                "host_monotonic_ns": stamps,
                "sensor_skew_ms": (max(stamps.values()) - min(stamps.values())) / 1e6,
                "sensor_age_ms": (now - min(stamps.values())) / 1e6,
                "front_valid_depth_fraction": float(np.count_nonzero(depth) / depth.size),
            }
            try:
                mask = provider._pink_component(front.pixels)
                yy, xx = np.nonzero(mask & (depth >= 80) & (depth <= 1500))
                valid = depth[yy, xx]
                if valid.size < 12:
                    raise ValueError("pink label has fewer than 12 valid depth pixels")
                median_mm = float(np.median(valid))
                mad_mm = float(np.median(np.abs(valid.astype(float) - median_mm)))
                if mad_mm > 25:
                    raise ValueError("pink label depth MAD exceeds 25 mm")
                pixel = [float(np.median(xx)), float(np.median(yy))]
                camera_xyz = provider._deproject(*pixel, median_mm / 1000)
                left_base_xyz = (provider.transform @ np.r_[camera_xyz, 1.0])[:3]
                observation["pink_label"] = {
                    "found": True, "pixel_320x240": pixel,
                    "valid_depth_pixels": int(valid.size),
                    "depth_median_mm": median_mm, "depth_mad_mm": mad_mm,
                    "point_in_left_base_m": left_base_xyz.tolist(),
                }
            except ValueError as exc:
                observation["pink_label"] = {"found": False, "reason": str(exc)}
            result["observations"].append(observation)
            if index == 0 and image_dir is not None:
                import cv2
                image_dir.mkdir(parents=True, exist_ok=True)
                for name, frame in frames.items():
                    if not cv2.imwrite(str(image_dir / f"{name}.png"),
                                       cv2.cvtColor(frame.pixels, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"could not retain {name}")
                if not cv2.imwrite(str(image_dir / "front_depth_mm.png"), depth):
                    raise OSError("could not retain front aligned depth")
                observation["saved_images"] = {
                    path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in [*(image_dir / f"{name}.png" for name in frames),
                                 image_dir / "front_depth_mm.png"]
                }
    finally:
        source.close()
    result["passed"] = (
        len(result["observations"]) == samples
        and all(o["sensor_age_ms"] <= 150 and o["sensor_skew_ms"] <= 100
                and o["pink_label"]["found"] for o in result["observations"])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.samples <= 50:
        parser.error("samples must be 1..50")
    report = audit(args.samples, args.image_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output),
                      "samples": len(report["observations"])}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
