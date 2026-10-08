#!/usr/bin/env python3
"""Build a self-depth reference from retained empty, open, near-home wrist frames.

Offline only. This calibrates observed self pixels at one encoder opening, not
a jaw-width law or a new collision hull. Inspect the output overlay before use.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
from robots.arx.deployment.wrist_self_mask import protected_colour
from robots.arx.gateway.gripper_geometry import ArxGripperGeometry
from robots.arx.gateway.real_config import load_real_hardware_config


def calibrate(sources, hardware_path, hardware_sha, output):
    import cv2

    provider = PickTubeRgbdProvider()
    provider.validate_hardware(load_real_hardware_config(hardware_path, hardware_sha))
    root = Path(__file__).resolve().parents[2]
    cad = root / "robots/arx/manifests/real/ac_one_gripper_hulls.json"
    geometry = ArxGripperGeometry(cad, hashlib.sha256(cad.read_bytes()).hexdigest())
    value = json.loads(sources.read_text())
    if value.get("robot_commands_sent") is not False or len(value["observations"]) < 8:
        raise ValueError("reference requires at least eight read-only empty home samples")
    depths, rgbs, grips, evidence = [], [], [], []
    stamps = []
    for row in value["observations"]:
        path = Path(row["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("self reference source SHA differs")
        h = row["observation"]["hardware"]
        state = np.asarray(h["measured_state"])
        if (np.max(np.abs(state[7:13])) > .15 or state[13] > -2.0
                or h["sensor_age_ms"] > 150 or h["sensor_skew_ms"] > 100
                or not h["device_health"]["transport_responsive"]
                or h["device_health"]["fault_codes"]):
            raise ValueError("self reference requires healthy measured near-home open jaws")
        with np.load(path, allow_pickle=False) as images:
            features = provider.observe(row["observation"], dict(images))
            if (features["privileged.selected.target_gripper_distance_m"] < .25
                    or features["privileged.interaction.gripper_contact"]
                    or features["privileged.interaction.grasped"]):
                raise ValueError("self reference target is not separated from empty jaws")
            depths.append(images["right_depth_mm"].copy())
            rgbs.append(images["right_rgb"].copy())
        grips.append(state[13]); stamps.append(h["depth_monotonic_ns"]["right_depth_mm"])
        evidence.append({"snapshot_sha256": row["sha256"], "observation_id": row["observation"]["observation_id"]})
    if len(set(stamps)) != len(stamps) or np.ptp(grips) > .03:
        raise ValueError("self reference depth frames repeat or opening is unstable")
    array = np.asarray(depths)
    valid = (array >= 70) & (array <= 200)
    count = valid.sum(axis=0)
    minimum = np.where(valid,array,10000).min(axis=0)
    maximum = np.where(valid,array,0).max(axis=0)
    stable = (count >= 3) & (maximum-minimum <= 5)
    median = np.zeros(array.shape[1:])
    median[stable] = np.nanmedian(np.where(valid,array,np.nan)[:,stable], axis=0)
    y, x = np.nonzero(stable)
    points = np.array([provider._deproject_intrinsics(float(px), float(py), float(median[py, px])/1000,
                                                    provider.wrist_intrinsics) for px, py in zip(x, y)])
    local = (provider.wrist_mount @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    pose = np.eye(4); pose[:3, 3] = geometry.tcp_offset
    inside = geometry.occupied_mask(local, pose, margin_m=.03)
    # Reference colour rejects a table/rack surface even in a CAD neighbourhood.
    for rgb in rgbs:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        inside &= ~protected_colour(rgb)[y,x] & (hsv[y,x,2] <= 100)
    x, y = x[inside], y[inside]
    if len(x) < 128:
        raise ValueError("insufficient stable observed gripper self pixels")
    v = {"schema_version":"arx.wrist.empty-self.v1", "empty_home_observed":True,
         "camera_serial":"260422275847", "intrinsics_sha256":h["camera_calibration_sha256"]["right_rgb"],
         "wrist_mount_sha256":provider.wrist_mount_sha, "cad_sha256":geometry.sha256,
         "hardware_sha256":hardware_sha, "image_shape":list(median.shape),
         "right_gripper_feedback":float(np.median(grips)), "gripper_feedback_tolerance":.08,
         "depth_tolerance_mm":5, "pixels_xy_depth_mm":np.c_[x,y,median[y,x]].tolist(),
         "source_samples":evidence, "source_manifest_sha256":hashlib.sha256(sources.read_bytes()).hexdigest(),
         "limits":["One empty encoder opening only; no interpolation or jaw-width calibration.",
                   "Current depth must match each pixel; coloured scene and target remain protected.",
                   "CAD swept collision checks are unchanged."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(v, indent=2)+"\n")
    rgb = rgbs[-1].copy(); rgb[y,x] = [0,255,255]
    cv2.imwrite(str(output.with_suffix(".png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return {"self_pixels":len(x), "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "robot_commands_sent":False, "opening_feedback":v["right_gripper_feedback"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--hardware-config", type=Path, required=True)
    p.add_argument("--hardware-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(calibrate(a.sources, a.hardware_config, a.hardware_sha256, a.output)))
