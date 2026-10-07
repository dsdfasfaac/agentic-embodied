#!/usr/bin/env python3
"""Fit a scoped wrist target registration from retained synchronized RGB-D/FK.

No hardware is opened or commanded. A stationary pink label seen by both
cameras supplies paired points; a separate temporal pose segment is held out.
This is not a full six-DOF checkerboard hand-eye calibration.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.picktube_rgbd_provider import (
    PickTubeRgbdProvider, CONTROLLER_FK_SHA256, RIGHT_INTRINSICS_SHA256, RIGHT_SERIAL,
    FRONT_INTRINSICS_SHA256, FRONT_SERIAL,
)
from robots.arx.gateway.motion import CommandKinematics


def fit_rigid(camera_points, body_points):
    a, b = np.asarray(camera_points), np.asarray(body_points)
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = b.mean(0) - rotation @ a.mean(0)
    return transform, np.linalg.svd(ac, compute_uv=False)


def calibrate(gateway, *, fit_max_step=60):
    provider, tracker = PickTubeRgbdProvider(), PickTubeRgbdProvider()
    kin = CommandKinematics(provider.controller_fk.calibration.model_copy(update={"tcp_offset": [0., 0., 0.]}))
    intrinsic_file = Path(__file__).resolve().parents[2] / "robots/arx/manifests/real/dodo_right_rgb_d405_intrinsics.json"
    if hashlib.sha256(intrinsic_file.read_bytes()).hexdigest() != RIGHT_INTRINSICS_SHA256:
        raise ValueError("right RGB intrinsics SHA differs")
    intrinsics = json.loads(intrinsic_file.read_text())["camera"]["intrinsics"]
    pairs, rejected, seen = [], [], set()
    with sqlite3.connect(f"file:{gateway / 'journal.sqlite3'}?mode=ro", uri=True) as db:
        records = db.execute("select payload from records where kind=? order by sequence", ("grasp_sensor_evidence",))
        for row in records:
            item = json.loads(row[0]); obs = item["observation"]; hardware = obs["hardware"]
            stamp = hardware["depth_monotonic_ns"].get("right_depth_mm")
            if stamp in seen:
                continue
            seen.add(stamp)
            path = gateway / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("sensor snapshot SHA differs")
            state = np.asarray(hardware["measured_state"])
            if (state.shape != (14,) or not np.isfinite(state).all()
                    or hardware.get("right_tcp_frame") != "right_arm_local_base"
                    or hardware.get("right_tcp_monotonic_ns") != hardware["state_monotonic_ns"]):
                raise ValueError("fresh audited controller FK is required")
            ee, _, _ = provider.controller_fk.fk(state[7:13])
            if np.linalg.norm(ee - np.asarray(hardware["right_tcp_xyz_m"])) > .01:
                raise ValueError("controller FK differs from joint sample")
            for camera, serial, sha in (("front_rgb", FRONT_SERIAL, FRONT_INTRINSICS_SHA256),
                                         ("right_rgb", RIGHT_SERIAL, RIGHT_INTRINSICS_SHA256)):
                if (hardware.get("camera_health", {}).get(camera, {}).get("device_id") != serial
                        or hardware.get("camera_health", {}).get(camera, {}).get("responsive") is not True
                        or hardware.get("camera_calibration_sha256", {}).get(camera) != sha):
                    raise ValueError("calibration camera identity differs")
            if (abs(state[13]) < 2. or hardware.get("sensor_skew_ms", float('inf')) > 100
                    or hardware.get("sensor_age_ms", float('inf')) > 150
                    or stamp != hardware["camera_monotonic_ns"]["right_rgb"]):
                raise ValueError("calibration requires an open gripper and synchronized RGB-D/FK")
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    provider.measure_distance(arrays["front_rgb"], arrays["front_depth_mm"], np.zeros(3))
                    target = provider.last_target_left + [0., .5, 0.]
                    mask = tracker._pink_component(arrays["right_rgb"])
                    depth = arrays["right_depth_mm"]
                    if depth.dtype != np.uint16 or depth.shape != (240,320):
                        raise ValueError("right depth must be uint16 millimetres")
                    valid = mask & (depth >= 80) & (depth <= 1500)
                    yy, xx = np.nonzero(valid)
                    support = depth[valid].astype(float)
                    if len(xx) < 12 or np.median(abs(support - np.median(support))) > 25:
                        raise ValueError("right label depth insufficient or inconsistent")
                    point = provider._deproject_intrinsics(float(np.median(xx)), float(np.median(yy)),
                                                          float(np.median(support))/1000., intrinsics)
                p, rotation, _ = kin.fk(state[7:13])
                pairs.append({"step": obs["step_index"], "observation_id": obs["observation_id"],
                              "snapshot_sha256": item["sha256"], "camera_m": point.tolist(),
                              "link6_m": (rotation.T @ (target - p)).tolist()})
            except ValueError as exc:
                rejected.append({"observation_id": obs["observation_id"], "reason": str(exc)})
    training = [p for p in pairs if p["step"] <= fit_max_step]
    heldout = [p for p in pairs if p["step"] > fit_max_step]
    if len(training) < 20 or len(heldout) < 20:
        raise ValueError("at least twenty fit and twenty held-out fresh samples are required")
    transform, spread = fit_rigid([p["camera_m"] for p in training], [p["link6_m"] for p in training])
    for pair in pairs:
        pair["residual_m"] = float(np.linalg.norm((transform @ np.r_[pair["camera_m"],1.])[:3] - pair["link6_m"]))
    rms = float(np.sqrt(np.mean([p["residual_m"]**2 for p in heldout])))
    maximum = max(p["residual_m"] for p in heldout)
    return {"schema_version":"arx.real.wrist_mount.v1", "camera_serial":RIGHT_SERIAL,
            "parent_frame":"right_link6_after_rotation", "controller_fk_sha256":CONTROLLER_FK_SHA256,
            "rgb_intrinsics_sha256":RIGHT_INTRINSICS_SHA256, "T_link6_from_camera":transform.tolist(),
            "quality":{"accepted": bool(rms <= .008 and maximum <= .015 and spread[-1] >= .005
                                         and np.linalg.norm(transform[:3,3]) <= .25),
                       "scope":"stationary pink-label registration during open-gripper pregrasp; not a board hand-eye calibration",
                       "fit_max_step":fit_max_step, "fit_samples":len(training), "heldout_samples":len(heldout),
                       "heldout_rms_m":rms, "heldout_max_m":maximum,
                       "fit_spread_singular_values_m":spread.tolist()},
            "gateway":str(gateway.resolve()), "robot_commands_sent":False,
            "samples":pairs, "rejected":rejected}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', type=Path, required=True)
    parser.add_argument('--fit-max-step', type=int, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); result=calibrate(args.gateway, fit_max_step=args.fit_max_step)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['quality']))


if __name__ == '__main__':
    main()
