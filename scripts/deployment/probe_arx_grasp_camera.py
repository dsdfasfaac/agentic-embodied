#!/usr/bin/env python3
"""Infer target-only grasps from an audited camera sample, without arm feedback.

This is a proposal audit only. It cannot perform IK, certify a trajectory, issue
a motion token or send a command. Use probe_arx_grasp_snapshot for paired joints.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.deployment.picktube_rgbd_provider import (
    CALIBRATION_SHA256, FRONT_INTRINSICS_SHA256, FRONT_SERIAL, PickTubeRgbdProvider,
)
from robots.manipulation.grasp_proposals import LocalGraspService, TargetCloud, rigid_pose


def probe(audit_path, image_dir, endpoint, expected_model_sha256=None):
    url = urlsplit(endpoint)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username or url.password or url.query or url.fragment):
        raise ValueError("grasp inference must run on dodo loopback")
    audit = json.loads(audit_path.read_text())
    if (audit.get("schema_version") != "arx.live.camera.audit.v1"
            or audit.get("front_extrinsic_sha256") != CALIBRATION_SHA256):
        raise ValueError("expected an audited front D405 camera sample")
    sample = audit["observations"][0]
    if not sample["pink_label"]["found"] or sample["sensor_skew_ms"] > 100 or sample["sensor_age_ms"] > 150:
        raise ValueError("saved sample did not pass camera freshness/target checks")
    camera = next(c for c in audit["cameras"] if c["name"] == "front_rgb")
    if camera["serial"] != FRONT_SERIAL or camera["rgb_intrinsics_sha256"] != FRONT_INTRINSICS_SHA256:
        raise ValueError("front camera identity differs")
    for name in ("front_rgb.png", "front_depth_mm.png"):
        expected = sample["saved_images"][name]["sha256"]
        if hashlib.sha256((image_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("saved image SHA differs: " + name)
    rgb = cv2.cvtColor(cv2.imread(str(image_dir / "front_rgb.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(image_dir / "front_depth_mm.png"), cv2.IMREAD_UNCHANGED)
    if depth.dtype != np.uint16 or depth.shape != (240, 320):
        raise ValueError("expected aligned uint16 millimetre depth")
    provider = PickTubeRgbdProvider()
    calibration = Path(__file__).resolve().parents[2] / "robots/arx/manifests/real/dodo_front_rgb_d405_intrinsics.json"
    provider.validate_hardware(SimpleNamespace(cameras=[SimpleNamespace(
        name="front_rgb", serial=FRONT_SERIAL, depth_enabled=True,
        capture_width=640, capture_height=480, width=320, height=240,
        calibration_sha256=FRONT_INTRINSICS_SHA256, calibration_file=calibration)],
        right_gripper_closed_policy=0., right_gripper_open_policy=-3.4))
    mask = provider._pink_component(rgb)
    valid = (depth >= 80) & (depth <= 1500)
    yy, xx = np.nonzero(mask & valid)
    z = depth[yy, xx].astype(float)
    if len(xx) < 32 or np.median(np.abs(z - np.median(z))) > 25:
        raise ValueError("model requires at least 32 consistent pink-label depth pixels")
    points = np.asarray([provider._deproject(float(x), float(y), float(d) / 1000.)
                         for x, y, d in zip(xx, yy, z)])
    sy, sx = np.nonzero(valid[::2, ::2])
    scene = np.asarray([provider._deproject(float(x * 2), float(y * 2), float(depth[y * 2, x * 2]) / 1000.)
                        for x, y in zip(sx, sy)])
    target = provider._deproject(float(np.median(xx)), float(np.median(yy)), float(np.median(z)) / 1000.)
    transform = provider.transform.copy()
    transform[:3, 3] += [0., .5, 0.]
    cloud = TargetCloud("pink_label", points, scene, target, transform, {
        "camera_serial": FRONT_SERIAL, "intrinsics_sha256": FRONT_INTRINSICS_SHA256,
        "calibration_sha256": CALIBRATION_SHA256,
        "camera_monotonic_ns": sample["host_monotonic_ns"]["front_rgb"],
        "object_point_count": len(points), "segmentation_scope": "visible_pink_label_surface",
        "object_cloud_sha256": hashlib.sha256(points.tobytes()).hexdigest()})
    service = LocalGraspService("graspgen", endpoint, expected_model_sha256=expected_model_sha256)
    candidates = service.propose(cloud, max_candidates=8)
    model_tcp = rigid_pose(service.last_evidence["health"]["transform_grasp_from_model_tcp"])
    target_base = (transform @ np.r_[target, 1.])[:3]
    for candidate in candidates:
        candidate["transform_base"] = (transform @ rigid_pose(candidate["transform_camera"])).tolist()
        tcp = rigid_pose(candidate["transform_base"]) @ model_tcp
        candidate["model_tcp_base_xyz_m"] = tcp[:3, 3].tolist()
        candidate["model_tcp_target_distance_m"] = float(np.linalg.norm(tcp[:3, 3] - target_base))
    return {"schema_version": "arx.grasp.camera-proposal-audit.v1",
            "offline": True, "robot_status_used": False, "robot_commands_sent": False,
            "live_motion_eligible": False, "ik_review_performed": False,
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "target_evidence": cloud.evidence, "model_evidence": service.last_evidence,
            "target_right_base_xyz_m": target_base.tolist(),
            "candidates": candidates,
            "limitations": ["Visible label surface only; full tube segmentation is not available.",
                             "No synchronized arm feedback or ARX gripper transfer validation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-audit", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18093")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.camera_audit, args.image_dir, args.endpoint, args.expected_model_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"candidates": len(result["candidates"]), "output": str(args.output),
                      "robot_commands_sent": False}))


if __name__ == "__main__":
    main()
