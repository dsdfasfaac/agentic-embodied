# Copyright (c) 2026 Zetta Contributors
"""Dodo AC one PickTube observer: aligned D405 depth and controller FK.

The fixed front camera calibration maps its optical frame to the LEFT arm
local base. The right controller reports TCP position in the RIGHT arm local
base, which is 0.5 m in negative Y from the left local base. The offset is
the AC one model relation documented in ARX_X5_CAMERA_TRANSFORMS_HANDOFF.md.
The output point is the visible pink label centre, a grasp-target proxy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CALIBRATION_SHA256 = "854854c1d0e512ccfe6411c1d3ebf8b74394f9138e1a713f4660169ab6cded10"
FRONT_INTRINSICS_SHA256 = "75dcfabf7b3f76d46a99edbd95b25b61409dcc4509f61e73bacd48c358364f33"
FRONT_SERIAL = "260422272500"
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "manifests/real/dodo_front_d405_rgbd_calibration_BL_source.json"
RIGHT_JOINT_IDS = [f"right_joint_{i}" for i in range(1, 7)]


def _pinned_json(path: Path, digest: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"PickTube calibration SHA differs: {path}")
    return json.loads(raw)


class PickTubeRgbdProvider:
    def __init__(self):
        self.calibration = _pinned_json(CALIBRATION_PATH, CALIBRATION_SHA256)
        if self.calibration.get("quality", {}).get("accepted") is not True:
            raise ValueError("front camera extrinsic was not accepted")
        self.transform = np.asarray(self.calibration["transforms"]["T_B_from_C"], dtype=np.float64)
        if self.transform.shape != (4, 4) or not np.isfinite(self.transform).all() or not np.allclose(
            self.transform[3], [0, 0, 0, 1], atol=1e-8
        ):
            raise ValueError("invalid front camera to left base extrinsic")
        self.last_centre: tuple[float, float] | None = None
        self.closed_policy: float | None = None
        self.open_policy: float | None = None

    def validate_hardware(self, config) -> None:
        front = config.cameras[0]
        if (
            front.name != "front_rgb" or front.serial != FRONT_SERIAL
            or not front.depth_enabled
            or (front.capture_width, front.capture_height) != (640, 480)
            or (front.width, front.height) != (320, 240)
            or front.calibration_sha256 != FRONT_INTRINSICS_SHA256
        ):
            raise ValueError("PickTube requires the pinned 640x480 front D405 with aligned depth")
        if hashlib.sha256(front.calibration_file.read_bytes()).hexdigest() != FRONT_INTRINSICS_SHA256:
            raise ValueError("front D405 runtime calibration SHA differs")
        self.closed_policy = float(config.right_gripper_closed_policy)
        self.open_policy = float(config.right_gripper_open_policy)

    def feature_sources(self):
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        common = {"provider_id": "dodo-ac-one-picktube-rgbd-v1", "provider_sha256": source_sha,
                  "max_age_ms": 150}
        return [
            {**common, "name": "privileged.interaction.gripper_closed",
             "source_kind": "joint_feedback", "source_ids": ["right_gripper_policy"],
             "scalar_type": "boolean", "units": "boolean"},
            {**common, "name": "privileged.selected.target_gripper_distance_m",
             "source_kind": "rgbd_fused",
             "source_ids": ["front_rgb", "front_depth_mm", *RIGHT_JOINT_IDS],
             "scalar_type": "number", "units": "m"},
        ]

    def _pink_component(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        if rgb.dtype != np.uint8 or rgb.shape != (240, 320, 3):
            raise ValueError("PickTube front RGB must be 320x240 uint8")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red, green, blue = (rgb[..., i].astype(np.int16) for i in range(3))
        hsv_mask = (((hsv[..., 0] >= 155) | (hsv[..., 0] <= 6))
                    & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 65))
        rgb_mask = ((red >= green + 18) & (blue >= green + 5)
                    & (red >= blue - 30) & (red >= 55))
        mask = (hsv_mask | rgb_mask).astype(np.uint8)
        mask[:10] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for component in range(1, count):
            x, y, width, height, area = stats[component]
            if area < 8 or width < 2 or height < 2:
                continue
            cx, cy = map(float, centroids[component])
            if self.last_centre is None:
                # All 50 PickTube episode starts contain a 79-107 pixel
                # label (320x240). Admit a margin, but reject large pink
                # objects such as packaging before establishing a track.
                if (cy >= 0.38 * rgb.shape[0] or not 50 <= area <= 180
                        or width > 30 or height > 22):
                    continue
                score = -area
            else:
                # Across 1251 sampled tracked frames the largest component
                # was 636 pixels, 38x28. Larger blobs are out of this
                # task's validated visual envelope and may be distractors.
                if area > 700 or width > 45 or height > 40:
                    continue
                displacement = float(np.hypot(cx - self.last_centre[0], cy - self.last_centre[1]))
                if displacement > 80:
                    continue
                score = displacement - 0.01 * area
            candidates.append((score, component, cx, cy))
        if not candidates:
            raise ValueError("pink tube label is not reliably visible in front RGB")
        _, component, cx, cy = min(candidates)
        self.last_centre = (cx, cy)
        return labels == component

    def _deproject(self, x: float, y: float, depth_m: float) -> np.ndarray:
        import pyrealsense2 as rs

        source = self.calibration["camera"]["intrinsics"]
        intrinsic = rs.intrinsics()
        intrinsic.width, intrinsic.height = 320, 240
        intrinsic.fx, intrinsic.fy = source["fx"] / 2, source["fy"] / 2
        intrinsic.ppx = (source["ppx"] + 0.5) / 2 - 0.5
        intrinsic.ppy = (source["ppy"] + 0.5) / 2 - 0.5
        intrinsic.model = rs.distortion.inverse_brown_conrady
        intrinsic.coeffs = source["coeffs"]
        return np.asarray(rs.rs2_deproject_pixel_to_point(intrinsic, [x, y], depth_m), dtype=np.float64)

    def measure_distance(self, rgb: np.ndarray, depth_mm: np.ndarray,
                         right_tcp_xyz_m: np.ndarray) -> float:
        depth = np.asarray(depth_mm)
        if depth.dtype != np.uint16 or depth.shape != (240, 320):
            raise ValueError("aligned front D405 millimetre depth required")
        component = self._pink_component(np.asarray(rgb))
        support = depth[component]
        support = support[(support >= 80) & (support <= 1500)]
        if support.size < 12:
            raise ValueError("pink tube has insufficient valid metric depth")
        median_mm = float(np.median(support))
        if float(np.median(np.abs(support.astype(np.float64) - median_mm))) > 25:
            raise ValueError("pink tube depth is inconsistent")
        yy, xx = np.nonzero(component & (depth >= 80) & (depth <= 1500))
        x, y = float(np.median(xx)), float(np.median(yy))
        camera_xyz = self._deproject(x, y, median_mm / 1000)
        target_left = (self.transform @ np.r_[camera_xyz, 1.0])[:3]
        tcp_right = np.asarray(right_tcp_xyz_m, dtype=np.float64)
        if tcp_right.shape != (3,) or not np.isfinite(tcp_right).all():
            raise ValueError("controller FK right TCP must be finite XYZ metres")
        tcp_left = tcp_right + np.array([0.0, -0.5, 0.0])
        return float(np.linalg.norm(target_left - tcp_left))

    def observe(self, observation, images):
        if self.closed_policy is None or self.open_policy is None:
            raise ValueError("PickTube provider hardware was not validated")
        hardware = observation["hardware"]
        if hardware.get("right_tcp_monotonic_ns") != hardware.get("state_monotonic_ns"):
            raise ValueError("controller FK TCP must share the fresh joint feedback timestamp")
        state = np.asarray(hardware["measured_state"], dtype=np.float64)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("fresh 14D feedback required")
        tcp_right = np.asarray(hardware.get("right_tcp_xyz_m"), dtype=np.float64)
        if (tcp_right.shape != (3,) or not np.isfinite(tcp_right).all()
                or hardware.get("right_tcp_frame") != "right_arm_local_base"):
            raise ValueError("controller FK right TCP in local base is required")
        distance = self.measure_distance(
            np.asarray(images["front_rgb"]),
            np.asarray(images.get("front_depth_mm")), tcp_right,
        )
        gripper_span = self.open_policy - self.closed_policy
        gripper_fraction = (float(state[13]) - self.closed_policy) / gripper_span
        return {
            "privileged.interaction.gripper_closed": gripper_fraction <= 0.25,
            "privileged.selected.target_gripper_distance_m": distance,
        }


def create_provider():
    return PickTubeRgbdProvider()
