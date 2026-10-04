# Copyright (c) 2026 Zetta Contributors
"""Dodo AC one PickTube observer: aligned D405 depth and checked arm FK.

The fixed front camera calibration maps its optical frame to the LEFT arm
local base. The right controller reports TCP position in the RIGHT arm local
base, which is 0.5 m in negative Y from the left local base. The offset is
the AC one model relation documented in ARX_X5_CAMERA_TRANSFORMS_HANDOFF.md.
The controller's end_pos is the link-six endpoint. A recorded-trajectory
calibration checks that point against joint feedback; nominal AC one tool
geometry then estimates the gripper centre. The visible pink label is a target
proxy. Contact and grasp are conservative RGB-D/current/kinematic estimates,
not tactile or MuJoCo contact flags.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from robots.arx.gateway.motion import Calibration, CommandKinematics

CALIBRATION_SHA256 = "854854c1d0e512ccfe6411c1d3ebf8b74394f9138e1a713f4660169ab6cded10"
FRONT_INTRINSICS_SHA256 = "75dcfabf7b3f76d46a99edbd95b25b61409dcc4509f61e73bacd48c358364f33"
FRONT_SERIAL = "260422272500"
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "manifests/real/dodo_front_d405_rgbd_calibration_BL_source.json"
CONTROLLER_FK_PATH = Path(__file__).resolve().parents[1] / "manifests/real/dodo_right_controller_ee_fk.json"
CONTROLLER_FK_SHA256 = "159964e1ac841d5490e6de9bbc00c2afdbc11c981076d5b5428bb68f5b2bbb86"
NOMINAL_CHAIN_PATH = Path(__file__).resolve().parents[1] / "manifests/real/ac_one_nominal_chain.json"
NOMINAL_CHAIN_SHA256 = "9ffc93ed44190f9e78a58f4010a5d65d7be828b12543233825ad41ce31c6c1ee"
RIGHT_JOINT_IDS = [f"right_joint_{i}" for i in range(1, 7)]
RIGHT_GRIPPER_CURRENT = "right_gripper_current_native"
# The first 100 frames of 50 accepted PickTube recordings have a 99th
# percentile empty/open right gripper current of 0.0904 native units.
CONTACT_CURRENT_THRESHOLD = 0.16
CONTACT_DISTANCE_MAX_M = 0.05
RETAINED_RELATIVE_DRIFT_MAX_M = 0.012
GRASP_LIFT_MIN_M = 0.005
SUCCESS_LIFT_MIN_M = 0.01
SUCCESS_HOLD_FRAMES = 5


def _pinned_json(path: Path, digest: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"PickTube calibration SHA differs: {path}")
    return json.loads(raw)


class PickTubeRgbdProvider:
    def __init__(self):
        self.calibration = _pinned_json(CALIBRATION_PATH, CALIBRATION_SHA256)
        controller = Calibration.model_validate(_pinned_json(CONTROLLER_FK_PATH, CONTROLLER_FK_SHA256))
        nominal = Calibration.model_validate(_pinned_json(NOMINAL_CHAIN_PATH, NOMINAL_CHAIN_SHA256))
        self.controller_fk = CommandKinematics(controller)
        self.tool_fk = CommandKinematics(controller.model_copy(update={"tcp_offset": nominal.tcp_offset}))
        if self.calibration.get("quality", {}).get("accepted") is not True:
            raise ValueError("front camera extrinsic was not accepted")
        self.transform = np.asarray(self.calibration["transforms"]["T_B_from_C"], dtype=np.float64)
        if self.transform.shape != (4, 4) or not np.isfinite(self.transform).all() or not np.allclose(
            self.transform[3], [0, 0, 0, 1], atol=1e-8
        ):
            raise ValueError("invalid front camera to left base extrinsic")
        self.last_centre: tuple[float, float] | None = None
        self.last_depth_centre: tuple[float, float] | None = None
        self.last_target_left: np.ndarray | None = None
        self.last_target_ns: int | None = None
        self.closed_policy: float | None = None
        self.open_policy: float | None = None
        self.initial_target_left: np.ndarray | None = None
        self.contact_relative_left: np.ndarray | None = None
        self.contact_tool_left: np.ndarray | None = None
        self.contact_rearmed = True
        self.success_hold_frames = 0
        self.success_latched = False
        self.last_observation_id: str | None = None
        self.last_values: dict | None = None

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
        if hasattr(config, "right"):
            bounds = [(link.limits[0], link.limits[1]) for link in self.controller_fk.calibration.links]
            configured = list(zip(config.right.joint_min_rad, config.right.joint_max_rad))
            if not np.allclose(configured, bounds, rtol=0, atol=1e-8):
                raise ValueError("right controller joint limits differ from audited FK envelope")

    def feature_sources(self):
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        common = {"provider_id": "dodo-ac-one-picktube-rgbd-v2", "provider_sha256": source_sha,
                  "max_age_ms": 150}
        return [
            {**common, "name": "privileged.interaction.gripper_closed",
             "source_kind": "joint_feedback", "source_ids": ["right_gripper_policy"],
             "scalar_type": "boolean", "units": "boolean"},
            {**common, "name": "privileged.interaction.gripper_contact",
             "source_kind": "rgbd_fused",
             "source_ids": ["front_rgb", "front_depth_mm", *RIGHT_JOINT_IDS,
                            "right_gripper_policy", RIGHT_GRIPPER_CURRENT],
             "scalar_type": "boolean", "units": "boolean"},
            {**common, "name": "privileged.interaction.lift_m",
             "source_kind": "camera_rgbd", "source_ids": ["front_rgb", "front_depth_mm"],
             "scalar_type": "number", "units": "m"},
            {**common, "name": "privileged.interaction.grasped",
             "source_kind": "rgbd_fused",
             "source_ids": ["front_rgb", "front_depth_mm", *RIGHT_JOINT_IDS,
                            "right_gripper_policy", RIGHT_GRIPPER_CURRENT],
             "scalar_type": "boolean", "units": "boolean"},
            {**common, "name": "privileged.interaction.success",
             "source_kind": "rgbd_fused",
             "source_ids": ["front_rgb", "front_depth_mm", *RIGHT_JOINT_IDS,
                            "right_gripper_policy", RIGHT_GRIPPER_CURRENT],
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
        rack_mask = (((hsv[..., 0] >= 18) & (hsv[..., 0] <= 45))
                     & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 90)).astype(np.uint8)
        rack_count, _, rack_stats, _ = cv2.connectedComponentsWithStats(rack_mask, 8)
        if rack_count < 2:
            raise ValueError("yellow test-tube rack is not visible in front RGB")
        rack_index = int(np.argmax(rack_stats[1:, cv2.CC_STAT_AREA]) + 1)
        rack_x, rack_y, rack_w, rack_h, rack_area = map(int, rack_stats[rack_index])
        if rack_area < 500 or rack_w < 60 or rack_h < 10:
            raise ValueError("yellow test-tube rack is not reliably visible")
        hsv_mask = (((hsv[..., 0] >= 155) | (hsv[..., 0] <= 6))
                    & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 65))
        rgb_mask = ((red >= green + 18) & (blue >= green + 5)
                    & (red >= blue - 30) & (red >= 55))
        # The current dodo D405 exposure renders the pale pink label nearly
        # neutral, while the table and blue/green tubes remain cyan.
        pale_mask = ((red >= green - 10) & (blue >= green + 4)
                     & (blue >= red) & (red >= 90) & (blue >= 100))
        mask = (hsv_mask | rgb_mask | pale_mask).astype(np.uint8)
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
                # The label starts above the yellow rack. This is decisive
                # when a pink sticker elsewhere in view has similar size.
                if (not rack_x - 5 <= cx <= rack_x + rack_w + 5
                        or not rack_y - 40 <= cy <= rack_y + 20
                        or not 25 <= area <= 200
                        or width > 30 or height > 24):
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
        self.last_target_left = target_left.copy()
        self.last_depth_centre = self.last_centre
        tcp_right = np.asarray(right_tcp_xyz_m, dtype=np.float64)
        if tcp_right.shape != (3,) or not np.isfinite(tcp_right).all():
            raise ValueError("controller FK right TCP must be finite XYZ metres")
        tcp_left = tcp_right + np.array([0.0, -0.5, 0.0])
        return float(np.linalg.norm(target_left - tcp_left))

    def _interaction_values(self, target_left: np.ndarray, tool_left: np.ndarray,
                            distance: float, gripper_closed: bool, current: float) -> dict:
        if self.initial_target_left is None:
            self.initial_target_left = target_left.copy()
        lift = float(target_left[2] - self.initial_target_left[2])
        relative = target_left - tool_left
        if not gripper_closed:
            self.contact_relative_left = None
            self.contact_tool_left = None
            self.contact_rearmed = True
        elif self.contact_relative_left is not None and (
            distance > CONTACT_DISTANCE_MAX_M or
            np.linalg.norm(relative - self.contact_relative_left) > RETAINED_RELATIVE_DRIFT_MAX_M
        ):
            self.contact_relative_left = None
            self.contact_tool_left = None
            self.contact_rearmed = False
        if (gripper_closed and self.contact_rearmed
                and self.contact_relative_left is None
                and distance <= CONTACT_DISTANCE_MAX_M
                and abs(float(current)) >= CONTACT_CURRENT_THRESHOLD):
            self.contact_relative_left = relative.copy()
            self.contact_tool_left = tool_left.copy()
            self.contact_rearmed = False
        contact = self.contact_relative_left is not None
        grasped = bool(contact and lift >= GRASP_LIFT_MIN_M and
                       np.linalg.norm(tool_left - self.contact_tool_left) >= GRASP_LIFT_MIN_M)
        self.success_hold_frames = self.success_hold_frames + 1 if (
            grasped and lift >= SUCCESS_LIFT_MIN_M
        ) else 0
        self.success_latched |= self.success_hold_frames >= SUCCESS_HOLD_FRAMES
        return {
            "privileged.interaction.gripper_closed": gripper_closed,
            "privileged.interaction.gripper_contact": contact,
            "privileged.interaction.lift_m": lift,
            "privileged.interaction.grasped": grasped,
            "privileged.interaction.success": self.success_latched,
            "privileged.selected.target_gripper_distance_m": distance,
        }

    def observe(self, observation, images):
        observation_id = observation.get("observation_id")
        if observation_id is not None and observation_id == self.last_observation_id:
            return dict(self.last_values)
        if self.closed_policy is None or self.open_policy is None:
            raise ValueError("PickTube provider hardware was not validated")
        hardware = observation["hardware"]
        if hardware.get("right_tcp_monotonic_ns") != hardware.get("state_monotonic_ns"):
            raise ValueError("controller FK TCP must share the fresh joint feedback timestamp")
        state = np.asarray(hardware["measured_state"], dtype=np.float64)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("fresh 14D feedback required")
        controller_ee = np.asarray(hardware.get("right_tcp_xyz_m"), dtype=np.float64)
        if (controller_ee.shape != (3,) or not np.isfinite(controller_ee).all()
                or hardware.get("right_tcp_frame") != "right_arm_local_base"):
            raise ValueError("controller FK right TCP in local base is required")
        right_joints = state[7:13]
        predicted_ee, _, _ = self.controller_fk.fk(right_joints)
        if np.linalg.norm(predicted_ee - controller_ee) > 0.01:
            raise ValueError("right controller FK differs from fresh joint feedback")
        tool_centre, _, _ = self.tool_fk.fk(right_joints)
        gripper_span = self.open_policy - self.closed_policy
        gripper_fraction = (float(state[13]) - self.closed_policy) / gripper_span
        gripper_closed = gripper_fraction <= 0.25
        current = hardware.get("auxiliary_feedback", {}).get(RIGHT_GRIPPER_CURRENT)
        current_stamp = hardware.get("auxiliary_monotonic_ns", {}).get(RIGHT_GRIPPER_CURRENT)
        if (type(current) not in (int, float) or not np.isfinite(current)
                or current_stamp != hardware.get("state_monotonic_ns")):
            raise ValueError("fresh right gripper motor current is required")
        depth_stamp = hardware.get("depth_monotonic_ns", {}).get("front_depth_mm")
        try:
            distance = self.measure_distance(
                np.asarray(images["front_rgb"]),
                np.asarray(images.get("front_depth_mm")), tool_centre,
            )
            if type(depth_stamp) is not int or depth_stamp <= 0:
                raise ValueError("front D405 depth timestamp is required")
            self.last_target_ns = depth_stamp
        except ValueError as exc:
            if (str(exc) != "pink tube has insufficient valid metric depth"
                    or gripper_fraction <= 0.25
                    or self.last_target_left is None or self.last_target_ns is None
                    or self.last_centre is None or self.last_depth_centre is None
                    or np.hypot(self.last_centre[0] - self.last_depth_centre[0],
                                self.last_centre[1] - self.last_depth_centre[1]) > 5
                    or type(depth_stamp) is not int
                    or not 0 <= depth_stamp - self.last_target_ns <= 1_500_000_000):
                raise
            # The tube remains at its measured rack pixel while the gripper is
            # open. A short D405 depth dropout may reuse that 3D point; the
            # gripper position still comes from this fresh joint sample.
            tcp_left = tool_centre + np.array([0.0, -0.5, 0.0])
            distance = float(np.linalg.norm(self.last_target_left - tcp_left))
        target_left = self.last_target_left
        tool_left = tool_centre + np.array([0.0, -0.5, 0.0])
        values = self._interaction_values(
            target_left, tool_left, distance, gripper_closed, float(current),
        )
        if observation_id is not None:
            self.last_observation_id, self.last_values = observation_id, values.copy()
        return values


def create_provider():
    return PickTubeRgbdProvider()
