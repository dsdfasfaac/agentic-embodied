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
from robots.arx.deployment.feature_observation import FeatureObservationUnavailable

CALIBRATION_SHA256 = "854854c1d0e512ccfe6411c1d3ebf8b74394f9138e1a713f4660169ab6cded10"
FRONT_INTRINSICS_SHA256 = "75dcfabf7b3f76d46a99edbd95b25b61409dcc4509f61e73bacd48c358364f33"
FRONT_SERIAL = "260422272500"
RIGHT_SERIAL = "260422275847"
RIGHT_INTRINSICS_SHA256 = "09644e8ab297c023f87415e1610f46bf15916b69b731970e4c964ae0bc7f1add"
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
        self.wrist_mount = None
        self.wrist_tracker = None
        self.wrist_validations = 0
        self.wrist_validation_stamp = None
        self.last_camera_target_ns = {}
        self.target_baselines = {}
        self.last_interaction_camera = None
        self.acquired_target_left = None
        self.lift_reference_ns = {}
        self.last_observation_quality = {"status": "observed", "unavailable_features": []}
        self.max_component_pixels = (700, 45, 40)
        self.max_tracking_displacement_px = 80

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
        right = config.cameras[2] if len(config.cameras) == 3 else None
        if right is not None and getattr(right, "robot_mount_calibration_file", None) is not None:
            if (right.name != "right_rgb" or right.serial != RIGHT_SERIAL or not right.depth_enabled
                    or right.calibration_sha256 != RIGHT_INTRINSICS_SHA256):
                raise ValueError("right wrist observer requires pinned RGB intrinsics and aligned depth")
            mount = _pinned_json(right.robot_mount_calibration_file, right.robot_mount_calibration_sha256)
            matrix = np.asarray(mount.get("T_link6_from_camera"))
            if (mount.get("schema_version") != "arx.real.wrist_mount.v1"
                    or mount.get("camera_serial") != RIGHT_SERIAL
                    or mount.get("parent_frame") != "right_link6_after_rotation"
                    or mount.get("controller_fk_sha256") != CONTROLLER_FK_SHA256
                    or mount.get("rgb_intrinsics_sha256") != RIGHT_INTRINSICS_SHA256
                    or mount.get("quality", {}).get("accepted") is not True
                    or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
                    or abs(np.linalg.det(matrix[:3, :3]) - 1) > 1e-6):
                raise ValueError("invalid or unverified right wrist mount calibration")
            self.wrist_mount = matrix
            self.wrist_mount_sha = right.robot_mount_calibration_sha256
            self.wrist_intrinsics = _pinned_json(right.calibration_file, RIGHT_INTRINSICS_SHA256)["camera"]["intrinsics"]
            self.wrist_tracker = PickTubeRgbdProvider()
            # The wrist approaches the label; pixel area grows with proximity.
            # Its separate metric size and cross-camera checks remain below.
            self.wrist_tracker.max_component_pixels = (8000, 100, 120)
            self.wrist_tracker.max_tracking_displacement_px = 25
            self.link6_fk = CommandKinematics(self.controller_fk.calibration.model_copy(update={"tcp_offset": [0., 0., 0.]}))

    def feature_sources(self):
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        common = {"provider_id": "dodo-ac-one-picktube-rgbd-v3", "provider_sha256": source_sha,
                  "max_age_ms": 150}
        sources = [
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
        if self.wrist_mount is not None:
            for source in sources[1:]:
                source["provider_id"] = "dodo-picktube-dual-rgbd-" + self.wrist_mount_sha
                source["source_ids"] = list(dict.fromkeys([
                    *source["source_ids"], "right_rgb", "right_depth_mm", *RIGHT_JOINT_IDS]))
                source["source_kind"] = "rgbd_fused"
        return sources

    def _pink_component(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        if rgb.dtype != np.uint8 or rgb.shape != (240, 320, 3):
            raise ValueError("PickTube front RGB must be 320x240 uint8")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red, green, blue = (rgb[..., i].astype(np.int16) for i in range(3))
        # Rack context establishes target identity on acquisition. Once the
        # label is tracked, the moving arm can obscure the rack; each new
        # frame must still supply a bounded, visible pink component and depth.
        if self.last_centre is None:
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
                max_area, max_width, max_height = self.max_component_pixels
                if area > max_area or width > max_width or height > max_height:
                    continue
                displacement = float(np.hypot(cx - self.last_centre[0], cy - self.last_centre[1]))
                if displacement > self.max_tracking_displacement_px:
                    continue
                score = displacement - 0.01 * area
            candidates.append((score, component, cx, cy))
        if not candidates:
            raise ValueError("pink tube label is not reliably visible in front RGB")
        _, component, cx, cy = min(candidates)
        self.last_centre = (cx, cy)
        return labels == component

    def _deproject(self, x: float, y: float, depth_m: float) -> np.ndarray:
        return self._deproject_intrinsics(x, y, depth_m, self.calibration["camera"]["intrinsics"])

    @staticmethod
    def _deproject_intrinsics(x, y, depth_m, source):
        import pyrealsense2 as rs
        intrinsic = rs.intrinsics()
        intrinsic.width, intrinsic.height = 320, 240
        intrinsic.fx, intrinsic.fy = source["fx"] / 2, source["fy"] / 2
        intrinsic.ppx = (source["ppx"] + 0.5) / 2 - 0.5
        intrinsic.ppy = (source["ppy"] + 0.5) / 2 - 0.5
        intrinsic.model = rs.distortion.inverse_brown_conrady
        intrinsic.coeffs = source["coeffs"]
        return np.asarray(rs.rs2_deproject_pixel_to_point(intrinsic, [x, y], depth_m), dtype=np.float64)

    def _label_sample(self, rgb, depth_mm, tracker, transform, deproject, camera, hardware):
        rgb, depth = np.asarray(rgb), np.asarray(depth_mm)
        if depth.dtype != np.uint16 or depth.shape != (240, 320):
            raise ValueError("aligned D405 millimetre depth required")
        mask = tracker._pink_component(rgb)
        valid = mask & (depth >= 80) & (depth <= 1500)
        yy, xx = np.nonzero(valid)
        # Near an occluding gripper the few remaining depth pixels can land
        # on its edge. Require support over the label before trusting a 3D
        # centroid or using it to contradict the other camera.
        if len(xx) < 12 or len(xx) < .8 * np.count_nonzero(mask):
            raise ValueError("pink tube has insufficient valid metric depth")
        support = depth[valid].astype(float)
        median = float(np.median(support))
        if np.median(np.abs(support - median)) > 25:
            raise ValueError("pink tube depth is inconsistent")
        if camera == "right_rgb":
            if ((xx.max() - xx.min() + 1) * median / 1000. / (self.wrist_intrinsics["fx"] / 2) > .04
                    or (yy.max() - yy.min() + 1) * median / 1000. / (self.wrist_intrinsics["fy"] / 2) > .05):
                raise ValueError("pink tube label is not reliably visible in front RGB")
        point = deproject(float(np.median(xx)), float(np.median(yy)), median / 1000.)
        depth_key = camera.removesuffix("_rgb") + "_depth_mm"
        stamp = hardware.get("depth_monotonic_ns", {}).get(depth_key)
        if stamp != hardware.get("camera_monotonic_ns", {}).get(camera) or type(stamp) is not int or stamp <= 0:
            raise ValueError("RGB-D identity timestamps differ")
        return {"camera": camera, "depth_key": depth_key, "stamp": stamp,
                "mask": mask, "depth_support_fraction": len(xx) / np.count_nonzero(mask),
                "transform_left": transform, "point_camera": point,
                "point_left": (transform @ np.r_[point, 1.])[:3], "deproject": deproject}

    def target_sample(self, images, hardware):
        """Use the wrist only after fresh cross-camera agreement established identity."""
        transient = ("pink tube", "yellow test-tube rack")
        front, front_error, wrist = None, None, None
        state = np.asarray(hardware["measured_state"])
        closed = self.closed_policy if self.closed_policy is not None else 0.
        opened = self.open_policy if self.open_policy is not None else -3.4
        fraction = (float(state[13]) - closed) / (opened - closed)
        reference = self.acquired_target_left if fraction > .30 else self.last_target_left
        def continuous(sample, tracker, old_centre):
            if reference is not None and np.linalg.norm(sample["point_left"] - reference) > (.025 if fraction > .30 else .035):
                tracker.last_centre = old_centre
                raise ValueError("pink tube target position changed inconsistently")
            return sample
        front_centre = self.last_centre
        try:
            front = self._label_sample(images["front_rgb"], images.get("front_depth_mm"),
                self, self.transform, self._deproject, "front_rgb", hardware)
            front = continuous(front, self, front_centre)
        except ValueError as exc:
            if not str(exc).startswith(transient):
                raise
            front_error = exc
            front = None
            self.last_centre = front_centre
        cross_error = None
        if self.wrist_mount is not None:
            if (hardware.get("camera_health", {}).get("right_rgb", {}).get("device_id") != RIGHT_SERIAL
                    or hardware.get("camera_calibration_sha256", {}).get("right_rgb") != RIGHT_INTRINSICS_SHA256):
                raise ValueError("right wrist camera identity differs")
            p, rotation, _ = self.link6_fk.fk(state[7:13])
            pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = p + [0., -.5, 0.]
            transform = pose @ self.wrist_mount
            deproject = lambda x, y, z: self._deproject_intrinsics(x, y, z, self.wrist_intrinsics)
            if front is not None and self.wrist_tracker.last_centre is None:
                point = (np.linalg.inv(transform) @ np.r_[front["point_left"], 1.])[:3]
                if point[2] > 0:
                    i = self.wrist_intrinsics
                    self.wrist_tracker.last_centre = (
                        float(i["fx"] / 2 * point[0] / point[2] + (i["ppx"] + .5) / 2 - .5),
                        float(i["fy"] / 2 * point[1] / point[2] + (i["ppy"] + .5) / 2 - .5))
            try:
                wrist_centre = self.wrist_tracker.last_centre
                wrist = self._label_sample(images["right_rgb"], images.get("right_depth_mm"),
                    self.wrist_tracker, transform, deproject, "right_rgb", hardware)
                wrist = continuous(wrist, self.wrist_tracker, wrist_centre)
            except ValueError as exc:
                if not str(exc).startswith(transient):
                    raise
                # A missing sample cannot disprove the already established
                # identity. Reacquisition still needs fresh RGB-D and the
                # world-position continuity check above; never reuse depth.
                self.wrist_tracker.last_centre = wrist_centre
                if self.wrist_validations < 3:
                    self.wrist_validations = 0
                wrist = None
            if front is not None and wrist is not None:
                cross_error = float(np.linalg.norm(front["point_left"] - wrist["point_left"]))
                if cross_error <= .015:
                    if wrist["stamp"] != self.wrist_validation_stamp:
                        self.wrist_validations += 1
                        self.wrist_validation_stamp = wrist["stamp"]
                else:
                    self.wrist_validations = 0
        sample = front
        if sample is None and wrist is not None and self.wrist_validations >= 3:
            if self.last_target_left is not None and np.linalg.norm(wrist["point_left"] - self.last_target_left) <= .025:
                sample = wrist
        if sample is None:
            raise front_error or ValueError("pink tube label is not reliably visible in front RGB")
        if self.acquired_target_left is None:
            self.acquired_target_left = sample["point_left"].copy()
        if front is not None:
            self.target_baselines.setdefault("front_rgb", front["point_left"].copy())
        if wrist is not None and cross_error is not None and cross_error <= .015:
            self.target_baselines.setdefault("right_rgb", wrist["point_left"].copy())
        self.last_observation_quality = {
            "status": "observed", "unavailable_features": [], "camera": sample["camera"],
            "target_monotonic_ns": sample["stamp"], "cross_camera_error_m": cross_error,
            "depth_support_fraction": sample.get("depth_support_fraction"),
            "wrist_identity_validations": self.wrist_validations,
            "wrist_mount_sha256": getattr(self, "wrist_mount_sha", None),
        }
        return sample

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
        # Recorded PickTube grasps settle near -0.85 policy units; with the
        # 0.0 closed and -3.4 open endpoints this is about 25% open.
        gripper_closed = gripper_fraction <= 0.30
        current = hardware.get("auxiliary_feedback", {}).get(RIGHT_GRIPPER_CURRENT)
        current_stamp = hardware.get("auxiliary_monotonic_ns", {}).get(RIGHT_GRIPPER_CURRENT)
        if (type(current) not in (int, float) or not np.isfinite(current)
                or current_stamp != hardware.get("state_monotonic_ns")):
            raise ValueError("fresh right gripper motor current is required")
        depth_stamp = hardware.get("depth_monotonic_ns", {}).get("front_depth_mm")
        if type(depth_stamp) is not int or depth_stamp <= 0:
            raise ValueError("front D405 depth timestamp is required")
        try:
            sample = self.target_sample(images, hardware)
            depth_stamp = sample["stamp"]
            if depth_stamp <= self.last_camera_target_ns.get(sample["camera"], 0):
                raise ValueError("target depth frame is not new")
            self.last_target_left = sample["point_left"].copy()
            self.last_depth_centre = self.last_centre
            distance = float(np.linalg.norm(self.last_target_left - (tool_centre + [0., -.5, 0.])))
            self.last_target_ns = depth_stamp
            self.last_camera_target_ns[sample["camera"]] = depth_stamp
        except ValueError as exc:
            expected = {
                "yellow test-tube rack is not visible in front RGB": "target_not_visible",
                "yellow test-tube rack is not reliably visible": "target_not_visible",
                "pink tube label is not reliably visible in front RGB": "target_not_visible",
                "pink tube has insufficient valid metric depth": "target_depth_unavailable",
                "pink tube depth is inconsistent": "target_depth_inconsistent",
                "pink tube target position changed inconsistently": "target_tracking_discontinuity",
                "target depth frame is not new": "target_frame_repeated",
            }
            if str(exc) not in expected:
                raise
            self.success_hold_frames = 0
            available = {"privileged.interaction.gripper_closed": gripper_closed}
            raise FeatureObservationUnavailable(
                str(exc), available=available,
                unavailable=[source["name"] for source in self.feature_sources()
                             if source["name"] not in available],
                reason_code=expected[str(exc)],
                last_valid_target_monotonic_ns=self.last_target_ns,
            ) from exc
        target_left = self.last_target_left
        tool_left = tool_centre + np.array([0.0, -0.5, 0.0])
        if not gripper_closed and abs(float(current)) < CONTACT_CURRENT_THRESHOLD:
            # Rebase while the empty open gripper approaches. This removes
            # view-dependent label/FK bias before contact, using real depth,
            # without treating camera motion as a tube lift.
            self.target_baselines[sample["camera"]] = target_left.copy()
            self.lift_reference_ns[sample["camera"]] = depth_stamp
        self.initial_target_left = self.target_baselines[sample["camera"]]
        self.last_observation_quality["lift_reference_monotonic_ns"] = self.lift_reference_ns.get(sample["camera"])
        if sample["camera"] != self.last_interaction_camera:
            self.success_hold_frames = 0
        self.last_interaction_camera = sample["camera"]
        values = self._interaction_values(
            target_left, tool_left, distance, gripper_closed, float(current),
        )
        if observation_id is not None:
            self.last_observation_id, self.last_values = observation_id, values.copy()
        return values


def create_provider():
    return PickTubeRgbdProvider()
