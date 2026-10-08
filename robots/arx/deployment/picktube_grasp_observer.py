"""Target-labelled D405 point clouds using the existing audited PickTube frames."""

import hashlib

import numpy as np

from robots.manipulation.grasp_proposals import TargetCloud
from .picktube_rgbd_provider import (
    CALIBRATION_SHA256,
    CONTROLLER_FK_SHA256,
    FRONT_INTRINSICS_SHA256,
    FRONT_SERIAL,
    NOMINAL_CHAIN_SHA256,
    RIGHT_SERIAL,
    RIGHT_INTRINSICS_SHA256,
)


class PickTubeGraspObserver:
    def __init__(self, provider, *, cloud_mode="label_only"):
        if cloud_mode not in {"label_only", "upright_tube"}:
            raise ValueError("unknown target cloud mode")
        self.provider = provider
        self.cloud_mode = cloud_mode

    def cloud(self, observation, images):
        provider = self.provider
        hardware = observation["hardware"]
        sample = provider.target_sample(images, hardware)
        camera, depth_key = sample["camera"], sample["depth_key"]
        serial = FRONT_SERIAL if camera == "front_rgb" else RIGHT_SERIAL
        intrinsics_sha = (
            FRONT_INTRINSICS_SHA256
            if camera == "front_rgb"
            else RIGHT_INTRINSICS_SHA256
        )
        calibration_sha = (
            CALIBRATION_SHA256 if camera == "front_rgb" else provider.wrist_mount_sha
        )
        rgb = np.asarray(images[camera])
        depth = np.asarray(images[depth_key])
        if depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
            raise ValueError("aligned front D405 depth must be uint16 millimetres")
        mask = sample["mask"]
        # Use the same admitted physical range as the target observer.
        valid = (depth >= sample.get("depth_min_mm", 80)) & (depth <= 1500)
        yy, xx = np.nonzero(mask & valid)
        if len(xx) < 12:
            raise ValueError("pink label requires at least 12 fresh depth points")
        z_mm = depth[yy, xx].astype(float)
        median = np.median(z_mm)
        if np.median(np.abs(z_mm - median)) > 25:
            raise ValueError("pink target depth is inconsistent")
        # A target-only cloud prevents another colour from entering model input.
        object_cloud = np.asarray(
            [
                sample["deproject"](float(x), float(y), z / 1000.0)
                for x, y, z in zip(xx, yy, z_mm)
            ]
        )
        target = sample["point_camera"]
        # Sampling at two-pixel intervals retains ~4 mm spacing at 0.4 m.
        sy, sx = np.nonzero(valid[::2, ::2])
        sy, sx = sy * 2, sx * 2
        scene = np.asarray(
            [
                sample["deproject"](float(x), float(y), float(depth[y, x]) / 1000.0)
                for x, y in zip(sx, sy)
            ]
        )
        transform = sample["transform_left"].copy()
        # The original extrinsic is left-base; all ARX grasp plans use right-base.
        transform[:3, 3] += [0.0, 0.5, 0.0]
        state = np.asarray(hardware["measured_state"])
        predicted, _, _ = provider.controller_fk.fk(state[7:13])
        feedback = np.asarray(hardware.get("right_tcp_xyz_m"))
        if (
            hardware.get("right_tcp_monotonic_ns") != hardware["state_monotonic_ns"]
            or hardware.get("right_tcp_frame") != "right_arm_local_base"
            or feedback.shape != (3,)
            or not np.isfinite(feedback).all()
            or np.linalg.norm(predicted - feedback) > 0.01
        ):
            raise ValueError(
                "grasp controller FK differs from the measured joint sample"
            )
        if (
            hardware.get("camera_health", {}).get(camera, {}).get("device_id") != serial
            or hardware.get("camera_calibration_sha256", {}).get(camera)
            != intrinsics_sha
            or hardware.get("depth_monotonic_ns", {}).get(depth_key)
            != hardware.get("camera_monotonic_ns", {}).get(camera)
        ):
            raise ValueError(
                "front RGB/depth camera identity, calibration or timestamp differs"
            )
        scope = "visible_pink_label_surface"
        if self.cloud_mode == "upright_tube":
            from robots.manipulation.target_clouds import upright_tube_mask

            vy, vx = np.nonzero(valid)
            camera_points = np.asarray(
                [
                    sample["deproject"](float(x), float(y), float(depth[y, x]) / 1000.0)
                    for x, y in zip(vx, vy)
                ]
            )
            base_points = (
                transform @ np.c_[camera_points, np.ones(len(camera_points))].T
            ).T[:, :3]
            grid = np.full((*depth.shape, 3), np.nan)
            grid[vy, vx] = base_points
            seed_base = (transform @ np.r_[target, 1.0])[:3]
            surface_mask = upright_tube_mask(rgb, grid, seed_base, mask)
            selected = surface_mask[vy, vx]
            object_cloud = camera_points[selected]
            mask = surface_mask
            scope = "measured_seed_connected_upright_tube_surface"
        return TargetCloud(
            "pink_label",
            object_cloud,
            scene,
            target,
            transform,
            {
                "observation_id": observation["observation_id"],
                "camera": camera,
                "camera_serial": serial,
                "depth_monotonic_ns": hardware["depth_monotonic_ns"][depth_key],
                "state_monotonic_ns": hardware["state_monotonic_ns"],
                "calibration_sha256": calibration_sha,
                "intrinsics_sha256": intrinsics_sha,
                "controller_fk_sha256": CONTROLLER_FK_SHA256,
                "motion_joint_bounds_rad": [
                    link.limits for link in provider.tool_fk.calibration.links
                ],
                "tool_geometry_sha256": NOMINAL_CHAIN_SHA256,
                "object_cloud_sha256": hashlib.sha256(
                    object_cloud.tobytes()
                ).hexdigest(),
                "scene_cloud_sha256": hashlib.sha256(scene.tobytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "object_point_count": len(object_cloud),
                "scene_point_count": len(scene),
                "target_pixel_xy": [float(np.median(xx)), float(np.median(yy))],
                "target_base_xyz_m": (transform @ np.r_[target, 1.0])[:3].tolist(),
                "frame": "right_arm_local_base",
                "depth_units": "m",
                "segmentation_scope": scope,
                "limitations": [
                    "Label is a tube surface proxy, not a complete tube mesh.",
                    "TCP depth clearance does not certify full arm or finger collision freedom.",
                ],
            },
        )
