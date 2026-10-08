"""Robot independent sensor cloud interface and proposal engines.

No arm handles, simulator data, or motor commands are available to these engines.
All poses are grasp-frame poses in the supplied camera optical coordinate frame.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TargetCloud:
    target_id: str
    object_camera_m: np.ndarray
    scene_camera_m: np.ndarray
    target_camera_m: np.ndarray
    camera_to_base: np.ndarray
    evidence: dict
    target_surface_camera_m: np.ndarray | None = None
    robot_self_camera_m: np.ndarray | None = None

    def __post_init__(self):
        for points in (self.object_camera_m, self.scene_camera_m, self.target_surface_camera_m,
                       self.robot_self_camera_m):
            if points is None:
                continue
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or not np.isfinite(points).all()
            ):
                raise ValueError("metric sensor point cloud must be finite Nx3")
        if (
            self.target_camera_m.shape != (3,)
            or not np.isfinite(self.target_camera_m).all()
            or np.linalg.norm(self.target_camera_m) < 1e-6
        ):
            raise ValueError("target camera point must be finite and nonzero")
        rigid_pose(self.camera_to_base)


class GraspProposalEngine(Protocol):
    def propose(self, cloud: TargetCloud, *, max_candidates: int) -> list[dict]: ...


def rigid_pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
        or np.linalg.det(pose[:3, :3]) < 0.999
    ):
        raise ValueError("grasp candidate is not a finite rigid pose")
    return pose


def transfer_grasp_pose(transform_camera, camera_to_base, model_from_robot_tcp):
    """Map a learned model-base pose to a robot TCP without sending motion.

    T_base_from_tcp = T_base_from_camera @ T_camera_from_model @
    T_model_from_tcp. A model's base origin is not its finger centre.
    """
    return (
        rigid_pose(camera_to_base)
        @ rigid_pose(transform_camera)
        @ rigid_pose(model_from_robot_tcp)
    )


def parallel_jaw_candidates(candidates, *, max_candidates):
    """Explicit two-finger half-turn variants around model +Z approach.

    The variant exchanges the closing-axis sign. Its model score is inherited
    for ranking, not recomputed or asserted to be a new success probability.
    Motion review must still check the robot pose and swept scene.
    """
    half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
    result = []
    for index, item in enumerate(candidates):
        source = dict(item, model_candidate_index=index, pose_variant="model_original")
        result.append(source)
        result.append(
            dict(
                source,
                transform_camera=(
                    rigid_pose(item["transform_camera"]) @ half_turn
                ).tolist(),
                pose_variant="parallel_jaw_half_turn",
                score_semantic="inherited_not_rescored",
            )
        )
        if len(result) >= max_candidates:
            break
    return result[:max_candidates]


class LocalGraspService:
    """Contact-GraspNet / GraspGen protocol, shared by robot adapters."""

    def __init__(
        self,
        engine: str,
        endpoint: str | None,
        *,
        timeout_s=60.0,
        expected_gripper=None,
        expected_model_sha256=None,
        sampling_options=None,
    ):
        self.engine, self.endpoint, self.timeout_s = engine, endpoint, timeout_s
        self.expected_gripper, self.expected_model_sha256 = (
            expected_gripper,
            expected_model_sha256,
        )
        self.last_evidence = {}
        self.sampling_options = sampling_options or {}

    def propose(self, cloud, *, max_candidates):
        if not self.endpoint:
            raise ValueError(f"{self.engine} local model service is unconfigured")
        if len(cloud.object_camera_m) < 32:
            raise ValueError("grasp model requires at least 32 target depth points")
        with urllib.request.urlopen(
            self.endpoint.rstrip("/") + "/health", timeout=self.timeout_s
        ) as response:
            health = json.loads(response.read(1024 * 1024))
        if not health.get("ready", health.get("ok", False)):
            raise ValueError("local grasp model is not ready")
        if self.expected_gripper and health.get("gripper_id") != self.expected_gripper:
            raise ValueError("local grasp model gripper identity differs")
        if (
            self.expected_model_sha256
            and health.get("model_sha256") != self.expected_model_sha256
        ):
            raise ValueError("local grasp model SHA differs")
        if self.engine == "graspgen":
            centroid = cloud.object_camera_m.mean(axis=0)
            payload = {
                "point_cloud": (cloud.object_camera_m - centroid).tolist(),
                "frame": "object_centered_opencv_camera_xyz_m",
                "topk_num_grasps": max_candidates,
                "filter_collisions": False,
            }
            payload.update(self.sampling_options)
            if payload.get("horizontal_closing_max") is not None:
                payload["up_camera"] = (
                    cloud.camera_to_base[:3, :3].T @ np.array([0.0, 0.0, 1.0])
                ).tolist()
            route = "/generate"
        elif self.engine == "contact_graspnet":
            centroid = np.zeros(3)
            payload = {
                "point_cloud": cloud.object_camera_m.tolist(),
                "frame": "opencv_camera_xyz_m",
                "max_candidates": max_candidates,
            }
            route = "/propose"
        else:
            raise ValueError("unknown grasp engine")
        request = urllib.request.Request(
            self.endpoint.rstrip("/") + route,
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = json.loads(response.read(4 * 1024 * 1024))
        if raw.get("ok") is False or raw.get("error"):
            raise ValueError("local grasp service failed: " + str(raw.get("error")))
        result = []
        for item in raw.get("grasps", raw.get("candidates", []))[:max_candidates]:
            pose = rigid_pose(
                item.get("transform_model")
                if self.engine == "graspgen"
                else item.get("transform_camera")
            )
            pose[:3, 3] += centroid
            score = float(item.get("score", 0.0))
            if not np.isfinite(score):
                raise ValueError("non-finite grasp score")
            result.append({"transform_camera": pose.tolist(), "score": score})
        if not result:
            raise ValueError("local grasp service returned no valid target grasps")
        self.last_evidence = {
            "health": health,
            "response_evidence": raw.get("evidence", {}),
            "proposal_only": True,
            "scene_collision_filter_applied": False,
        }
        return result


def geometry_proposal(
    cloud: TargetCloud,
    tcp_rotation_base,
    *,
    surface_offset_m=0.0,
    orientation_search_rad=0.0,
):
    """Tube-specific target approach with the current ARX TCP orientation.

    The visible label is a surface proxy. A measured tube-radius correction may
    be configured; zero leaves the proxy unchanged. This is explicitly geometric,
    and makes no learned grasp quality claim.
    """
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = cloud.camera_to_base[:3, :3].T @ tcp_rotation_base
    target = cloud.target_camera_m.copy()
    camera_pose[:3, 3] = target + surface_offset_m * target / np.linalg.norm(target)
    result = [
        {
            "transform_camera": camera_pose.tolist(),
            "score": 1.0,
            "pose_kind": "arx_tcp",
            "quality_claim": "target_proxy_only",
        }
    ]
    if orientation_search_rad:
        if not 0 < orientation_search_rad <= 0.1:
            raise ValueError("geometric orientation search must be in 0..0.1 rad")
        from scipy.spatial.transform import Rotation

        for vector in (
            [0.0, orientation_search_rad, 0.0],
            [0.0, -orientation_search_rad, 0.0],
            [0.0, 0.0, -orientation_search_rad],
            [0.0, 0.0, orientation_search_rad],
            [orientation_search_rad, 0.0, 0.0],
            [-orientation_search_rad, 0.0, 0.0],
        ):
            pose = camera_pose.copy()
            pose[:3, :3] = cloud.camera_to_base[:3, :3].T @ (
                Rotation.from_rotvec(vector).as_matrix() @ tcp_rotation_base
            )
            result.append(
                {
                    "transform_camera": pose.tolist(),
                    "score": 0.99,
                    "pose_kind": "arx_tcp",
                    "quality_claim": "target_proxy_only",
                    "world_rotation_delta_rad": vector,
                }
            )
    return result
