# Copyright (c) 2026 Zetta Contributors
"""Pinned external-scene adapter for reBot G1-D + Dex1-1 MuJoCo assets.

The reBot bundle builds MJCF from its URDF at runtime.  This bridge deliberately
loads one fixed adapter path and class only after hashing the adapter, selected
scene configuration, URDF, and every referenced mesh.  It is not a generic
Python entrypoint mechanism.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import importlib.util
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np

from rollout_runtime.backends.mujoco_session import MujocoSessionError

__all__ = [
    "REBOT_G1D_ACTION_NAMES",
    "REBOT_G1D_TASK_SPECS",
    "RebotG1DSession",
    "compute_rebot_asset_manifest",
]

REBOT_G1D_ACTION_NAMES = (
    "LZ_mt_Joint",
    "LZ_it_Joint",
    "Yaw_Joint",
    "torso_Joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)

REBOT_G1D_TASK_SPECS = {
    "grasp": {
        "scene_config": "config/g1d_mujoco.yaml",
        "adapter_path": "simulation/g1d_mujoco_env.py",
        "skill_path": "scripts/sim_geometric_grasp.py",
    },
    "fallen": {
        "scene_config": "config/g1d_fallen_bottle_to_bin.yaml",
        "adapter_path": "simulation/g1d_mujoco_env_bin.py",
        "skill_path": "scripts/sim_fallen_bottle_to_bin.py",
    },
}
"""Fixed task-to-scene/code mapping for the audited reBot skill bundle."""

_HASHED_CODE_PATHS = (
    "simulation/g1d_mujoco_env.py",
    "simulation/g1d_mujoco_env_bin.py",
    "simulation/geometric_grasp.py",
    "scripts/sim_geometric_grasp.py",
    "scripts/sim_fallen_bottle_to_bin.py",
)
_MANIFEST_DOMAIN = b"agentic-embodied-rebot-g1d-assets-v2\0"
_SUCCESS_HOLD_STEPS = 25


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise MujocoSessionError(
            "reBot G1-D scene loading requires PyYAML; install the project "
            "with the 'mujoco-rebot' extra"
        ) from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MujocoSessionError(
            f"cannot read reBot scene config {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("simulation"), Mapping
    ):
        raise MujocoSessionError(
            f"reBot scene config {path} must contain a 'simulation' mapping"
        )
    return payload


def _resolve_asset_file(root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise MujocoSessionError(f"{label} must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise MujocoSessionError(f"{label} escapes the reBot asset root: {relative!r}")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise MujocoSessionError(
            f"{label} is missing or outside the reBot asset root: {relative!r}"
        ) from exc
    if not resolved.is_file():
        raise MujocoSessionError(f"{label} is not a regular file: {relative!r}")
    return resolved


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def compute_rebot_asset_manifest(
    asset_root: str | Path,
    scene_config_path: str = "config/g1d_mujoco.yaml",
) -> dict[str, Any]:
    """Hash the executable adapter and all files needed by one reBot scene.

    Args:
        asset_root: Extracted, repository-external reBot asset directory.
        scene_config_path: Relative YAML path under ``asset_root``.

    Returns:
        A JSON-safe manifest containing the aggregate digest and per-file hashes.

    Raises:
        MujocoSessionError: A path escapes the root, a required file is missing,
            or the YAML/URDF structure is invalid.
    """
    try:
        root = Path(asset_root).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise MujocoSessionError(
            f"reBot asset root does not exist: {asset_root!s}"
        ) from exc
    if not root.is_dir():
        raise MujocoSessionError(f"reBot asset root is not a directory: {root}")

    code_files = {
        _resolve_asset_file(root, relative, label="reBot audited code")
        for relative in _HASHED_CODE_PATHS
    }
    scene_config = _resolve_asset_file(
        root, scene_config_path, label="reBot scene config"
    )
    payload = _load_yaml(scene_config)
    simulation = payload["simulation"]
    urdf_relative = simulation.get("urdf_path")
    if not isinstance(urdf_relative, str) or not urdf_relative:
        raise MujocoSessionError(
            "reBot simulation.urdf_path must be a non-empty string"
        )
    urdf = _resolve_asset_file(root, urdf_relative, label="reBot URDF")
    try:
        urdf_root = ET.parse(urdf).getroot()
    except (ET.ParseError, OSError) as exc:
        raise MujocoSessionError(f"cannot parse reBot URDF {urdf}: {exc}") from exc

    files = {*code_files, scene_config, urdf}
    for mesh in urdf_root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            raise MujocoSessionError("reBot URDF contains a mesh without filename")
        mesh_path = _resolve_asset_file(
            root,
            (PurePosixPath(urdf_relative).parent / filename).as_posix(),
            label="reBot mesh",
        )
        files.add(mesh_path)

    aggregate = hashlib.sha256(_MANIFEST_DOMAIN)
    entries: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest, size = _sha256_file(path)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        entries.append({"path": relative, "sha256": digest, "bytes": size})
    return {
        "schema_version": 2,
        "scene_config": scene_config.relative_to(root).as_posix(),
        "sha256": aggregate.hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "files": entries,
    }


def _load_adapter(root: Path, manifest_sha256: str, adapter_path: str) -> ModuleType:
    adapter = _resolve_asset_file(root, adapter_path, label="reBot adapter")
    root_fingerprint = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    adapter_fingerprint = hashlib.sha256(adapter_path.encode("utf-8")).hexdigest()[:8]
    module_name = (
        f"_agentic_rebot_g1d_{manifest_sha256[:12]}_"
        f"{root_fingerprint}_{adapter_fingerprint}"
    )
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, adapter)
    if spec is None or spec.loader is None:
        raise MujocoSessionError(
            f"cannot create module spec for reBot adapter {adapter}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:  # noqa: BLE001 - pinned external adapter boundary
        sys.modules.pop(module_name, None)
        raise MujocoSessionError(
            f"cannot load pinned reBot adapter: {type(exc).__name__}: {exc}"
        ) from exc
    return module


class RebotG1DSession:
    """One pinned reBot G1-D/Dex1-1 scene with a Gym-like blocking API."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._closed = False
        self._step_index = 0
        self._initial_bottle_position: np.ndarray | None = None
        self._grasp_bottle_height: float | None = None
        self._max_lift_height = 0.0
        self._selected_arm: str | None = None
        self._success_streak = 0
        task_value = self._config.get("rebot_task")
        self._task = None if task_value is None else str(task_value)
        root_value = self._config.get("asset_root")
        expected_digest = str(self._config.get("asset_manifest_sha256") or "")
        scene_config_path = str(
            self._config.get("rebot_scene_config") or "config/g1d_mujoco.yaml"
        )
        try:
            root = Path(str(root_value)).expanduser().resolve(strict=True)
            manifest = compute_rebot_asset_manifest(root, scene_config_path)
        except MujocoSessionError:
            raise
        except (OSError, ValueError) as exc:
            raise MujocoSessionError(f"cannot resolve reBot assets: {exc}") from exc
        actual_digest = str(manifest["sha256"])
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise MujocoSessionError(
                "reBot asset manifest mismatch: "
                f"expected {expected_digest}, calculated {actual_digest}"
            )

        payload = _load_yaml(
            _resolve_asset_file(root, scene_config_path, label="reBot scene config")
        )
        simulation = copy.deepcopy(dict(payload["simulation"]))
        render = dict(simulation.get("render") or {})
        render["width"] = int(self._config["image_width"])
        render["height"] = int(self._config["image_height"])
        camera_name = self._config.get("camera_name")
        self._camera_name = None if camera_name is None else str(camera_name)
        simulation["render"] = render

        adapter_path = "simulation/g1d_mujoco_env.py"
        if self._task is not None:
            task_spec = REBOT_G1D_TASK_SPECS.get(self._task)
            if task_spec is None:
                raise MujocoSessionError(f"unsupported reBot task {self._task!r}")
            if scene_config_path != task_spec["scene_config"]:
                raise MujocoSessionError(
                    f"reBot task {self._task!r} requires scene config "
                    f"{task_spec['scene_config']!r}"
                )
            adapter_path = task_spec["adapter_path"]
        elif scene_config_path == REBOT_G1D_TASK_SPECS["fallen"]["scene_config"]:
            # Even a non-task smoke of the fallen scene must contain the bin.
            adapter_path = REBOT_G1D_TASK_SPECS["fallen"]["adapter_path"]

        module = _load_adapter(root, actual_digest, adapter_path)
        action_names = tuple(getattr(module, "CONTROLLED_JOINTS", ()))
        if action_names != REBOT_G1D_ACTION_NAMES:
            raise MujocoSessionError(
                "pinned reBot adapter exposes an unexpected controlled-joint order"
            )
        env_type = getattr(module, "G1DMujocoEnv", None)
        if not isinstance(env_type, type):
            raise MujocoSessionError("pinned reBot adapter has no G1DMujocoEnv class")
        self._action_names = action_names
        try:
            self._env = env_type(simulation, headless=True)
            indices = self._env._joint_indices
            low = np.asarray(
                [float(indices[name].lower) for name in self._action_names],
                dtype=np.float32,
            )
            high = np.asarray(
                [float(indices[name].upper) for name in self._action_names],
                dtype=np.float32,
            )
        except BaseException as exc:  # noqa: BLE001 - native scene boundary
            env = getattr(self, "_env", None)
            if env is not None:
                env.close()
            dependency_hint = ""
            if np.__version__ != "1.26.4":
                dependency_hint = (
                    " Verify that the dedicated 'mujoco-rebot' extra is installed "
                    "(including numpy==1.26.4)."
                )
            raise MujocoSessionError(
                "cannot construct reBot G1-D scene: "
                f"{type(exc).__name__}: {exc}.{dependency_hint}"
            ) from exc
        if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
            self.close()
            raise MujocoSessionError("reBot controlled-joint bounds are invalid")
        self.descriptor = {
            "action_shape": (len(self._action_names),),
            "action_low": low,
            "action_high": high,
            "action_names": self._action_names,
            "asset_manifest_sha256": actual_digest,
        }

    def _observation(self, raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
        observation = dict(raw or self._env.observation())
        observation["joint_targets"] = np.asarray(
            [self._env.targets[name] for name in self._action_names], dtype=np.float64
        )
        observation["water_bottle_position"] = self._env.get_body_position(
            "water_bottle"
        )
        return observation

    def reset(
        self, *, seed: int | None, options: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            raw_options = dict(options or {})
            unknown = sorted(
                set(raw_options)
                - {"bottle_position", "bottle_quaternion_wxyz", "randomize_bottle"}
            )
            if unknown:
                raise ValueError(f"unknown reBot reset options: {unknown}")
            raw = self._env.reset()
            scene = self._env.config.get("scene", {})
            nominal = np.asarray(scene["bottle_position"], dtype=np.float64)
            if "bottle_position" in raw_options:
                position = np.asarray(raw_options["bottle_position"], dtype=np.float64)
            else:
                position = nominal.copy()
                randomize = bool(
                    raw_options.get(
                        "randomize_bottle",
                        self._config.get("rebot_randomize_bottle", True),
                    )
                )
                if randomize:
                    jitter = np.asarray(
                        scene.get("bottle_xy_jitter_m", [0.0, 0.0]),
                        dtype=np.float64,
                    )
                    if jitter.shape != (2,) or np.any(jitter < 0.0):
                        raise ValueError(
                            "scene.bottle_xy_jitter_m must contain two non-negative values"
                        )
                    position[:2] += np.random.default_rng(seed).uniform(-jitter, jitter)
            quaternion = np.asarray(
                raw_options.get(
                    "bottle_quaternion_wxyz",
                    scene.get("fallen_bottle_quaternion", [1.0, 0.0, 0.0, 0.0]),
                ),
                dtype=np.float64,
            )
            if position.shape != (3,):
                raise ValueError("bottle_position must contain three values")
            if quaternion.shape != (4,):
                raise ValueError("bottle_quaternion_wxyz must contain four values")
            self._env.set_free_body_pose("water_bottle_joint", position, quaternion)
            raw = self._env.observation()
            self._step_index = 0
            self._initial_bottle_position = self._env.get_body_position("water_bottle")
            self._grasp_bottle_height = None
            self._max_lift_height = 0.0
            self._selected_arm = None
            self._success_streak = 0
            return self._observation(raw), {
                "provider": "rebot_g1d",
                "seed": seed,
                "bottle_position": self._initial_bottle_position.tolist(),
                "asset_manifest_sha256": self.descriptor["asset_manifest_sha256"],
            }
        except BaseException as exc:  # noqa: BLE001 - native scene boundary
            raise MujocoSessionError(
                f"reBot reset failed: {type(exc).__name__}: {exc}"
            ) from exc

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        try:
            vector = np.asarray(action, dtype=np.float64)
            if vector.shape != (len(self._action_names),):
                raise ValueError(
                    f"expected action shape {(len(self._action_names),)}, got {vector.shape}"
                )
            self._env.set_joint_targets(
                dict(zip(self._action_names, vector, strict=True))
            )
            raw = self._env.step()
            self._step_index += 1
            max_steps = self._config.get("max_episode_steps")
            truncated = max_steps is not None and self._step_index >= int(max_steps)
            bottle = self._env.get_body_position("water_bottle")
            initial = self._initial_bottle_position
            lift_height = 0.0 if initial is None else float(bottle[2] - initial[2])
            contacts = self._env.contact_pairs()
            info = {
                "provider": "rebot_g1d",
                "step": self._step_index,
                "bottle_position": bottle.tolist(),
                "bottle_lift_height": lift_height,
                "contact_count": len(contacts),
            }
            terminated = False
            if self._task is not None:
                evaluation = self._evaluate_task(bottle, contacts)
                info.update(evaluation)
                terminated = bool(evaluation["is_success"])
            return (
                self._observation(raw),
                0.0,
                terminated,
                bool(truncated),
                info,
            )
        except BaseException as exc:  # noqa: BLE001 - native scene boundary
            raise MujocoSessionError(
                f"reBot step failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _bottle_gripper_sides(contacts: list[tuple[str, str]]) -> set[str]:
        sides: set[str] = set()
        for pair in contacts:
            lowered = tuple(name.lower() for name in pair)
            if not any("water_bottle" in name for name in lowered):
                continue
            for side in ("left", "right"):
                if any(side in name and "dex1" in name for name in lowered):
                    sides.add(side)
        return sides

    def _evaluate_task(
        self, bottle: np.ndarray, contacts: list[tuple[str, str]]
    ) -> dict[str, Any]:
        """Evaluate task success from simulator state, independently of policy."""
        sides = self._bottle_gripper_sides(contacts)
        if self._selected_arm is None and sides:
            self._selected_arm = sorted(sides)[0]
        if self._grasp_bottle_height is None and sides:
            self._grasp_bottle_height = float(bottle[2])
        if self._grasp_bottle_height is not None:
            self._max_lift_height = max(
                self._max_lift_height,
                float(bottle[2]) - self._grasp_bottle_height,
            )

        scene = self._env.config["scene"]
        motion = self._env.config["motion"]
        final_gripper_contact = bool(sides)
        metrics: dict[str, Any] = {
            "rebot_task": self._task,
            "success_evaluator": "runtime_state_v1",
            "selected_arm": self._selected_arm,
            "max_bottle_lift_height": self._max_lift_height,
            "final_gripper_bottle_contact": final_gripper_contact,
        }
        candidate = False
        if self._task == "grasp":
            table_top = float(scene["table_position"][2] + scene["table_half_size"][2])
            table_near_edge = float(
                scene["table_position"][0] - scene["table_half_size"][0]
            )
            payload_retained = bool(
                final_gripper_contact
                and float(bottle[2]) > table_top - 0.02
                and float(bottle[0]) < table_near_edge
            )
            return_error: float | None = None
            if self._selected_arm is not None:
                ready = np.asarray(
                    motion["fixed_ready_tcp_position"], dtype=np.float64
                ).copy()
                ready[1] = (
                    abs(ready[1]) if self._selected_arm == "left" else -abs(ready[1])
                )
                gripper = self._env.get_gripper_pose(self._selected_arm)[:3, 3]
                return_error = float(np.linalg.norm(gripper - ready))
            candidate = bool(
                self._max_lift_height > 0.08
                and payload_retained
                and return_error is not None
                and return_error < 0.03
            )
            metrics.update(
                payload_retained=payload_retained,
                fixed_ready_return_error_m=return_error,
            )
        elif self._task == "fallen":
            bin_config = scene["trash_bin"]
            bin_position = np.asarray(bin_config["position"], dtype=np.float64)
            inner_half_width = float(
                bin_config["half_width_m"] - bin_config["wall_thickness_m"]
            )
            bin_rim_z = float(bin_position[2] + bin_config["height_m"])
            relative = np.asarray(bottle, dtype=np.float64) - bin_position
            deposited = bool(
                abs(float(relative[0])) < inner_half_width - 0.015
                and abs(float(relative[1])) < inner_half_width - 0.015
                and float(bin_position[2]) < float(bottle[2]) < bin_rim_z
            )
            candidate = bool(
                self._max_lift_height > 0.08 and deposited and not final_gripper_contact
            )
            metrics.update(
                bottle_relative_to_bin=relative.tolist(),
                deposited=deposited,
            )
        else:  # pragma: no cover - constructor validates the fixed set
            raise MujocoSessionError(f"unsupported reBot task {self._task!r}")

        self._success_streak = self._success_streak + 1 if candidate else 0
        metrics.update(
            success_candidate=candidate,
            success_hold_steps=self._success_streak,
            required_success_hold_steps=_SUCCESS_HOLD_STEPS,
            is_success=self._success_streak >= _SUCCESS_HOLD_STEPS,
        )
        return metrics

    def render(self) -> np.ndarray | None:
        if self._config.get("render_mode") != "rgb_array":
            return None
        try:
            frame = np.asarray(self._env.render(camera_name=self._camera_name)["img"])
        except BaseException as exc:  # noqa: BLE001 - native renderer boundary
            raise MujocoSessionError(
                f"reBot render failed: {type(exc).__name__}: {exc}"
            ) from exc
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise MujocoSessionError(
                "reBot RGB render must return uint8 HWC with 3 or 4 channels, "
                f"got shape={frame.shape}, dtype={frame.dtype}"
            )
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return np.ascontiguousarray(frame)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()
