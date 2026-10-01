# Copyright (c) 2026 Zetta Contributors
"""Strict, dependency-light contracts for the ARX X5 integration."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

__all__ = [
    "ARX_ACTION_DIM",
    "ARX_CAMERA_NAMES",
    "ARX_GRIPPER_INDICES",
    "ArxCameraSpec",
    "ArxControlSpec",
    "ArxModelContract",
    "ArxStartingScenes",
    "ArxTaskManifest",
    "load_model_contract",
    "load_task_manifest",
]

ARX_ACTION_DIM = 14
ARX_CAMERA_NAMES = ("front_rgb", "left_rgb", "right_rgb")
ARX_GRIPPER_INDICES = (6, 13)
_MODEL_SCHEMA = "cosmos3_edge_arx_task7_v1"
_TASK_SCHEMA = "zetta_arx_task_v1"
_T = TypeVar("_T")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return {str(key): item for key, item in value.items()}


def _strict_dataclass(cls: type[_T], value: Any, context: str) -> _T:
    payload = _object(value, context)
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {unknown}")
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid {context}: {exc}") from exc


def _reject_unknown(cls: type[Any], payload: Mapping[str, Any], context: str) -> None:
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"{context} has unknown keys: {unknown}")


def _finite_tuple(
    value: Sequence[Any], length: int, context: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{context} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{context} must contain only finite values")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class ArxCameraSpec:
    """One model-facing RGB camera."""

    name: str
    width: int = 320
    height: int = 240
    channels: int = 3
    dtype: str = "uint8"
    color_order: str = "RGB"
    calibration_id: str = "unassigned"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera name must not be empty")
        if (self.width, self.height, self.channels) != (320, 240, 3):
            raise ValueError("Task7 cameras must be uint8 RGB [240,320,3]")
        if self.dtype != "uint8" or self.color_order != "RGB":
            raise ValueError("Task7 cameras must use uint8 RGB")
        if not self.calibration_id:
            raise ValueError("camera calibration_id must not be empty")

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> ArxCameraSpec:
        return _strict_dataclass(cls, value, context)


@dataclasses.dataclass(frozen=True, slots=True)
class ArxModelContract:
    """Pinned Cosmos3-Edge Task7 serving contract."""

    schema_version: str
    domain_name: str
    domain_id: int | None
    cameras: tuple[ArxCameraSpec, ...]
    state_dim: int
    action_dim: int
    action_horizon: int
    conditioning_fps: float
    action_semantic: str
    action_normalization: str
    prompt_format: str
    view_mode: str

    def __post_init__(self) -> None:
        if self.schema_version != _MODEL_SCHEMA:
            raise ValueError(f"schema_version must be {_MODEL_SCHEMA!r}")
        if self.domain_name not in {"arx-task7-x5", "arx_task7"}:
            raise ValueError("unsupported ARX Task7 domain name")
        if self.domain_name == "arx-task7-x5" and self.domain_id != 17:
            raise ValueError("legacy Task7 requires domain ID 17")
        if self.domain_name == "arx_task7" and self.domain_id != 17:
            raise ValueError("checkpoint Task7 requires domain ID 17")
        if tuple(camera.name for camera in self.cameras) != ARX_CAMERA_NAMES:
            raise ValueError(f"camera order must be {ARX_CAMERA_NAMES}")
        if self.state_dim != ARX_ACTION_DIM or self.action_dim != ARX_ACTION_DIM:
            raise ValueError("Task7 state/action dimension must be 14")
        if self.action_horizon != 32:
            raise ValueError("Task7 action horizon must be 32")
        if float(self.conditioning_fps) != 15.0:
            raise ValueError("Task7 conditioning_fps must be 15")
        if self.action_semantic != "continuous raw positions":
            raise ValueError("Task7 actions must be continuous raw positions")
        if self.action_normalization != "raw":
            raise ValueError("Task7 actions must not use client-side normalization")
        if self.prompt_format != "json" or self.view_mode != "concat_view":
            raise ValueError("Task7 requires JSON prompts and concat_view")

    @classmethod
    def from_mapping(cls, value: Any) -> ArxModelContract:
        payload = _object(value, "ARX model contract")
        _reject_unknown(cls, payload, "ARX model contract")
        raw_cameras = payload.get("cameras")
        if not isinstance(raw_cameras, (list, tuple)):
            raise ValueError("ARX model contract cameras must be an array")
        payload["cameras"] = tuple(
            ArxCameraSpec.from_mapping(item, f"cameras[{index}]")
            for index, item in enumerate(raw_cameras)
        )
        return _strict_dataclass(cls, payload, "ARX model contract")


@dataclasses.dataclass(frozen=True, slots=True)
class ArxControlSpec:
    """Task-specific model-state and command-processing settings."""

    lock_left_arm: bool = False
    lock_right_arm: bool = False
    lock_right_gripper: bool = False
    zero_left_model_state: bool = False
    zero_right_model_state: bool = False
    model_left_gripper: float | None = None
    model_right_gripper: float | None = None
    arm_filter_alpha: float = 0.35
    gripper_filter_alpha: float = 0.35
    max_joint_step: float = 0.035
    max_gripper_step: float = 0.08
    max_joint_delta: float = 0.5
    max_gripper_delta: float = 1.0
    gripper_command_offsets: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.lock_left_arm and self.lock_right_arm:
            raise ValueError("both ARX arms cannot be locked")
        if self.lock_right_arm and self.lock_right_gripper:
            raise ValueError("lock_right_arm already locks its gripper")
        for name in ("arm_filter_alpha", "gripper_filter_alpha"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in (
            "max_joint_step",
            "max_gripper_step",
            "max_joint_delta",
            "max_gripper_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        offsets = _finite_tuple(self.gripper_command_offsets, 2, "gripper offsets")
        object.__setattr__(self, "gripper_command_offsets", offsets)
        for name in ("model_left_gripper", "model_right_gripper"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or null")

    @classmethod
    def from_mapping(cls, value: Any) -> ArxControlSpec:
        payload = _object(value, "ARX control spec")
        if "gripper_command_offsets" in payload:
            payload["gripper_command_offsets"] = tuple(
                payload["gripper_command_offsets"]
            )
        return _strict_dataclass(cls, payload, "ARX control spec")


@dataclasses.dataclass(frozen=True, slots=True)
class ArxStartingScenes:
    """Allowlisted deterministic scene selection policy."""

    selection: str
    default_scene_id: str
    allowed_scene_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.selection not in {"fixed", "seeded_uniform", "explicit"}:
            raise ValueError("invalid starting-scene selection mode")
        if not self.allowed_scene_ids or any(not item for item in self.allowed_scene_ids):
            raise ValueError("allowed_scene_ids must contain non-empty IDs")
        if len(set(self.allowed_scene_ids)) != len(self.allowed_scene_ids):
            raise ValueError("allowed_scene_ids must be unique")
        if self.default_scene_id not in self.allowed_scene_ids:
            raise ValueError("default_scene_id must be allowlisted")

    @classmethod
    def from_mapping(cls, value: Any) -> ArxStartingScenes:
        payload = _object(value, "starting_scenes")
        raw_ids = payload.get("allowed_scene_ids")
        if not isinstance(raw_ids, (list, tuple)):
            raise ValueError("allowed_scene_ids must be an array")
        payload["allowed_scene_ids"] = tuple(str(item) for item in raw_ids)
        return _strict_dataclass(cls, payload, "starting_scenes")


@dataclasses.dataclass(frozen=True, slots=True)
class ArxTaskManifest:
    """Immutable task behavior and starting-scene allowlist."""

    schema_version: str
    task_id: int
    name: str
    display_name: str
    instruction: str
    start_state: tuple[float, ...]
    execution_steps: int
    max_steps: int
    control: ArxControlSpec
    starting_scenes: ArxStartingScenes
    success: dict[str, Any]
    failure: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != _TASK_SCHEMA:
            raise ValueError(f"schema_version must be {_TASK_SCHEMA!r}")
        if not 0 <= self.task_id < 7:
            raise ValueError("task_id must be in [0,6]")
        if not self.name or not self.display_name or not self.instruction.strip():
            raise ValueError("task names and instruction must not be empty")
        object.__setattr__(
            self, "start_state", _finite_tuple(self.start_state, 14, "start_state")
        )
        if not 1 <= self.execution_steps <= 32:
            raise ValueError("execution_steps must be in [1,32]")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        for name, value in (("success", self.success), ("failure", self.failure)):
            if not isinstance(value, dict) or not value.get("evaluator"):
                raise ValueError(f"{name} must name an evaluator")
            unknown = sorted(set(value) - {"evaluator", "parameters"})
            if unknown:
                raise ValueError(f"{name} has unknown keys: {unknown}")
            if not isinstance(value.get("parameters", {}), dict):
                raise ValueError(f"{name}.parameters must be an object")

    @classmethod
    def from_mapping(cls, value: Any) -> ArxTaskManifest:
        payload = _object(value, "ARX task manifest")
        _reject_unknown(cls, payload, "ARX task manifest")
        payload["control"] = ArxControlSpec.from_mapping(payload.get("control", {}))
        payload["starting_scenes"] = ArxStartingScenes.from_mapping(
            payload.get("starting_scenes")
        )
        if "start_state" in payload:
            payload["start_state"] = tuple(payload["start_state"])
        for key in ("success", "failure"):
            payload[key] = _object(payload.get(key), key)
        return _strict_dataclass(cls, payload, "ARX task manifest")


def _load_json_yaml(path: str | Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML subset used by pinned manifests."""
    source = Path(path)
    try:
        return _object(json.loads(source.read_text(encoding="utf-8")), str(source))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source} must use the dependency-free JSON-compatible YAML subset"
        ) from exc


def load_model_contract(path: str | Path) -> ArxModelContract:
    return ArxModelContract.from_mapping(_load_json_yaml(path))


def load_task_manifest(path: str | Path) -> ArxTaskManifest:
    return ArxTaskManifest.from_mapping(_load_json_yaml(path))
