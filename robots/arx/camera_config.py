# Copyright (c) 2026 Zetta Contributors
"""Strict loader for calibrated ARX Task7 MuJoCo cameras."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from robots.arx.contracts import ARX_CAMERA_NAMES


@dataclass(frozen=True, slots=True)
class MujocoCamera:
    semantic_name: str
    role: str
    parent_body: str
    pose_xyz_m: tuple[float, float, float] | None
    pose_quat_wxyz: tuple[float, float, float, float] | None
    calibration_id: str


@dataclass(frozen=True, slots=True)
class DiagnosticCamera:
    name: str
    parent_body: str
    pose_xyz_m: tuple[float, float, float]
    xyaxes: tuple[float, float, float, float, float, float]
    fovy_deg: float


@dataclass(frozen=True, slots=True)
class ArxCameraConfig:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    cameras: tuple[MujocoCamera, ...]
    diagnostic_cameras: tuple[DiagnosticCamera, ...] = ()

    @property
    def vertical_fovy_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    @property
    def missing_extrinsics(self) -> tuple[str, ...]:
        return tuple(c.semantic_name for c in self.cameras if c.pose_xyz_m is None or c.pose_quat_wxyz is None)

    def require_complete(self) -> None:
        if self.missing_extrinsics:
            raise ValueError("ARX camera extrinsics are not calibrated: " + ", ".join(self.missing_extrinsics))


def _pose(value: object, size: int, label: str) -> tuple[float, ...] | None:
    if value is None or (isinstance(value, list) and any(item is None for item in value)):
        return None
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} finite values or be null")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def load_camera_config(path: str | Path) -> ArxCameraConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "zetta_arx_camera_config_v1":
        raise ValueError("unsupported ARX camera configuration schema")
    if payload.get("intrinsics_shared") is not True:
        raise ValueError("Task7 currently requires explicitly shared intrinsics")
    calibration = json.loads(Path(payload["intrinsics_source"]).read_text(encoding="utf-8"))
    intrinsics = calibration["camera"]["intrinsics"]
    width, height = (int(item) for item in payload["image_size"])
    source_width, source_height = int(calibration["camera"]["width"]), int(calibration["camera"]["height"])
    sx, sy = width / source_width, height / source_height
    raw = payload["cameras"]
    if tuple(raw) != ARX_CAMERA_NAMES:
        raise ValueError(f"camera order must be {ARX_CAMERA_NAMES}")
    cameras = tuple(MujocoCamera(name, item["role"], item["parent_body"], _pose(item["pose_xyz_m"], 3, f"{name}.pose_xyz_m"), _pose(item["pose_quat_wxyz"], 4, f"{name}.pose_quat_wxyz"), item["calibration_id"]) for name, item in raw.items())
    diagnostics = tuple(
        DiagnosticCamera(
            name, item["parent_body"],
            _pose(item["pose_xyz_m"], 3, f"{name}.pose_xyz_m"),
            _pose(item["xyaxes"], 6, f"{name}.xyaxes"), float(item["fovy_deg"]),
        )
        for name, item in payload.get("diagnostic_cameras", {}).items()
    )
    if any(item.pose_xyz_m is None or item.xyaxes is None for item in diagnostics):
        raise ValueError("diagnostic camera poses must be complete")
    return ArxCameraConfig(width, height, float(intrinsics["fx"]) * sx, float(intrinsics["fy"]) * sy, float(intrinsics["ppx"]) * sx, float(intrinsics["ppy"]) * sy, cameras, diagnostics)
