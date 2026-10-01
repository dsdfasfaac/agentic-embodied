# Copyright (c) 2026 Zetta Contributors
"""Frozen ARX hardware configuration and worker-side device construction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from robots.arx.contracts import ARX_CAMERA_NAMES, load_model_contract, load_task_manifest
from zetta.evolution.jsonio import file_sha256

from .arx_ros2_device import ArxRos2Device, Ros2Topics
from .arx_x5_device import ArmCalibration, ArxX5Device
from .contracts import StrictModel
from .real_backend import CameraIdentity, RealBackend, RealBackendConfig
from .real_camera import RealSenseCameraSource, RealSenseCameraSpec

REAL_JOINT_CHANNELS = tuple(
    [f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper_policy"]
    + [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper_policy"]
)


class ArmSettings(StrictModel):
    can_port: str = Field(min_length=1)
    arm_type: Literal[0, 1, 2]
    joint_min_rad: tuple[float, float, float, float, float, float]
    joint_max_rad: tuple[float, float, float, float, float, float]
    gripper_native_min: float
    gripper_native_max: float
    gripper_policy_scale: float
    gripper_policy_offset: float

    def calibration(self) -> ArmCalibration:
        return ArmCalibration(**self.model_dump())


class CameraSettings(StrictModel):
    name: Literal["front_rgb", "left_rgb", "right_rgb"]
    serial: str = Field(min_length=1)
    calibration_id: str = Field(min_length=1)
    calibration_file: Path
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    capture_width: int = Field(gt=0)
    capture_height: int = Field(gt=0)
    capture_fps: int = Field(gt=0)
    depth_enabled: bool = False

    def identity(self) -> CameraIdentity:
        return CameraIdentity(
            self.name, self.serial, self.calibration_id,
            self.calibration_sha256, self.width, self.height,
        )


class TimingSettings(StrictModel):
    control_hz: float = Field(gt=0)
    max_sensor_skew_ms: float = Field(gt=0)
    max_sensor_age_ms: float = Field(gt=0)
    observation_timeout_s: float = Field(gt=0)
    arrival_timeout_s: float = Field(gt=0)
    feedback_poll_s: float = Field(gt=0)
    position_tolerance: tuple[float, ...] = Field(min_length=14, max_length=14)


class RealHardwareConfig(StrictModel):
    schema_version: Literal["arx.real.hardware.v1"]
    arm_transport: Literal["arx_ros2", "arx_sdk"]
    camera_transport: Literal["realsense"]
    left: ArmSettings
    right: ArmSettings
    command_left: bool
    command_right: bool
    cameras: tuple[CameraSettings, CameraSettings, CameraSettings]
    timing: TimingSettings
    right_gripper_closed_policy: float
    right_gripper_open_policy: float
    ros2_topics: dict[str, str] | None = None

    @model_validator(mode="after")
    def check(self):
        if tuple(camera.name for camera in self.cameras) != ARX_CAMERA_NAMES:
            raise ValueError("three ordered camera names required")
        if len({camera.serial for camera in self.cameras}) != 3:
            raise ValueError("camera serials must be unique")
        if self.left.can_port == self.right.can_port:
            raise ValueError("left/right CAN ports must differ")
        if not self.command_left and not self.command_right:
            raise ValueError("at least one arm must be commandable")
        if self.right_gripper_closed_policy == self.right_gripper_open_policy:
            raise ValueError("right gripper endpoints must differ")
        if self.ros2_topics is not None and set(self.ros2_topics) != {
            "left_status", "right_status", "left_command", "right_command"
        }:
            raise ValueError("four ROS2 topic names are required")
        return self


def load_real_hardware_config(path: Path, expected_sha256: str) -> RealHardwareConfig:
    if file_sha256(path) != expected_sha256:
        raise ValueError("real hardware configuration SHA-256 mismatch")
    return RealHardwareConfig.model_validate_json(Path(path).read_text())


def validate_real_hardware_config(
    config: RealHardwareConfig, task_path: Path, model_path: Path,
) -> None:
    task = load_task_manifest(task_path)
    model = load_model_contract(model_path)
    if config.command_left == task.control.lock_left_arm:
        raise ValueError("left arm command setting conflicts with task lock")
    if config.command_right == task.control.lock_right_arm:
        raise ValueError("right arm command setting conflicts with task lock")
    if abs(config.timing.control_hz - model.conditioning_fps) > 1e-6:
        raise ValueError("real control frequency differs from VLA contract")
    for actual, expected in zip(config.cameras, model.cameras):
        if (
            actual.name, actual.width, actual.height, actual.calibration_id
        ) != (
            expected.name, expected.width, expected.height, expected.calibration_id
        ):
            raise ValueError(f"camera differs from VLA contract: {actual.name}")
        if file_sha256(actual.calibration_file) != actual.calibration_sha256:
            raise ValueError(f"camera calibration file SHA differs: {actual.name}")
        calibration = json.loads(actual.calibration_file.read_text())
        if (
            calibration.get("schema_version") != "arx.real.rgb_intrinsics.v1"
            or calibration.get("calibration_scope") != "rgb_intrinsics_only"
        ):
            raise ValueError(f"unsupported RGB calibration schema/scope: {actual.name}")
        camera = calibration.get("camera", {})
        intrinsics = camera.get("intrinsics")
        if not isinstance(intrinsics, dict) or set(intrinsics) != {
            "fx", "fy", "ppx", "ppy", "coeffs", "distortion_model"
        }:
            raise ValueError(f"RGB intrinsics incomplete: {actual.name}")
        numbers = [intrinsics[key] for key in ("fx", "fy", "ppx", "ppy")]
        coeffs = intrinsics["coeffs"]
        if (
            not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numbers)
            or numbers[0] <= 0 or numbers[1] <= 0
            or not isinstance(coeffs, list) or len(coeffs) != 5
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in coeffs)
            or not isinstance(intrinsics["distortion_model"], str)
            or not intrinsics["distortion_model"]
        ):
            raise ValueError(f"RGB intrinsics invalid: {actual.name}")
        if (
            camera.get("logical_name"), camera.get("serial"),
            camera.get("width"), camera.get("height"),
        ) != (
            actual.name, actual.serial,
            actual.capture_width, actual.capture_height,
        ):
            raise ValueError(f"camera calibration identity/geometry differs: {actual.name}")
    right = config.right.calibration()
    right.to_native(config.right_gripper_closed_policy)
    right.to_native(config.right_gripper_open_policy)
    config.left.calibration().to_native(task.start_state[6])
    right.to_native(task.start_state[13])


def build_real_backend(
    *, config: RealHardwareConfig, task_path: Path, model_path: Path,
) -> RealBackend:
    validate_real_hardware_config(config, task_path, model_path)
    task = load_task_manifest(task_path)
    timing = config.timing
    backend_config = RealBackendConfig(
        cameras=tuple(camera.identity() for camera in config.cameras),
        state_units=("rad",) * 6 + ("policy_gripper",)
        + ("rad",) * 6 + ("policy_gripper",),
        control_hz=timing.control_hz,
        max_sensor_skew_ms=timing.max_sensor_skew_ms,
        max_sensor_age_ms=timing.max_sensor_age_ms,
        observation_timeout_s=timing.observation_timeout_s,
        arrival_timeout_s=timing.arrival_timeout_s,
        feedback_poll_s=timing.feedback_poll_s,
        position_tolerance=timing.position_tolerance,
        depth_cameras=tuple(camera.name for camera in config.cameras if camera.depth_enabled),
    )
    if config.arm_transport == "arx_ros2":
        arms = ArxRos2Device.from_ros2(
            topics=Ros2Topics(**config.ros2_topics) if config.ros2_topics else Ros2Topics(),
            left_calibration=config.left.calibration(),
            right_calibration=config.right.calibration(),
            command_left=config.command_left,
            command_right=config.command_right,
        )
    else:
        arms = ArxX5Device.from_official_sdk(
            left_calibration=config.left.calibration(),
            right_calibration=config.right.calibration(),
            command_left=config.command_left,
            command_right=config.command_right,
        )
    try:
        cameras = RealSenseCameraSource(
            tuple(
                RealSenseCameraSpec(
                    camera.identity(), camera.serial, camera.calibration_file,
                    camera.capture_width, camera.capture_height, camera.capture_fps,
                    camera.depth_enabled,
                )
                for camera in config.cameras
            ),
            max_skew_ms=timing.max_sensor_skew_ms,
        )
    except BaseException:
        arms.close()
        raise
    return RealBackend(arms=arms, cameras=cameras, config=backend_config, task=task)
