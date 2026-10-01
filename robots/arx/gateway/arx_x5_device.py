# Copyright (c) 2026 Zetta Contributors
"""ARX_X5 SingleArm adapter; hardware-neutral gateway sees only 14D positions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .real_backend import CommandReceipt, DeviceHealth, JointSample


@dataclass(frozen=True)
class ArmCalibration:
    """Six SDK joint radians and one SDK-native gripper coordinate per arm.

    policy_gripper = sdk_gripper * gripper_policy_scale + gripper_policy_offset.
    The SDK documents different native ranges for X5-2023 and X5-2025,
    without a stable physical width unit; deployment must freeze this map.
    """

    can_port: str
    arm_type: int
    joint_min_rad: tuple[float, ...]
    joint_max_rad: tuple[float, ...]
    gripper_native_min: float
    gripper_native_max: float
    gripper_policy_scale: float
    gripper_policy_offset: float

    def __post_init__(self):
        if not self.can_port or self.arm_type not in (0, 1, 2):
            raise ValueError("ARX CAN port and SDK arm type must be explicit")
        if len(self.joint_min_rad) != 6 or len(self.joint_max_rad) != 6:
            raise ValueError("six joint bounds are required")
        values = (
            *self.joint_min_rad, *self.joint_max_rad,
            self.gripper_native_min, self.gripper_native_max,
            self.gripper_policy_scale, self.gripper_policy_offset,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ARX unit calibration must be finite")
        if any(low >= high for low, high in zip(self.joint_min_rad, self.joint_max_rad)):
            raise ValueError("invalid ARX joint bounds")
        if self.arm_type == 2 and any(
            low < -10 or high > 10
            for low, high in zip(self.joint_min_rad, self.joint_max_rad)
        ):
            raise ValueError("configured joint bounds exceed X5-2025 SDK URDF placeholder bounds")
        if self.gripper_native_min >= self.gripper_native_max or not self.gripper_policy_scale:
            raise ValueError("invalid ARX gripper calibration")

    def to_policy(self, native: float) -> float:
        return native * self.gripper_policy_scale + self.gripper_policy_offset

    def to_native(self, policy: float) -> float:
        result = (policy - self.gripper_policy_offset) / self.gripper_policy_scale
        if not self.gripper_native_min <= result <= self.gripper_native_max:
            raise ValueError("gripper command exceeds calibrated SDK-native range")
        return result


class ArxX5Device:
    """Two official SingleArm instances under the generic ArmDevice interface.

    SingleArm setter return values are not a CAN acknowledgement. A receipt
    reports only that all selected SDK calls returned without an exception.
    """

    def __init__(
        self, *, left, right, left_calibration: ArmCalibration,
        right_calibration: ArmCalibration, command_left: bool,
        command_right: bool,
    ):
        if not (command_left or command_right):
            raise ValueError("at least one arm must be commandable")
        self.left, self.right = left, right
        self.left_calibration = left_calibration
        self.right_calibration = right_calibration
        self.command_left, self.command_right = command_left, command_right
        self._closed = False

    @classmethod
    def from_official_sdk(
        cls, *, left_calibration: ArmCalibration,
        right_calibration: ArmCalibration, command_left: bool,
        command_right: bool,
    ) -> "ArxX5Device":
        # The official repo's setup.sh must make bimanual and its compiled
        # arx_x5_python extension importable. Import only when actually used.
        from bimanual import SingleArm

        left = SingleArm({
            "can_port": left_calibration.can_port,
            "type": left_calibration.arm_type,
        })
        right = SingleArm({
            "can_port": right_calibration.can_port,
            "type": right_calibration.arm_type,
        })
        return cls(
            left=left, right=right,
            left_calibration=left_calibration,
            right_calibration=right_calibration,
            command_left=command_left,
            command_right=command_right,
        )

    @staticmethod
    def _read_one(arm, calibration: ArmCalibration) -> np.ndarray:
        native = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        if native.shape != (7,) or not np.isfinite(native).all():
            raise ValueError("ARX SDK must return six joints and one gripper")
        if not (
            calibration.gripper_native_min - 1e-3
            <= native[6]
            <= calibration.gripper_native_max + 1e-3
        ):
            raise ValueError("ARX gripper feedback outside calibrated range")
        if np.any(native[:6] < np.asarray(calibration.joint_min_rad) - 1e-3) or np.any(
            native[:6] > np.asarray(calibration.joint_max_rad) + 1e-3
        ):
            raise ValueError("ARX joint feedback outside configured controller-coordinate bounds")
        result = native.copy()
        result[6] = calibration.to_policy(float(native[6]))
        return result.astype(np.float32)

    def read(self) -> JointSample:
        if self._closed:
            raise RuntimeError("ARX device is closed")
        started = time.monotonic_ns()
        left = self._read_one(self.left, self.left_calibration)
        right = self._read_one(self.right, self.right_calibration)
        finished = time.monotonic_ns()
        return JointSample(
            np.concatenate((left, right)), finished, started,
            DeviceHealth(
                transport_responsive=True,
                diagnostics_available=False,
                detail="fresh SDK read; motor fault bits and CAN ACK unavailable",
            ),
        )

    @staticmethod
    def _native_target(target: np.ndarray, calibration: ArmCalibration):
        joints = target[:6].astype(float)
        if np.any(joints < calibration.joint_min_rad) or np.any(
            joints > calibration.joint_max_rad
        ):
            raise ValueError("ARX joint command exceeds calibrated radian bounds")
        return joints.tolist(), calibration.to_native(float(target[6]))

    def send(self, target: np.ndarray, command_id: str) -> CommandReceipt:
        if self._closed:
            raise RuntimeError("ARX device is closed")
        positions = np.asarray(target, dtype=np.float32)
        if positions.shape != (14,) or not np.isfinite(positions).all():
            raise ValueError("ARX target must be finite 14D")
        # Validate both arms before writing either. SDK exceptions after the
        # first write remain uncertain and the gateway closes the episode.
        left_args = (
            self._native_target(positions[:7], self.left_calibration)
            if self.command_left else None
        )
        right_args = (
            self._native_target(positions[7:], self.right_calibration)
            if self.command_right else None
        )
        if left_args is not None:
            self.left.set_joint_positions(positions=left_args[0])
            self.left.set_gripper_pos(left_args[1])
        if right_args is not None:
            self.right.set_joint_positions(positions=right_args[0])
            self.right.set_gripper_pos(right_args[1])
        return CommandReceipt(
            command_id=command_id,
            sent_monotonic_ns=time.monotonic_ns(),
            status="sdk_call_returned",
            detail="SDK setter returned; no hardware acknowledgement exposed",
        )

    def close(self) -> None:
        # Official SingleArm has no documented close method. Do not issue a
        # potentially moving go_home/protect command as implicit cleanup.
        self._closed = True
