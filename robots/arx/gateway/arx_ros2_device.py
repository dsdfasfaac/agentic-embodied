# Copyright (c) 2026 Zetta Contributors
"""ARX X5 ROS2 remote_slave adapter for the controllers already running on dodo."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .arx_x5_device import ArmCalibration
from .real_backend import CommandReceipt, DeviceHealth, JointSample


@dataclass(frozen=True)
class Ros2Topics:
    left_status: str = "/arm_slave_l_status"
    right_status: str = "/arm_slave_r_status"
    left_command: str = "/arm_master_l_status"
    right_command: str = "/arm_master_r_status"


class ArxRos2Device:
    """14D ArmDevice over RobotStatus topics; ROS publish is not hardware ACK."""

    def __init__(
        self, *, node: Any, message_type: type, topics: Ros2Topics,
        left_calibration: ArmCalibration, right_calibration: ArmCalibration,
        command_left: bool, command_right: bool,
        max_status_age_ms: float = 100,
        max_pair_skew_ms: float = 30,
        keepalive_hz: float = 30,
        command_lease_s: float = 0.25,
        cleanup=None,
    ):
        if not (command_left or command_right):
            raise ValueError("at least one arm must be commandable")
        if min(max_status_age_ms, max_pair_skew_ms, keepalive_hz, command_lease_s) <= 0:
            raise ValueError("ROS2 timing limits must be positive")
        self.node, self.message_type, self.topics = node, message_type, topics
        self.left_calibration, self.right_calibration = left_calibration, right_calibration
        self.command_left, self.command_right = command_left, command_right
        self.max_status_age_ns = round(max_status_age_ms * 1e6)
        self.max_pair_skew_ns = round(max_pair_skew_ms * 1e6)
        self.keepalive_period_s = 1 / keepalive_hz
        self.command_lease_ns = round(command_lease_s * 1e9)
        self._cleanup = cleanup
        self._lock = threading.Lock()
        self._latest = {"left": None, "right": None}
        self._fault = None
        self._last_messages = None
        self._last_send_ns = 0
        self._closed = threading.Event()
        self._publishers = {
            "left": node.create_publisher(message_type, topics.left_command, 10),
            "right": node.create_publisher(message_type, topics.right_command, 10),
        }
        node.create_subscription(
            message_type, topics.left_status, lambda msg: self._on_status("left", msg), 10
        )
        node.create_subscription(
            message_type, topics.right_status, lambda msg: self._on_status("right", msg), 10
        )
        self._keepalive = threading.Thread(
            target=self._keepalive_loop, name="arx-ros2-command-lease", daemon=True
        )
        self._keepalive.start()

    @classmethod
    def from_ros2(cls, **kwargs) -> "ArxRos2Device":
        import rclpy
        from arx5_arm_msg.msg import RobotStatus
        from rclpy.executors import SingleThreadedExecutor

        owned_context = not rclpy.ok()
        if owned_context:
            rclpy.init()
        node = rclpy.create_node("zetta_arx_real_backend")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        stop = threading.Event()

        def spin():
            while not stop.is_set() and rclpy.ok():
                executor.spin_once(timeout_sec=0.05)

        thread = threading.Thread(target=spin, name="arx-ros2-status", daemon=True)
        thread.start()

        def cleanup():
            stop.set()
            thread.join(timeout=1.0)
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()
            if owned_context and rclpy.ok():
                rclpy.shutdown()

        try:
            return cls(node=node, message_type=RobotStatus, cleanup=cleanup, **kwargs)
        except BaseException:
            cleanup()
            raise

    def _on_status(self, side: str, message):
        stamp = time.monotonic_ns()
        calibration = self.left_calibration if side == "left" else self.right_calibration
        try:
            native = np.asarray(message.joint_pos, dtype=np.float64)
            if native.shape != (7,) or not np.isfinite(native).all():
                raise ValueError("ROS2 status must contain finite 7D positions")
            if not (
                calibration.gripper_native_min - 1e-3
                <= native[6]
                <= calibration.gripper_native_max + 1e-3
            ):
                raise ValueError("ROS2 gripper feedback outside calibrated range")
            if np.any(native[:6] < np.asarray(calibration.joint_min_rad) - 1e-3) or np.any(
                native[:6] > np.asarray(calibration.joint_max_rad) + 1e-3
            ):
                raise ValueError("ROS2 joint feedback outside configured controller-coordinate bounds")
            end_pose = np.asarray(message.end_pos, dtype=np.float64)
            if end_pose.shape != (6,) or not np.isfinite(end_pose).all():
                raise ValueError("ROS2 controller FK end_pos must be finite 6D")
            currents = np.asarray(message.joint_cur, dtype=np.float64)
            if currents.shape != (7,) or not np.isfinite(currents).all():
                raise ValueError("ROS2 status must contain finite 7D motor currents")
            native[6] = calibration.to_policy(float(native[6]))
        except Exception as exc:
            with self._lock:
                self._fault = f"{side} status invalid: {exc}"
            return
        with self._lock:
            self._latest[side] = (native.astype(np.float32), stamp, end_pose[:3].copy(), currents.copy())

    def read(self) -> JointSample:
        if self._closed.is_set():
            raise RuntimeError("ARX ROS2 device is closed")
        with self._lock:
            left, right = self._latest["left"], self._latest["right"]
            fault = self._fault
        if fault:
            raise RuntimeError(fault)
        if left is None or right is None:
            raise TimeoutError("waiting for both ARX ROS2 status topics")
        now = time.monotonic_ns()
        stamps = (left[1], right[1])
        if now - min(stamps) > self.max_status_age_ns:
            raise TimeoutError("ARX ROS2 joint status is stale")
        if abs(stamps[0] - stamps[1]) > self.max_pair_skew_ns:
            raise TimeoutError("left/right ARX ROS2 status is unsynchronized")
        return JointSample(
            np.concatenate((left[0], right[0])), max(stamps), min(stamps),
            DeviceHealth(
                transport_responsive=True, diagnostics_available=False,
                detail="fresh ROS2 RobotStatus; no CAN ACK or motor fault bits",
            ),
            right_tcp_xyz_m=right[2],
            auxiliary_feedback={"right_gripper_current_native": float(right[3][6])},
        )

    def _message(self, positions: np.ndarray):
        message = self.message_type()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.end_pos = [0.0] * 6
        message.joint_pos = [float(value) for value in positions]
        message.joint_vel = [0.0] * 7
        message.joint_cur = [0.0] * 7
        return message

    @staticmethod
    def _native_target(policy: np.ndarray, calibration: ArmCalibration) -> np.ndarray:
        joints = policy[:6].astype(np.float64)
        if np.any(joints < calibration.joint_min_rad) or np.any(
            joints > calibration.joint_max_rad
        ):
            raise ValueError("ARX ROS2 joint command exceeds calibrated radian bounds")
        result = np.asarray(policy, dtype=np.float64).copy()
        result[6] = calibration.to_native_command(float(policy[6]))
        return result

    def send(self, target: np.ndarray, command_id: str) -> CommandReceipt:
        if self._closed.is_set():
            raise RuntimeError("ARX ROS2 device is closed")
        positions = np.asarray(target, dtype=np.float32)
        if positions.shape != (14,) or not np.isfinite(positions).all():
            raise ValueError("ARX ROS2 target must be finite 14D")
        # Validate both arms before publishing either to avoid avoidable partial writes.
        messages = {}
        if self.command_left:
            messages["left"] = self._message(
                self._native_target(positions[:7], self.left_calibration)
            )
        if self.command_right:
            messages["right"] = self._message(
                self._native_target(positions[7:], self.right_calibration)
            )
        with self._lock:
            for side, message in messages.items():
                self._publishers[side].publish(message)
            stamp = time.monotonic_ns()
            self._last_messages = messages
            self._last_send_ns = stamp
        return CommandReceipt(
            command_id, stamp, "ros_publish_returned",
            "ROS2 publisher returned; controller/CAN receipt unavailable",
            expected_feedback_target=tuple(float(x) for x in (
                positions + np.asarray(
                    [0.0] * 6 + [self.left_calibration.gripper_command_offset
                                   * self.left_calibration.gripper_policy_scale
                                   if self.command_left else 0.0]
                    + [0.0] * 6 + [self.right_calibration.gripper_command_offset
                                   * self.right_calibration.gripper_policy_scale
                                   if self.command_right else 0.0],
                    dtype=np.float32,
                )
            )),
        )

    def _keepalive_loop(self):
        while not self._closed.wait(self.keepalive_period_s):
            with self._lock:
                if (
                    self._last_messages is None
                    or time.monotonic_ns() - self._last_send_ns > self.command_lease_ns
                ):
                    continue
                for side, message in self._last_messages.items():
                    self._publishers[side].publish(
                        self._message(np.asarray(message.joint_pos, dtype=np.float64))
                    )

    def close(self):
        if not self._closed.is_set():
            self._closed.set()
            self._keepalive.join(timeout=1.0)
            if self._cleanup is not None:
                self._cleanup()
