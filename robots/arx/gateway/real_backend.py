# Copyright (c) 2026 Zetta Contributors
"""Hardware-neutral, fail-closed 14D backend for the ARX gateway."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Protocol

import numpy as np

from robots.arx.control import ActionProcessor
from robots.arx.contracts import ArxTaskManifest

from .backend import HardwareEvidence, PolicyObservation, StepCommit
from .public import CAMERAS


@dataclass(frozen=True)
class CameraIdentity:
    name: str
    device_id: str
    calibration_id: str
    calibration_sha256: str
    width: int
    height: int

    def __post_init__(self):
        if not self.device_id or not self.calibration_id:
            raise ValueError("camera device and calibration IDs are required")
        if len(self.calibration_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.calibration_sha256
        ):
            raise ValueError("camera calibration SHA-256 is required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("invalid camera dimensions")


@dataclass(frozen=True)
class CameraFrame:
    pixels: np.ndarray
    monotonic_ns: int
    identity: CameraIdentity
    depth_mm: np.ndarray | None = None


@dataclass(frozen=True)
class DeviceHealth:
    transport_responsive: bool
    fault_codes: tuple[str, ...] = ()
    diagnostics_available: bool = False
    detail: str = ""


@dataclass(frozen=True)
class JointSample:
    positions: np.ndarray
    monotonic_ns: int
    acquisition_started_ns: int
    health: DeviceHealth
    right_tcp_xyz_m: np.ndarray | None = None


@dataclass(frozen=True)
class CommandReceipt:
    command_id: str
    sent_monotonic_ns: int
    status: Literal["sdk_call_returned", "ros_publish_returned", "transport_acknowledged"]
    detail: str = ""
    expected_feedback_target: tuple[float, ...] | None = None


class CameraSource(Protocol):
    def capture(self, timeout_s: float) -> Mapping[str, CameraFrame]: ...
    def close(self) -> None: ...


class ArmDevice(Protocol):
    """Adapters normalize their native hardware into the gateway's 14D units."""

    def read(self) -> JointSample: ...
    def send(self, target: np.ndarray, command_id: str) -> CommandReceipt: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class RealBackendConfig:
    cameras: tuple[CameraIdentity, CameraIdentity, CameraIdentity]
    # Each arm: six revolute joints in radians, then one calibrated gripper
    # position in the VLA model's raw gripper coordinate.
    state_units: tuple[str, ...]
    control_hz: float
    max_sensor_skew_ms: float
    max_sensor_age_ms: float
    observation_timeout_s: float
    arrival_timeout_s: float
    feedback_poll_s: float
    position_tolerance: tuple[float, ...]
    depth_cameras: tuple[str, ...] = ()

    def __post_init__(self):
        if tuple(camera.name for camera in self.cameras) != CAMERAS:
            raise ValueError("camera names/order must match the VLA model")
        if len(set(self.depth_cameras)) != len(self.depth_cameras) or not set(self.depth_cameras) <= set(CAMERAS):
            raise ValueError("depth cameras must be named RGB cameras")
        if len(self.state_units) != 14 or tuple(
            unit for i, unit in enumerate(self.state_units) if i not in (6, 13)
        ) != ("rad",) * 12:
            raise ValueError("12 arm joint units must be radians")
        if self.state_units[6] != "policy_gripper" or self.state_units[13] != "policy_gripper":
            raise ValueError("gripper units must be explicit calibrated policy coordinates")
        if len(self.position_tolerance) != 14 or any(
            not math.isfinite(value) or value <= 0 for value in self.position_tolerance
        ):
            raise ValueError("14 finite positive position tolerances required")
        for value in (
            self.control_hz, self.max_sensor_skew_ms, self.max_sensor_age_ms,
            self.observation_timeout_s, self.arrival_timeout_s, self.feedback_poll_s,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("real backend timing limits must be finite and positive")
        if self.max_sensor_skew_ms >= self.max_sensor_age_ms:
            raise ValueError("sensor skew limit must be below age limit")


class RealBackend:
    """Implements gateway Backend; no hardware connection is opened by importing it.

    Camera timestamps are host monotonic capture/dequeue times. The caller must
    supply a source that maps all three cameras to that clock and an ArmDevice
    that reports fresh feedback. No command is issued by reset().
    """

    def __init__(
        self, *, arms: ArmDevice, cameras: CameraSource,
        config: RealBackendConfig, task: ArxTaskManifest,
    ):
        self.arms, self.cameras, self.config, self.task = arms, cameras, config, task
        self._processor: ActionProcessor | None = None
        self._started_ns: int | None = None
        self._next_send_ns: int | None = None
        self._last_frame_ns = {name: -1 for name in CAMERAS}
        self._last_state_ns = -1
        self._event_sink = None
        self._closed = False

    def set_event_sink(self, sink):
        self._event_sink = sink

    def _emit(self, kind: str, payload: dict):
        if self._event_sink is not None:
            self._event_sink(kind, payload)

    def _observe(self, *, after_ns: int | None = None) -> tuple[PolicyObservation, dict, dict, int]:
        expected = {camera.name: camera for camera in self.config.cameras}
        deadline = time.monotonic() + self.config.observation_timeout_s
        last_error = "no synchronized frame set"
        while time.monotonic() < deadline:
            remaining = max(0.001, deadline - time.monotonic())
            frames = self.cameras.capture(remaining)
            try:
                sample = self.arms.read()
            except TimeoutError as exc:
                last_error = str(exc)
                time.sleep(0.002)
                continue
            now = time.monotonic_ns()
            if set(frames) != set(CAMERAS):
                raise ValueError("three named RGB frames are required")
            if not sample.health.transport_responsive or sample.health.fault_codes:
                raise RuntimeError("arm device is not healthy")
            state = np.asarray(sample.positions, dtype=np.float32)
            if state.shape != (14,) or not np.isfinite(state).all():
                raise ValueError("arm feedback must be finite 14D positions")
            times = [sample.acquisition_started_ns, sample.monotonic_ns]
            images = {}
            feature_frames = {}
            camera_times = {}
            depth_times = {}
            for name in CAMERAS:
                frame = frames[name]
                if frame.identity != expected[name]:
                    raise ValueError(f"camera identity/calibration mismatch: {name}")
                pixels = np.asarray(frame.pixels)
                if (
                    pixels.dtype != np.uint8
                    or pixels.shape != (expected[name].height, expected[name].width, 3)
                ):
                    raise ValueError(f"camera geometry/RGB dtype mismatch: {name}")
                if frame.monotonic_ns <= self._last_frame_ns[name]:
                    last_error = f"stale camera frame: {name}"
                    time.sleep(0.002)
                    break
                images[name] = pixels.copy()
                camera_times[name] = frame.monotonic_ns
                times.append(frame.monotonic_ns)
                if name in self.config.depth_cameras:
                    depth = np.asarray(frame.depth_mm)
                    if depth.dtype != np.uint16 or depth.shape != pixels.shape[:2]:
                        raise ValueError(f"aligned metric depth missing or invalid: {name}")
                    feature_frames[name.removesuffix("_rgb") + "_depth_mm"] = depth.copy()
                    depth_times[name.removesuffix("_rgb") + "_depth_mm"] = frame.monotonic_ns
            else:
                if sample.monotonic_ns <= self._last_state_ns:
                    last_error = "stale joint feedback"
                    time.sleep(0.002)
                    continue
                if sample.acquisition_started_ns > sample.monotonic_ns:
                    raise ValueError("invalid joint sample timestamp")
                if any(stamp <= 0 or stamp > now for stamp in times):
                    raise ValueError("sensor timestamps must use host monotonic time")
                if after_ns is not None and min(times) <= after_ns:
                    last_error = "post-command sensor sample unavailable"
                    time.sleep(0.002)
                    continue
                age_ms = (now - min(times)) / 1e6
                skew_ms = (max(times) - min(times)) / 1e6
                if age_ms > self.config.max_sensor_age_ms:
                    last_error = f"sensor age {age_ms:.1f} ms exceeds limit"
                    time.sleep(0.002)
                    continue
                if skew_ms > self.config.max_sensor_skew_ms:
                    last_error = f"sensor skew {skew_ms:.1f} ms exceeds limit"
                    time.sleep(0.002)
                    continue
                self._last_frame_ns.update(camera_times)
                self._last_state_ns = sample.monotonic_ns
                evidence = {
                    "clock_domain": "host_monotonic_ns",
                    "observation_completed_ns": now,
                    "measured_state": state.tolist(),
                    "state_monotonic_ns": sample.monotonic_ns,
                    "state_acquisition_started_ns": sample.acquisition_started_ns,
                    "camera_monotonic_ns": camera_times,
                    "depth_monotonic_ns": depth_times,
                    "camera_calibration_sha256": {
                        name: expected[name].calibration_sha256 for name in CAMERAS
                    },
                    "sensor_age_ms": age_ms,
                    "sensor_skew_ms": skew_ms,
                    "device_health": asdict(sample.health),
                    "camera_health": {
                        name: {"responsive": True, "device_id": expected[name].device_id}
                        for name in CAMERAS
                    },
                }
                if sample.right_tcp_xyz_m is not None:
                    tcp = np.asarray(sample.right_tcp_xyz_m, dtype=np.float64)
                    if tcp.shape != (3,) or not np.isfinite(tcp).all():
                        raise ValueError("right controller FK TCP must be finite XYZ metres")
                    evidence["right_tcp_xyz_m"] = tcp.tolist()
                    evidence["right_tcp_frame"] = "right_arm_local_base"
                    evidence["right_tcp_monotonic_ns"] = sample.monotonic_ns
                return PolicyObservation(images, state.copy()), evidence, feature_frames, now
        raise TimeoutError(last_error)

    def reset(self) -> StepCommit:
        if self._started_ns is not None or self._closed:
            raise RuntimeError("real backend cannot reset an existing episode")
        policy, observed, feature_frames, now = self._observe()
        start = np.asarray(self.task.start_state, dtype=np.float32).copy()
        start[[6, 13]] += np.asarray(
            self.task.control.gripper_command_offsets, dtype=np.float32
        )
        tolerance = np.asarray(self.config.position_tolerance, dtype=np.float32)
        mismatched = np.flatnonzero(np.abs(policy.state - start) > tolerance)
        if mismatched.size:
            raise ValueError(
                "real arm differs from frozen task start state at channels "
                + ",".join(str(int(index)) for index in mismatched)
            )
        command_state = policy.state.copy()
        command_state[[6, 13]] -= np.asarray(
            self.task.control.gripper_command_offsets, dtype=np.float32
        )
        self._processor = ActionProcessor(self.task, command_state)
        self._started_ns = now
        self._next_send_ns = now
        hardware = HardwareEvidence(
            observation=observed, command_receipt=None, arrival_verified=None,
            feature_frames=feature_frames,
        )
        return StepCommit(policy, policy.state.copy(), 0.0, False, {}, hardware=hardware)

    def step(self, raw_target: np.ndarray) -> StepCommit:
        if self._started_ns is None or self._closed:
            raise RuntimeError("real backend is not ready")
        raw = np.asarray(raw_target, dtype=np.float32)
        if raw.shape != (14,) or not np.isfinite(raw).all():
            raise ValueError("real target must be finite 14D positions")
        if self._processor is None:
            raise RuntimeError("real task control processor is unavailable")
        target, _ = self._processor.process(raw)
        now = time.monotonic_ns()
        if self._next_send_ns is not None and now < self._next_send_ns:
            time.sleep((self._next_send_ns - now) / 1e9)
        command_id = "cmd-" + uuid.uuid4().hex
        self._emit("command_dispatch_started", {
            "command_id": command_id, "target": target.tolist(),
            "monotonic_ns": time.monotonic_ns(),
        })
        receipt = self.arms.send(target.copy(), command_id)
        if receipt.command_id != command_id or receipt.sent_monotonic_ns <= 0:
            raise ValueError("arm device returned invalid command receipt")
        arrival_target = np.asarray(
            receipt.expected_feedback_target if receipt.expected_feedback_target is not None
            else target, dtype=np.float32,
        )
        if arrival_target.shape != (14,) or not np.isfinite(arrival_target).all():
            raise ValueError("arm device returned invalid expected feedback target")
        self._emit("command_sent", {
            "command_id": command_id, "target": target.tolist(),
            "expected_feedback_target": arrival_target.tolist(),
            "receipt": asdict(receipt),
        })
        period_ns = round(1e9 / self.config.control_hz)
        self._next_send_ns = max(time.monotonic_ns(), receipt.sent_monotonic_ns) + period_ns
        deadline = time.monotonic() + self.config.arrival_timeout_s
        tolerances = np.asarray(self.config.position_tolerance, dtype=np.float32)
        # The final synchronized feedback, rather than SDK return, decides arrival.
        policy, observed, feature_frames, observation_ns = self._observe(after_ns=receipt.sent_monotonic_ns)
        arrived = (
            observed["state_monotonic_ns"] > receipt.sent_monotonic_ns
            and bool(np.all(np.abs(policy.state - arrival_target) <= tolerances))
        )
        while not arrived and time.monotonic() < deadline:
            time.sleep(min(self.config.feedback_poll_s, max(0, deadline - time.monotonic())))
            policy, observed, feature_frames, observation_ns = self._observe(after_ns=receipt.sent_monotonic_ns)
            arrived = (
                observed["state_monotonic_ns"] > receipt.sent_monotonic_ns
                and bool(np.all(np.abs(policy.state - arrival_target) <= tolerances))
            )
        observed["position_error"] = (policy.state - arrival_target).tolist()
        observed["expected_feedback_target"] = arrival_target.tolist()
        self._emit("arrival_observed" if arrived else "arrival_unverified", {
            "command_id": command_id,
            "state_monotonic_ns": observed["state_monotonic_ns"],
            "measured_state": policy.state.tolist(),
            "position_error": observed["position_error"],
            "verified": arrived,
        })
        hardware = HardwareEvidence(
            observation=observed,
            command_receipt=asdict(receipt),
            arrival_verified=arrived,
            feature_frames=feature_frames,
        )
        return StepCommit(
            policy, target.copy(), (observation_ns - self._started_ns) / 1e9,
            False, {}, hardware=hardware,
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.cameras.close()
            finally:
                self.arms.close()
