# Copyright (c) 2026 Zetta Contributors
"""Software-synchronized three-view OpenCV source for the real backend."""

from __future__ import annotations

import hashlib
import json
import math
import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .public import CAMERAS
from .real_backend import CameraFrame, CameraIdentity


@dataclass(frozen=True)
class OpenCVCameraSpec:
    identity: CameraIdentity
    device: int | str
    calibration_file: Path
    capture_fps: float


class OpenCVCameraSource:
    """Capture continuously; select a bounded-skew RGB trio by host dequeue time.

    This is software synchronization. Hardware-triggered cameras can implement
    CameraSource and return exposure times mapped to the same monotonic clock.
    """

    def __init__(self, specs: tuple[OpenCVCameraSpec, ...], *, max_skew_ms: float):
        if tuple(spec.identity.name for spec in specs) != CAMERAS:
            raise ValueError("three ordered ARX camera specs are required")
        if max_skew_ms <= 0:
            raise ValueError("camera skew limit must be positive")
        import cv2

        self._cv2 = cv2
        self._max_skew_ns = round(max_skew_ms * 1e6)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._buffers = {name: deque(maxlen=8) for name in CAMERAS}
        self._errors = {}
        self._captures = {}
        self._threads = []
        try:
            for spec in specs:
                if str(spec.device) != spec.identity.device_id:
                    raise ValueError(f"OpenCV device ID differs from path: {spec.identity.name}")
                digest = hashlib.sha256(Path(spec.calibration_file).read_bytes()).hexdigest()
                if digest != spec.identity.calibration_sha256:
                    raise ValueError(f"camera calibration SHA mismatch: {spec.identity.name}")
                cap = cv2.VideoCapture(spec.device)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f"camera cannot open: {spec.identity.name}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.identity.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.identity.height)
                cap.set(cv2.CAP_PROP_FPS, spec.capture_fps)
                self._captures[spec.identity.name] = cap
            for spec in specs:
                thread = threading.Thread(
                    target=self._read_loop, args=(spec,), daemon=True,
                    name="camera-" + spec.identity.name,
                )
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.close()
            raise

    def _read_loop(self, spec: OpenCVCameraSpec):
        name = spec.identity.name
        cap = self._captures[name]
        while not self._closed.is_set():
            ok, bgr = cap.read()
            stamp = time.monotonic_ns()
            if not ok or bgr is None:
                with self._lock:
                    self._errors[name] = "OpenCV capture failed"
                return
            if bgr.shape != (spec.identity.height, spec.identity.width, 3):
                with self._lock:
                    self._errors[name] = "camera returned unexpected geometry"
                return
            rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
            if rgb.dtype != np.uint8:
                with self._lock:
                    self._errors[name] = "camera returned non-uint8 RGB"
                return
            with self._lock:
                self._buffers[name].append(
                    CameraFrame(rgb.copy(), stamp, spec.identity)
                )

    def capture(self, timeout_s: float) -> dict[str, CameraFrame]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._closed.is_set():
            with self._lock:
                if self._errors:
                    raise RuntimeError(f"camera health failure: {self._errors}")
                buffers = tuple(tuple(self._buffers[name]) for name in CAMERAS)
            if all(buffers):
                candidates = (
                    frames for frames in itertools.product(*buffers)
                    if max(frame.monotonic_ns for frame in frames)
                    - min(frame.monotonic_ns for frame in frames)
                    <= self._max_skew_ns
                )
                selected = max(
                    candidates,
                    key=lambda frames: min(frame.monotonic_ns for frame in frames),
                    default=None,
                )
                if selected is not None:
                    return dict(zip(CAMERAS, selected))
            time.sleep(0.002)
        raise TimeoutError("three synchronized camera frames unavailable")

    def close(self) -> None:
        if not self._closed.is_set():
            self._closed.set()
            for cap in self._captures.values():
                cap.release()
            for thread in self._threads:
                thread.join(timeout=0.2)


@dataclass(frozen=True)
class RealSenseCameraSpec:
    identity: CameraIdentity
    serial: str
    calibration_file: Path
    capture_width: int
    capture_height: int
    capture_fps: int
    depth_enabled: bool = False


class RealSenseCameraSource:
    """Three D405 RGB streams selected by serial, synchronized by host dequeue time."""

    def __init__(self, specs: tuple[RealSenseCameraSpec, ...], *, max_skew_ms: float):
        if tuple(spec.identity.name for spec in specs) != CAMERAS:
            raise ValueError("three ordered RealSense camera specs are required")
        if len({spec.serial for spec in specs}) != 3:
            raise ValueError("RealSense camera serials must be unique")
        if max_skew_ms <= 0:
            raise ValueError("camera skew limit must be positive")
        import cv2
        import pyrealsense2 as rs

        self._cv2, self._rs = cv2, rs
        self._max_skew_ns = round(max_skew_ms * 1e6)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._buffers = {name: deque(maxlen=8) for name in CAMERAS}
        self._errors = {}
        self._pipelines = {}
        self._aligners = {}
        self._depth_scales = {}
        self._threads = []
        try:
            for spec in specs:
                if spec.identity.device_id != spec.serial:
                    raise ValueError(f"RealSense device ID must equal serial: {spec.identity.name}")
                digest = hashlib.sha256(Path(spec.calibration_file).read_bytes()).hexdigest()
                if digest != spec.identity.calibration_sha256:
                    raise ValueError(f"camera calibration SHA mismatch: {spec.identity.name}")
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(spec.serial)
                config.enable_stream(
                    rs.stream.color, spec.capture_width, spec.capture_height,
                    rs.format.rgb8, spec.capture_fps,
                )
                if spec.depth_enabled:
                    config.enable_stream(
                        rs.stream.depth, spec.capture_width, spec.capture_height,
                        rs.format.z16, spec.capture_fps,
                    )
                profile = pipeline.start(config)
                self._pipelines[spec.identity.name] = pipeline
                if spec.depth_enabled:
                    self._aligners[spec.identity.name] = rs.align(rs.stream.color)
                    scale = profile.get_device().first_depth_sensor().get_depth_scale()
                    if not math.isfinite(scale) or scale <= 0:
                        raise ValueError(f"RealSense depth scale invalid: {spec.identity.name}")
                    self._depth_scales[spec.identity.name] = scale
                calibration = json.loads(Path(spec.calibration_file).read_text())
                expected = calibration.get("camera", {}).get("intrinsics")
                if not isinstance(expected, dict):
                    raise ValueError(f"RealSense RGB intrinsics missing: {spec.identity.name}")
                active = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                if (active.width, active.height) != (
                    spec.capture_width, spec.capture_height
                ):
                    raise ValueError(f"RealSense profile geometry changed: {spec.identity.name}")
                for key in ("fx", "fy", "ppx", "ppy"):
                    if not math.isclose(
                        float(expected[key]), float(getattr(active, key)),
                        rel_tol=0, abs_tol=1e-3,
                    ):
                        raise ValueError(f"RealSense intrinsics changed: {spec.identity.name}/{key}")
                if not np.allclose(
                    np.asarray(expected["coeffs"], dtype=float),
                    np.asarray(active.coeffs, dtype=float),
                    rtol=0, atol=1e-5,
                ):
                    raise ValueError(f"RealSense distortion changed: {spec.identity.name}")
                if str(expected["distortion_model"]).split(".")[-1] != str(active.model).split(".")[-1]:
                    raise ValueError(f"RealSense distortion model changed: {spec.identity.name}")
            for spec in specs:
                thread = threading.Thread(
                    target=self._read_loop, args=(spec,), daemon=True,
                    name="realsense-" + spec.identity.name,
                )
                thread.start()
                self._threads.append(thread)
        except BaseException:
            self.close()
            raise

    def _read_loop(self, spec: RealSenseCameraSpec):
        name = spec.identity.name
        pipeline = self._pipelines[name]
        while not self._closed.is_set():
            try:
                frames = pipeline.wait_for_frames(1000)
                if spec.depth_enabled:
                    frames = self._aligners[name].process(frames)
                color = frames.get_color_frame()
                stamp = time.monotonic_ns()
                if not color:
                    raise RuntimeError("missing RGB frame")
                rgb = np.asarray(color.get_data())
                if rgb.shape != (spec.capture_height, spec.capture_width, 3):
                    raise ValueError("RealSense returned unexpected RGB geometry")
                if (
                    spec.capture_width != spec.identity.width
                    or spec.capture_height != spec.identity.height
                ):
                    rgb = self._cv2.resize(
                        rgb, (spec.identity.width, spec.identity.height),
                        interpolation=self._cv2.INTER_AREA,
                    )
                if rgb.dtype != np.uint8:
                    raise ValueError("RealSense returned non-uint8 RGB")
                depth_mm = None
                if spec.depth_enabled:
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        raise RuntimeError("missing aligned metric depth frame")
                    raw = np.asarray(depth_frame.get_data())
                    if raw.dtype != np.uint16 or raw.shape != (spec.capture_height, spec.capture_width):
                        raise ValueError("RealSense depth has unexpected format or geometry")
                    scaled = np.rint(raw.astype(np.float32) * (1000 * self._depth_scales[name]))
                    if np.any(scaled > 65535):
                        raise ValueError("RealSense depth exceeds uint16 millimetres")
                    depth_mm = scaled.astype(np.uint16)
                    if (spec.capture_width, spec.capture_height) != (spec.identity.width, spec.identity.height):
                        depth_mm = self._cv2.resize(
                            depth_mm, (spec.identity.width, spec.identity.height),
                            interpolation=self._cv2.INTER_NEAREST,
                        )
                with self._lock:
                    self._buffers[name].append(
                        CameraFrame(np.ascontiguousarray(rgb).copy(), stamp, spec.identity,
                                    None if depth_mm is None else np.ascontiguousarray(depth_mm).copy())
                    )
            except Exception as exc:
                if not self._closed.is_set():
                    with self._lock:
                        self._errors[name] = str(exc)
                return

    def capture(self, timeout_s: float) -> dict[str, CameraFrame]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._closed.is_set():
            with self._lock:
                if self._errors:
                    raise RuntimeError(f"RealSense camera health failure: {self._errors}")
                buffers = tuple(tuple(self._buffers[name]) for name in CAMERAS)
            if all(buffers):
                candidates = (
                    frames for frames in itertools.product(*buffers)
                    if max(frame.monotonic_ns for frame in frames)
                    - min(frame.monotonic_ns for frame in frames)
                    <= self._max_skew_ns
                )
                selected = max(
                    candidates,
                    key=lambda frames: min(frame.monotonic_ns for frame in frames),
                    default=None,
                )
                if selected is not None:
                    return dict(zip(CAMERAS, selected))
            time.sleep(0.002)
        raise TimeoutError("three synchronized RealSense RGB frames unavailable")

    def close(self) -> None:
        if not self._closed.is_set():
            self._closed.set()
            for pipeline in self._pipelines.values():
                pipeline.stop()
            for thread in self._threads:
                thread.join(timeout=0.2)
