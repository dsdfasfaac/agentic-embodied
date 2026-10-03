"""Metric PickTube observer and fail-closed depth tests."""

from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from robots.arx.deployment.picktube_rgbd_provider import (
    FRONT_INTRINSICS_SHA256, PickTubeRgbdProvider,
)


def test_picktube_distance_uses_aligned_depth_extrinsic_and_controller_fk():
    pytest.importorskip("cv2")
    pytest.importorskip("pyrealsense2")
    root = Path(__file__).resolve().parents[1]
    provider = PickTubeRgbdProvider()
    front = SimpleNamespace(
        name="front_rgb", serial="260422272500", depth_enabled=True,
        capture_width=640, capture_height=480, width=320, height=240,
        calibration_sha256=FRONT_INTRINSICS_SHA256,
        calibration_file=root / "robots/arx/manifests/real/dodo_front_rgb_d405_intrinsics.json",
    )
    provider.validate_hardware(SimpleNamespace(
        cameras=[front], right_gripper_closed_policy=0.0,
        right_gripper_open_policy=-3.4,
    ))
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[40:52, 150:164] = [230, 70, 150]
    depth = np.zeros((240, 320), dtype=np.uint16)
    depth[40:52, 150:164] = 500
    point_camera = provider._deproject(156.5, 45.5, 0.5)
    point_left = (provider.transform @ np.r_[point_camera, 1])[:3]
    right_tcp = point_left - [0, -0.5, 0] - [0.2, 0, 0]
    state = np.zeros(14)
    stamp = time.monotonic_ns()
    obs = {"hardware": {
        "measured_state": state.tolist(),
        "state_monotonic_ns": stamp,
        "right_tcp_monotonic_ns": stamp,
        "right_tcp_xyz_m": right_tcp.tolist(),
        "right_tcp_frame": "right_arm_local_base",
    }}
    result = provider.observe(obs, {"front_rgb": rgb, "front_depth_mm": depth})
    assert result["privileged.interaction.gripper_closed"] is True
    assert result["privileged.selected.target_gripper_distance_m"] == pytest.approx(0.2, abs=0.005)
    with pytest.raises(ValueError, match="insufficient valid metric depth"):
        provider.observe(obs, {"front_rgb": rgb, "front_depth_mm": np.zeros_like(depth)})
    with pytest.raises(ValueError, match="controller FK"):
        provider.observe({"hardware": {**obs["hardware"], "right_tcp_frame": "unknown"}},
                         {"front_rgb": rgb, "front_depth_mm": depth})
    with pytest.raises(ValueError, match="pinned 640x480"):
        provider.validate_hardware(SimpleNamespace(
            cameras=[SimpleNamespace(**{**front.__dict__, "depth_enabled": False})],
            right_gripper_closed_policy=0.0, right_gripper_open_policy=-3.4,
        ))


def test_large_pink_distractor_is_not_a_tracked_tube():
    pytest.importorskip("cv2")
    provider = PickTubeRgbdProvider()
    provider.last_centre = (170.0, 80.0)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[100:170, 130:210] = [230, 70, 150]
    with pytest.raises(ValueError, match="not reliably visible"):
        provider._pink_component(rgb)
