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
    rgb[70:105, 110:220] = [240, 240, 10]
    rgb[40:52, 150:164] = [230, 70, 150]
    depth = np.zeros((240, 320), dtype=np.uint16)
    depth[40:52, 150:164] = 500
    point_camera = provider._deproject(156.5, 45.5, 0.5)
    point_left = (provider.transform @ np.r_[point_camera, 1])[:3]
    state = np.zeros(14)
    controller_ee, _, _ = provider.controller_fk.fk(state[7:13])
    tool_centre, _, _ = provider.tool_fk.fk(state[7:13])
    expected_distance = np.linalg.norm(point_left - (tool_centre + [0, -0.5, 0]))
    stamp = time.monotonic_ns()
    def observation(measured=state, offset_ns=0):
        frame_stamp = time.monotonic_ns() + offset_ns
        return {"hardware": {
            "measured_state": measured.tolist(), "state_monotonic_ns": frame_stamp,
            "right_tcp_monotonic_ns": frame_stamp, "right_tcp_xyz_m": controller_ee.tolist(),
            "right_tcp_frame": "right_arm_local_base",
            "camera_monotonic_ns": {"front_rgb": frame_stamp},
            "depth_monotonic_ns": {"front_depth_mm": frame_stamp},
            "auxiliary_feedback": {"right_gripper_current_native": 0.07},
            "auxiliary_monotonic_ns": {"right_gripper_current_native": frame_stamp},
        }}
    obs = observation()
    result = provider.observe(obs, {"front_rgb": rgb, "front_depth_mm": depth})
    assert result["privileged.interaction.gripper_closed"] is True
    assert result["privileged.interaction.gripper_contact"] is False
    assert result["privileged.interaction.lift_m"] == 0.0
    assert result["privileged.interaction.grasped"] is False
    assert result["privileged.interaction.success"] is False
    assert result["privileged.selected.target_gripper_distance_m"] == pytest.approx(expected_distance, abs=0.005)
    settled_state = state.copy()
    settled_state[13] = -0.858
    settled = provider.observe(observation(settled_state), {"front_rgb": rgb, "front_depth_mm": depth})
    assert settled["privileged.interaction.gripper_closed"] is True
    from robots.arx.deployment.feature_observation import FeatureObservationUnavailable
    with pytest.raises(FeatureObservationUnavailable, match="insufficient valid metric depth"):
        provider.observe(observation(), {"front_rgb": rgb, "front_depth_mm": np.zeros_like(depth)})
    open_state = state.copy()
    open_state[13] = -2.5
    with pytest.raises(FeatureObservationUnavailable) as failure:
        provider.observe(observation(open_state), {"front_rgb": rgb, "front_depth_mm": np.zeros_like(depth)})
    assert failure.value.available["privileged.interaction.gripper_closed"] is False
    assert "privileged.interaction.success" in failure.value.unavailable
    assert provider.success_hold_frames == 0
    for bad, message in (({"right_tcp_frame": "unknown"}, "controller FK"),
                         ({"right_tcp_xyz_m": [1, 1, 1]}, "differs from fresh"),
                         ({"auxiliary_feedback": {}}, "fresh right gripper")):
        invalid = observation()
        invalid["hardware"].update(bad)
        with pytest.raises(ValueError, match=message):
            provider.observe(invalid, {"front_rgb": rgb, "front_depth_mm": depth})
    with pytest.raises(ValueError, match="pinned 640x480"):
        provider.validate_hardware(SimpleNamespace(
            cameras=[SimpleNamespace(**{**front.__dict__, "depth_enabled": False})],
            right_gripper_closed_policy=0.0, right_gripper_open_policy=-3.4,
        ))


def test_picktube_interaction_tracks_load_lift_retention_and_success():
    provider = PickTubeRgbdProvider()
    initial = np.array([0.0, 0.0, 0.0])
    open_values = provider._interaction_values(initial, initial, 0.0, False, 0.07)
    assert open_values["privileged.interaction.gripper_contact"] is False
    loaded = provider._interaction_values(initial, initial, 0.0, True, 0.20)
    assert loaded["privileged.interaction.gripper_contact"] is True
    for frame in range(5):
        position = np.array([0.0, 0.0, 0.012])
        values = provider._interaction_values(position, position, 0.0, True, 0.07)
        assert values["privileged.interaction.grasped"] is True
        assert values["privileged.interaction.success"] is (frame == 4)
    lost = PickTubeRgbdProvider()
    lost._interaction_values(initial, initial, 0.0, False, 0.07)
    lost._interaction_values(initial, initial, 0.0, True, 0.20)
    lifted = np.array([0.0, 0.0, 0.008])
    lost._interaction_values(lifted, lifted, 0.0, True, 0.07)
    slipped = lost._interaction_values(lifted, np.array([0.0, 0.0, 0.022]),
                                       0.014, True, 0.20)
    assert slipped["privileged.interaction.lift_m"] == pytest.approx(0.008)
    assert slipped["privileged.interaction.gripper_contact"] is False
    assert slipped["privileged.interaction.grasped"] is False
    assert slipped["privileged.interaction.success"] is False


def test_large_pink_distractor_is_not_a_tracked_tube():
    pytest.importorskip("cv2")
    provider = PickTubeRgbdProvider()
    provider.last_centre = (170.0, 80.0)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[70:95, 100:220] = [240, 240, 10]
    rgb[100:170, 130:210] = [230, 70, 150]
    with pytest.raises(ValueError, match="not reliably visible"):
        provider._pink_component(rgb)


def test_rack_context_selects_pale_tube_over_pink_sticker():
    pytest.importorskip("cv2")
    provider = PickTubeRgbdProvider()
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, :] = [10, 35, 35]
    rgb[70:105, 110:220] = [240, 240, 10]
    rgb[45:56, 145:156] = [130, 125, 140]
    rgb[59:71, 276:298] = [200, 80, 140]
    mask = provider._pink_component(rgb)
    yy, xx = np.nonzero(mask)
    assert 145 <= np.median(xx) <= 156
    assert 45 <= np.median(yy) <= 56


def test_tracked_label_survives_rack_occlusion_but_not_label_loss():
    pytest.importorskip("cv2")
    provider = PickTubeRgbdProvider()
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[70:105, 110:220] = [240, 240, 10]
    rgb[45:56, 145:156] = [130, 125, 140]
    provider._pink_component(rgb)
    # Physical pregrasp obscures the yellow rack, not the target label.
    rgb[70:105, 110:220] = 0
    mask = provider._pink_component(rgb)
    assert mask[45:56, 145:156].all()
    with pytest.raises(ValueError, match="rack is not visible"):
        PickTubeRgbdProvider()._pink_component(rgb)
    rgb[45:56, 145:156] = 0
    with pytest.raises(ValueError, match="pink tube label is not reliably visible"):
        provider._pink_component(rgb)
