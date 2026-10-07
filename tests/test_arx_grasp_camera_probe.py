"""Camera-only replay binds images to their audit and preserves model TCP frames."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from scripts.deployment import probe_arx_grasp_camera as camera_probe


def fixture(tmp_path, monkeypatch):
    rgb = np.zeros((240, 320, 3), np.uint8)
    depth = np.full((240, 320), 400, np.uint16)
    cv2.imwrite(str(tmp_path / "front_rgb.png"), rgb)
    cv2.imwrite(str(tmp_path / "front_depth_mm.png"), depth)
    sample = {"pink_label": {"found": True}, "sensor_skew_ms": 20., "sensor_age_ms": 30.,
              "host_monotonic_ns": {"front_rgb": 123}, "saved_images": {
                  name: {"sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
                  for name in ("front_rgb.png", "front_depth_mm.png")}}
    audit = {"schema_version": "arx.live.camera.audit.v1",
             "front_extrinsic_sha256": camera_probe.CALIBRATION_SHA256,
             "cameras": [{"name": "front_rgb", "serial": camera_probe.FRONT_SERIAL,
                          "rgb_intrinsics_sha256": camera_probe.FRONT_INTRINSICS_SHA256}],
             "observations": [sample]}
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))

    class Provider:
        transform = np.eye(4)

        def validate_hardware(self, config):
            pass

        def _pink_component(self, rgb):
            mask = np.zeros((240, 320), bool)
            mask[30:40, 150:156] = True
            return mask

        def _deproject(self, x, y, z):
            return np.array([x / 1000., y / 1000., z])

    class Service:
        def __init__(self, *args, **kwargs):
            tcp = np.eye(4)
            tcp[2, 3] = .195
            self.last_evidence = {"health": {"transform_grasp_from_model_tcp": tcp.tolist()}}

        def propose(self, cloud, **kwargs):
            pose = np.eye(4)
            pose[:3, 3] = cloud.target_camera_m - [0., 0., .195]
            return [{"transform_camera": pose.tolist(), "score": .8}]

    monkeypatch.setattr(camera_probe, "PickTubeRgbdProvider", Provider)
    monkeypatch.setattr(camera_probe, "LocalGraspService", Service)
    return path, audit


def test_replay_distinguishes_grasp_origin_from_finger_centre(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    result = camera_probe.probe(path, tmp_path, "http://127.0.0.1:18093")
    candidate = result["candidates"][0]
    assert candidate["model_tcp_target_distance_m"] == pytest.approx(0.)
    assert candidate["transform_base"][2][3] == pytest.approx(.205)
    assert candidate["model_tcp_base_xyz_m"][2] == pytest.approx(.4)
    assert result["target_evidence"]["object_point_count"] == 60
    assert result["robot_commands_sent"] is False
    assert result["robot_status_used"] is False
    assert result["live_motion_eligible"] is False
    assert result["ik_review_performed"] is False


def test_replay_rejects_changed_depth_and_stale_capture(tmp_path, monkeypatch):
    path, audit = fixture(tmp_path, monkeypatch)
    audit["observations"][0]["sensor_age_ms"] = 200
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="freshness"):
        camera_probe.probe(path, tmp_path, "http://127.0.0.1:18093")
    audit["observations"][0]["sensor_age_ms"] = 30
    path.write_text(json.dumps(audit))
    cv2.imwrite(str(tmp_path / "front_depth_mm.png"), np.full((240, 320), 800, np.uint16))
    with pytest.raises(ValueError, match="SHA"):
        camera_probe.probe(path, tmp_path, "http://127.0.0.1:18093")
