"""Hardware-neutral real backend and official ARX SingleArm adapter tests."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robots.arx.contracts import load_task_manifest
from robots.arx.gateway.arx_x5_device import ArmCalibration, ArxX5Device
from robots.arx.gateway.real_backend import (
    CameraFrame,
    CameraIdentity,
    CommandReceipt,
    DeviceHealth,
    GripperClosureSpec,
    JointSample,
    RealBackend,
    RealBackendConfig,
)


TASK = load_task_manifest(
    Path(__file__).resolve().parents[1] / "robots/arx/manifests/pickup_test_tube.yaml"
)


class FakeCameraSource:
    def __init__(self, identities, *, stale=False):
        self.identities = identities
        self.stale = stale
        self.first = None
        self.closed = False

    def capture(self, timeout_s):
        stamp = time.monotonic_ns()
        if self.first is None:
            self.first = stamp
        if self.stale:
            stamp = self.first
        return {
            identity.name: CameraFrame(
                np.zeros((identity.height, identity.width, 3), np.uint8),
                stamp,
                identity,
            )
            for identity in self.identities
        }

    def close(self):
        self.closed = True


class FakeArmDevice:
    def __init__(self, *, move=True, command_offset=0.0):
        self.positions = np.asarray(TASK.start_state, dtype=np.float32).copy()
        self.positions[[6, 13]] += np.asarray(
            TASK.control.gripper_command_offsets, dtype=np.float32
        )
        self.move = move
        self.command_offset = command_offset
        self.sent = []
        self.closed = False

    def read(self):
        start = time.monotonic_ns()
        return JointSample(
            self.positions.copy(),
            time.monotonic_ns(),
            start,
            DeviceHealth(True, diagnostics_available=True),
        )

    def send(self, target, command_id):
        self.sent.append(target.copy())
        expected = target.copy()
        expected[13] += self.command_offset
        if self.move:
            self.positions = expected.copy()
        return CommandReceipt(
            command_id, time.monotonic_ns(), "sdk_call_returned",
            expected_feedback_target=tuple(float(x) for x in expected),
        )

    def close(self):
        self.closed = True


def make_backend(*, move=True, stale=False, command_offset=0.0):
    identities = tuple(
        CameraIdentity(name, "device-" + name, "cal-" + name, "a" * 64, 2, 2)
        for name in ("front_rgb", "left_rgb", "right_rgb")
    )
    cameras = FakeCameraSource(identities, stale=stale)
    arm = FakeArmDevice(move=move, command_offset=command_offset)
    config = RealBackendConfig(
        cameras=identities,
        state_units=("rad",) * 6 + ("policy_gripper",)
        + ("rad",) * 6 + ("policy_gripper",),
        control_hz=1000,
        max_sensor_skew_ms=50,
        max_sensor_age_ms=200,
        observation_timeout_s=0.015,
        arrival_timeout_s=0.025,
        feedback_poll_s=0.001,
        position_tolerance=(0.001,) * 14,
    )
    backend = RealBackend(arms=arm, cameras=cameras, config=config, task=TASK)
    return backend, arm, cameras


def test_real_backend_reports_send_and_measured_arrival_separately():
    backend, arm, cameras = make_backend()
    events = []
    backend.set_event_sink(lambda kind, payload: events.append((kind, payload)))
    initial = backend.reset()
    assert initial.hardware.command_receipt is None
    assert not arm.sent
    raw = np.asarray(TASK.start_state, np.float32)
    raw[7] += 0.01
    commit = backend.step(raw)
    assert commit.hardware.arrival_verified is True
    assert commit.hardware.command_receipt["status"] == "sdk_call_returned"
    assert np.allclose(commit.policy.state, arm.sent[-1])
    assert [kind for kind, _ in events] == [
        "command_dispatch_started", "command_sent", "arrival_observed"
    ]
    assert commit.hardware.observation["sensor_skew_ms"] < 50
    backend.close()
    assert arm.closed and cameras.closed


def test_real_backend_arrival_uses_firmer_grip_command_feedback():
    backend, arm, _ = make_backend(command_offset=0.9)
    backend.reset()
    raw = np.asarray(TASK.start_state, np.float32)
    raw[7] += 0.01
    commit = backend.step(raw)
    assert commit.hardware.arrival_verified is True
    assert arm.sent[-1][13] == pytest.approx(-3.4)
    assert commit.policy.state[13] == pytest.approx(-2.5)
    assert commit.hardware.observation["expected_feedback_target"][13] == pytest.approx(-2.5)


def test_sensor_refresh_never_sends_a_command_and_rechecks_current_arrival():
    backend, arm, _ = make_backend(command_offset=.9)
    backend.reset()
    refreshed = backend.observe()
    assert arm.sent == []
    assert refreshed.hardware.command_receipt is None
    assert refreshed.hardware.arrival_verified is None
    raw = np.asarray(TASK.start_state, np.float32)
    raw[7] += .01
    sent = backend.step(raw)
    count = len(arm.sent)
    refreshed = backend.observe()
    assert len(arm.sent) == count
    assert refreshed.hardware.command_receipt is None
    assert refreshed.hardware.arrival_verified is True
    assert refreshed.command == pytest.approx(sent.command)
    arm.positions[7] += .1
    assert backend.observe().hardware.arrival_verified is False
    assert len(arm.sent) == count


def test_real_backend_rejects_unlocked_arm_outside_frozen_start_state():
    backend, arm, _ = make_backend()
    arm.positions[13] += 0.5
    with pytest.raises(ValueError, match="frozen task start state at channels 13"):
        backend.reset()
    assert arm.sent == []


def test_unverified_arrival_is_not_reported_as_acknowledgement():
    backend, arm, _ = make_backend(move=False)
    events = []
    backend.set_event_sink(lambda kind, payload: events.append(kind))
    backend.reset()
    raw = np.asarray(TASK.start_state, np.float32)
    raw[7] += 0.01
    commit = backend.step(raw)
    assert commit.hardware.command_receipt["status"] == "sdk_call_returned"
    assert commit.hardware.arrival_verified is False
    assert "command_sent" in events and "arrival_unverified" in events
    assert not np.allclose(commit.command, commit.policy.state)


def closure_backend():
    backend, _, _ = make_backend()
    backend.config = replace(
        backend.config, position_tolerance=(0.05,) * 13 + (0.1,),
        gripper_closures=(GripperClosureSpec(
            index=13, closed_feedback_policy=0.0, open_feedback_policy=-3.4,
            current_channel="right_gripper_current_native",
        ),),
    )
    return backend


def test_joint_targets_are_bounded_without_filter_windup_or_gripper_clipping():
    backend, arm, _ = make_backend()
    limits = [(-3.0, 3.0)] * 12
    limits[6] = (-0.1, 0.025)  # right joint 1
    backend.config = replace(backend.config, joint_command_bounds=tuple(limits))
    events = []
    backend.set_event_sink(lambda kind, payload: events.append((kind, payload)))
    backend.reset()
    raw = np.asarray(TASK.start_state, np.float32)
    raw[7] = 2.0
    raw[13] = 0.0
    for _ in range(3):
        commit = backend.step(raw)
        assert commit.command[7] == pytest.approx(0.025)
        assert backend._processor.previous[7] == pytest.approx(0.025)
        assert commit.hardware.arrival_verified
    assert len([event for event in events if event[0] == "joint_command_limited"]) == 3
    assert arm.sent[-1][13] > TASK.start_state[13]
    raw[7] = -1.0
    commit = backend.step(raw)
    assert commit.command[7] == pytest.approx(-0.01)  # immediately leaves bound, 0.035 step


def test_tightening_overdrive_references_physical_closed_position():
    backend = closure_backend()
    target = np.zeros(14)
    target[13] = 0.0140459538
    measured = np.zeros(14)
    measured[13] = -0.0944156647
    arrived, reference, details = backend._arrival_status(
        measured, {"state_monotonic_ns": 1_000_000_000}, target,
    )
    assert arrived and reference[13] == 0.0
    assert details["13"]["mode"] == "position"
    assert target[13] == pytest.approx(0.0140459538)  # dispatch remains auditable


def test_blocked_closure_requires_stable_near_closed_feedback_and_preserves_joint_gate():
    backend = closure_backend()
    target = np.zeros(14)
    target[13] = 0.9
    measured = np.zeros(14)
    measured[13] = -0.85
    hardware = {"state_monotonic_ns": 1_000_000_000,
                "auxiliary_feedback": {"right_gripper_current_native": 0.20}}
    assert not backend._arrival_status(measured, hardware, target)[0]
    hardware["state_monotonic_ns"] += 250_000_000
    arrived, _, details = backend._arrival_status(measured, hardware, target)
    assert arrived and details["13"]["mode"] == "closed_settled"
    assert details["13"]["current"] == 0.20
    measured[7] = 0.1
    assert not backend._arrival_status(measured, hardware, target)[0]
    measured[7] = 0.0
    target[13] = -2.5  # opening remains a position goal
    assert not backend._arrival_status(measured, hardware, target)[0]


def test_unsettled_or_open_gripper_cannot_satisfy_closing_command():
    backend = closure_backend()
    target = np.zeros(14)
    target[13] = 0.9
    measured = np.zeros(14)
    measured[13] = -0.85
    hardware = {"state_monotonic_ns": 1_000_000_000}
    assert not backend._arrival_status(measured, hardware, target)[0]
    measured[13] = -0.7
    hardware["state_monotonic_ns"] += 250_000_000
    assert not backend._arrival_status(measured, hardware, target)[0]
    measured[13] = -2.5
    hardware["state_monotonic_ns"] += 500_000_000
    assert not backend._arrival_status(measured, hardware, target)[0]


def test_measured_closed_band_can_settle_for_intermediate_closed_target():
    # Real g0001 stopped despite a 19% command and stable 23% measured opening.
    # Both are inside the calibrated closed band; raw errors remain auditable.
    backend = closure_backend()
    target = np.zeros(14)
    target[13] = -0.659932971
    measured = np.zeros(14)
    measured[13] = -0.786412239
    hardware = {"state_monotonic_ns": 1_000_000_000}
    assert not backend._arrival_status(measured, hardware, target)[0]
    hardware["state_monotonic_ns"] += 250_000_000
    arrived, reference, details = backend._arrival_status(measured, hardware, target)
    assert arrived and details["13"]["mode"] == "closed_settled"
    assert reference[13] == target[13]  # only overdrive clamps its reference
    measured[7] = 0.1
    assert not backend._arrival_status(measured, hardware, target)[0]
    measured[7] = 0
    target[13] = -1.1  # outside the closed band: exact position gate remains
    assert not backend._arrival_status(measured, hardware, target)[0]
    target[13] = -0.659932971
    measured[13] = -1.1
    assert not backend._arrival_status(measured, hardware, target)[0]


def test_stale_camera_prevents_motion_observation():
    backend, arm, _ = make_backend(stale=True)
    backend.reset()
    raw = np.asarray(TASK.start_state, np.float32)
    raw[7] += 0.01
    with pytest.raises(TimeoutError, match="stale camera frame"):
        backend.step(raw)
    assert len(arm.sent) == 1


def test_real_backend_rejects_missing_aligned_depth_and_keeps_metric_frame_private():
    backend, _, cameras = make_backend()
    backend.config = replace(backend.config, depth_cameras=("front_rgb",))
    with pytest.raises(ValueError, match="aligned metric depth missing"):
        backend.reset()
    original = cameras.capture

    def capture_with_depth(timeout_s):
        frames = original(timeout_s)
        frames["front_rgb"] = replace(
            frames["front_rgb"], depth_mm=np.full((2, 2), 500, np.uint16)
        )
        return frames

    cameras.capture = capture_with_depth
    commit = backend.reset()
    assert commit.hardware.feature_frames["front_depth_mm"].dtype == np.uint16
    assert commit.hardware.observation["depth_monotonic_ns"]["front_depth_mm"] > 0
    assert "front_depth_mm" not in commit.policy.images


class FakeSingleArm:
    def __init__(self, grip=-3.0):
        self.positions = np.array([0.0] * 6 + [grip])
        self.commands = []

    def get_joint_positions(self):
        return self.positions.copy()

    def get_ee_pose(self):
        return np.array([0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0])

    def set_joint_positions(self, *, positions):
        self.commands.append(("joints", positions))
        self.positions[:6] = positions

    def set_gripper_pos(self, value):
        self.commands.append(("gripper", value))
        self.positions[6] = value


def test_arx_adapter_converts_gripper_and_respects_locked_arm():
    left, right = FakeSingleArm(), FakeSingleArm()
    calibration = ArmCalibration(
        can_port="can1", arm_type=2,
        joint_min_rad=(-3.0,) * 6,
        joint_max_rad=(3.0,) * 6,
        gripper_native_min=-3.5,
        gripper_native_max=0.0,
        gripper_policy_scale=2.0,
        gripper_policy_offset=1.0,
    )
    device = ArxX5Device(
        left=left, right=right,
        left_calibration=calibration, right_calibration=calibration,
        command_left=False, command_right=True,
    )
    sample = device.read()
    state = sample.positions
    assert state.shape == (14,) and state[6] == state[13] == -5.0
    assert sample.right_tcp_xyz_m.tolist() == [0.2, -0.1, 0.3]
    target = state.copy()
    target[7] = 0.1
    target[13] = -3.0
    receipt = device.send(target, "cmd-test")
    assert receipt.status == "sdk_call_returned"
    assert not left.commands
    assert right.commands[-1] == ("gripper", -2.0)
    assert device.read().positions[13] == -3.0


def test_arx_adapter_applies_task_command_offset_without_changing_feedback():
    left, right = FakeSingleArm(), FakeSingleArm()
    calibration = ArmCalibration(
        can_port="can3", arm_type=2,
        joint_min_rad=(-3.0,) * 6, joint_max_rad=(3.0,) * 6,
        gripper_native_min=-3.5, gripper_native_max=1.0,
        gripper_policy_scale=1.0, gripper_policy_offset=0.0,
        gripper_command_offset=0.9,
    )
    device = ArxX5Device(
        left=left, right=right,
        left_calibration=calibration, right_calibration=calibration,
        command_left=False, command_right=True,
    )
    target = device.read().positions
    target[13] = -3.4
    receipt = device.send(target, "offset-test")
    assert right.commands[-1] == ("gripper", pytest.approx(-2.5))
    assert device.read().positions[13] == pytest.approx(-2.5)
    assert receipt.expected_feedback_target[13] == pytest.approx(-2.5)


class FakeRosPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeRosNode:
    def __init__(self):
        self.publishers = {}
        self.subscribers = {}

    def create_publisher(self, message_type, topic, depth):
        publisher = FakeRosPublisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, message_type, topic, callback, depth):
        self.subscribers[topic] = callback

    def get_clock(self):
        class Clock:
            def now(self):
                class Stamp:
                    def to_msg(self):
                        return object()
                return Stamp()
        return Clock()


class FakeRobotStatus:
    def __init__(self):
        self.header = type("Header", (), {})()
        self.joint_pos = [0.0] * 7
        self.joint_cur = [0.0] * 7
        self.end_pos = [0.2, -0.1, 0.3, 0.0, 0.0, 0.0]


def test_dodo_ros2_topics_use_status_feedback_and_publish_receipt():
    from robots.arx.gateway.arx_ros2_device import ArxRos2Device, Ros2Topics

    node = FakeRosNode()
    calibration = ArmCalibration(
        can_port="can1", arm_type=2,
        joint_min_rad=(-3.0,) * 6, joint_max_rad=(3.0,) * 6,
        gripper_native_min=-3.5, gripper_native_max=0.0,
        gripper_policy_scale=1.0, gripper_policy_offset=0.0,
    )
    device = ArxRos2Device(
        node=node, message_type=FakeRobotStatus, topics=Ros2Topics(),
        left_calibration=calibration, right_calibration=calibration,
        command_left=False, command_right=True,
        keepalive_hz=10, command_lease_s=0.01,
    )
    try:
        for topic in ("/arm_slave_l_status", "/arm_slave_r_status"):
            msg = FakeRobotStatus()
            msg.joint_pos[6] = -3.0
            msg.joint_cur[6] = 0.2
            node.subscribers[topic](msg)
        sample = device.read()
        state = sample.positions
        assert state.shape == (14,)
        assert sample.right_tcp_xyz_m.tolist() == [0.2, -0.1, 0.3]
        assert sample.auxiliary_feedback == {"right_gripper_current_native": 0.2}
        target = state.copy()
        target[7] = 0.1
        receipt = device.send(target, "ros-cmd")
        assert receipt.status == "ros_publish_returned"
        assert len(node.publishers["/arm_master_r_status"].messages) == 1
        assert not node.publishers["/arm_master_l_status"].messages
        assert node.publishers["/arm_master_r_status"].messages[0].joint_pos[0] == pytest.approx(0.1)
    finally:
        device.close()


def test_static_real_hardware_config_matches_vla_and_calibration_files(tmp_path):
    import json
    from robots.arx.gateway.real_config import (
        load_real_hardware_config, validate_real_hardware_config,
    )
    from zetta.evolution.jsonio import file_sha256

    root = Path(__file__).resolve().parents[1]
    model_path = root / "robots/arx/manifests/task7_model_a.yaml"
    task_path = root / "robots/arx/manifests/pickup_test_tube.yaml"
    model = json.loads(model_path.read_text())
    cameras = []
    for index, camera in enumerate(model["cameras"]):
        calibration = tmp_path / f"camera-{index}.json"
        calibration.write_text(json.dumps({
            "schema_version": "arx.real.rgb_intrinsics.v1",
            "calibration_scope": "rgb_intrinsics_only",
            "camera": {
                "logical_name": camera["name"],
                "serial": f"serial-{index}",
                "width": 640, "height": 480,
                "intrinsics": {
                    "fx": 395.0, "fy": 395.0, "ppx": 320.0, "ppy": 240.0,
                    "coeffs": [0.0] * 5,
                    "distortion_model": "inverse_brown_conrady",
                },
            },
        }))
        cameras.append({
            "name": camera["name"],
            "serial": f"serial-{index}",
            "calibration_id": camera["calibration_id"],
            "calibration_file": str(calibration),
            "calibration_sha256": file_sha256(calibration),
            "width": camera["width"],
            "height": camera["height"],
            "capture_width": 640,
            "capture_height": 480,
            "capture_fps": 30,
        })
    arm = {
        "can_port": "can1", "arm_type": 2,
        "joint_min_rad": [-4.0] * 6,
        "joint_max_rad": [4.0] * 6,
        "gripper_native_min": -4.0,
        "gripper_native_max": 1.0,
        "gripper_policy_scale": 1.0,
        "gripper_policy_offset": 0.0,
    }
    payload = {
        "schema_version": "arx.real.hardware.v1",
        "arm_transport": "arx_ros2",
        "camera_transport": "realsense",
        "left": arm,
        "right": {**arm, "can_port": "can3"},
        "command_left": False,
        "command_right": True,
        "cameras": cameras,
        "timing": {
            "control_hz": 15.0,
            "max_sensor_skew_ms": 30.0,
            "max_sensor_age_ms": 150.0,
            "observation_timeout_s": 1.0,
            "arrival_timeout_s": 0.25,
            "feedback_poll_s": 0.01,
            "position_tolerance": [0.01] * 14,
        },
        "right_gripper_closed_policy": 0.0,
        "right_gripper_open_policy": -3.4,
    }
    path = tmp_path / "real-hardware.json"
    path.write_text(json.dumps(payload))
    config = load_real_hardware_config(path, file_sha256(path))
    validate_real_hardware_config(config, task_path, model_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_real_hardware_config(path, "0" * 64)
    cameras[0]["serial"] = cameras[1]["serial"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="serials must be unique"):
        load_real_hardware_config(path, file_sha256(path))
    cameras[0]["serial"] = "serial-0"
    calibration = Path(cameras[0]["calibration_file"])
    calibration.write_text(json.dumps({
        "schema_version": "arx.real.rgb_intrinsics.v1",
        "calibration_scope": "rgb_intrinsics_only",
        "camera": {
            "logical_name": "front_rgb", "serial": "old-camera",
            "width": 640, "height": 480,
            "intrinsics": {
                "fx": 395.0, "fy": 395.0, "ppx": 320.0, "ppy": 240.0,
                "coeffs": [0.0] * 5,
                "distortion_model": "inverse_brown_conrady",
            },
        },
    }))
    cameras[0]["calibration_sha256"] = file_sha256(calibration)
    path.write_text(json.dumps(payload))
    config = load_real_hardware_config(path, file_sha256(path))
    with pytest.raises(ValueError, match="calibration identity"):
        validate_real_hardware_config(config, task_path, model_path)
