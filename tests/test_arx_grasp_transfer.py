"""Learned model-base poses preserve metric centres and ARX tool axes."""

import numpy as np
import pytest

from robots.manipulation.grasp_proposals import transfer_grasp_pose
from robots.arx.gateway.grasp_contracts import (
    GraspRecoveryConfig,
    ProposeGraspArgs,
    ReviewGraspArgs,
)


def mapping():
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.195],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_transfer_maps_arx_forward_and_closing_axes_at_finger_centre():
    model = np.eye(4)
    model[:3, 3] = [0.1, 0.2, 0.3]
    camera = np.eye(4)
    camera[:3, 3] = [0.4, -0.1, 0.05]
    result = transfer_grasp_pose(model, camera, mapping())
    assert result[:3, 3] == pytest.approx([0.5, 0.1, 0.545])
    assert result[:3, 0] == pytest.approx([0.0, 0.0, 1.0])
    assert result[:3, 1] == pytest.approx([1.0, 0.0, 0.0])
    assert result[:3, 2] == pytest.approx([0.0, 1.0, 0.0])
    assert model[:3, 3] == pytest.approx([0.1, 0.2, 0.3])


def test_transfer_offset_rotates_with_model_not_with_world():
    model = np.eye(4)
    model[:3, :3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    result = transfer_grasp_pose(model, np.eye(4), mapping())
    assert result[:3, 3] == pytest.approx([0.195, 0.0, 0.0])
    assert result[:3, 0] == pytest.approx([1.0, 0.0, 0.0])


def test_transfer_reflection_is_rejected():
    reflected = mapping()
    reflected[:3, 1] *= -1
    with pytest.raises(ValueError, match="rigid"):
        transfer_grasp_pose(np.eye(4), np.eye(4), reflected)


def commissioning_settings(**kwargs):
    return GraspRecoveryConfig(
        learned_pregrasp_commissioning=True,
        learned_grasp_to_tcp=np.eye(4).tolist(),
        learned_gripper_id="robotiq_2f_140",
        learned_model_sha256="a" * 64,
        **kwargs,
    )


def test_commissioning_requires_identity_and_denies_full_execution_phases():
    with pytest.raises(ValueError, match="identified gripper"):
        GraspRecoveryConfig(learned_pregrasp_commissioning=True)
    settings = commissioning_settings()
    settings.validate_execution_phases(["pregrasp"])
    for phase in ("engage", "lift"):
        with pytest.raises(ValueError, match="pregrasp only"):
            settings.validate_execution_phases(["pregrasp", phase])


def test_commissioning_reviews_learned_pregrasp_without_claiming_full_transfer():
    from types import SimpleNamespace
    from tests.test_arx_grasp_recovery import Observer, context
    from robots.arx.gateway.grasp_recovery import GraspRecovery

    settings = commissioning_settings(motion_enabled=True)
    engine = SimpleNamespace(
        propose=lambda cloud, **kw: [
            {"transform_camera": np.eye(4).tolist(), "score": 0.7}
        ]
    )
    observer = Observer()
    pose = np.eye(4)
    pose[:3, 3] = observer.target
    engine.propose = lambda cloud, **kw: [
        {"transform_camera": pose.tolist(), "score": 0.7}
    ]
    rec = GraspRecovery(
        settings,
        observer,
        closed_policy=0.0,
        open_policy=-3.4,
        engines={"graspgen": engine},
    )
    ctx = context()
    result = rec.propose(ProposeGraspArgs(engine="graspgen"), ctx)
    review = rec.review(ReviewGraspArgs(proposal_id=result["proposal_id"]), ctx)
    assert review["eligible"] is True
    assert review["checks"][0]["learned_transfer_physically_verified"] is False
    assert review["checks"][0]["learned_pregrasp_commissioning"] is True
    for phase in ("engage", "lift"):
        with pytest.raises(ValueError, match="transfer"):
            rec._goal(result["candidates"][0], "graspgen", phase, ctx["command"])


def test_learned_pose_too_far_from_selected_target_is_rejected():
    from types import SimpleNamespace
    from tests.test_arx_grasp_recovery import Observer, context
    from robots.arx.gateway.grasp_recovery import GraspRecovery

    observer = Observer()
    pose = np.eye(4)
    pose[:3, 3] = observer.target + [0.04, 0.0, 0.0]
    rec = GraspRecovery(
        commissioning_settings(),
        observer,
        closed_policy=0.0,
        open_policy=-3.4,
        engines={
            "graspgen": SimpleNamespace(
                propose=lambda cloud, **kw: [
                    {"transform_camera": pose.tolist(), "score": 0.99}
                ]
            )
        },
    )
    ctx = context()
    result = rec.propose(ProposeGraspArgs(engine="graspgen"), ctx)
    review = rec.review(ReviewGraspArgs(proposal_id=result["proposal_id"]), ctx)
    assert not review["eligible"]
    assert "selected target distance" in review["checks"][-1]["reason"]


def test_half_turn_preserves_centre_and_approach_but_exchanges_fingers():
    from robots.manipulation.grasp_proposals import parallel_jaw_candidates

    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    source = [{"transform_camera": pose.tolist(), "score": 0.8}]
    variants = parallel_jaw_candidates(source, max_candidates=2)
    original, flipped = [
        transfer_grasp_pose(c["transform_camera"], np.eye(4), mapping())
        for c in variants
    ]
    assert flipped[:3, 3] == pytest.approx(original[:3, 3])
    assert flipped[:3, 0] == pytest.approx(original[:3, 0])
    assert flipped[:3, 1] == pytest.approx(-original[:3, 1])
    assert variants[1]["score_semantic"] == "inherited_not_rescored"
    assert "pose_variant" not in source[0]
    assert len(parallel_jaw_candidates(source * 5, max_candidates=3)) == 3


def test_half_turn_requires_explicit_supported_gripper():
    with pytest.raises(ValueError, match="robotiq_2f_140"):
        GraspRecoveryConfig(learned_parallel_jaw_half_turn=True)


def test_review_denies_plan_outside_real_controller_joint_bounds():
    from tests.test_arx_grasp_recovery import Observer, context
    from robots.arx.gateway.grasp_recovery import GraspRecovery

    rec = GraspRecovery(
        GraspRecoveryConfig(),
        Observer(),
        closed_policy=0.0,
        open_policy=-3.4,
        joint_bounds=[[-0.001, 0.001]] * 6,
    )
    ctx = context()
    proposal = rec.propose(ProposeGraspArgs(), ctx)
    review = rec.review(ReviewGraspArgs(proposal_id=proposal["proposal_id"]), ctx)
    assert not review["eligible"]
    assert "real controller joint command bounds" in review["checks"][-1]["reason"]


def test_unverified_learned_commission_cannot_resume_vla_before_hardware(tmp_path):
    from types import SimpleNamespace
    from scripts.deployment.commission_arx_pregrasp import run

    path = tmp_path / "config.json"
    path.write_text(commissioning_settings().model_dump_json())
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="cannot resume VLA"):
        run(SimpleNamespace(grasp_config=path, resume_vla_once=True, output=output))
    assert not output.exists()


def make_history(tmp_path, *, past_stamp=10, target_stamp=20):
    import hashlib
    import json
    import sqlite3

    sensors = tmp_path / "grasp-sensors"
    sensors.mkdir()
    path = sensors / "past.npz"
    np.savez(path, front_rgb=np.zeros((1, 1, 3), dtype=np.uint8))
    target = {
        "observation_id": "obs-final",
        "hardware": {"state_monotonic_ns": target_stamp},
    }
    rows = [
        {
            "observation_id": "obs-past",
            "observation": {"hardware": {"state_monotonic_ns": past_stamp}},
            "path": "past.npz",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        {
            "observation_id": "obs-final",
            "observation": target,
            "path": "final.npz",
            "sha256": "a" * 64,
        },
        {
            "observation_id": "obs-future",
            "observation": {"hardware": {"state_monotonic_ns": 30}},
            "path": "missing-future.npz",
            "sha256": "b" * 64,
        },
    ]
    journal = tmp_path / "journal.sqlite3"
    with sqlite3.connect(journal) as db:
        db.execute("CREATE TABLE records(sequence INTEGER, kind TEXT, payload TEXT)")
        db.executemany(
            "INSERT INTO records VALUES(?, ?, ?)",
            [
                (i, "grasp_sensor_evidence", json.dumps(row))
                for i, row in enumerate(rows)
            ],
        )
    return journal, target, path


def test_offline_history_replays_unique_past_only_and_checks_sensor_sha(tmp_path):
    from types import SimpleNamespace
    from scripts.deployment.probe_arx_grasp_snapshot import replay_target_history

    journal, target, path = make_history(tmp_path)
    stamps = []
    def sample(images, hardware):
        stamps.append(hardware["state_monotonic_ns"])
        return {"point_left": np.array([.1, .2, .3]), "stamp": hardware["state_monotonic_ns"]}
    provider = SimpleNamespace(target_sample=sample)
    evidence = replay_target_history(provider, journal, target, "a" * 64)
    assert stamps == [10]
    assert provider.last_target_left == pytest.approx([.1, .2, .3])
    assert provider.last_target_ns == 10
    assert evidence["past_sample_count"] == 1
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="history sensor SHA"):
        replay_target_history(provider, journal, target, "a" * 64)


def test_offline_history_rejects_future_timestamps(tmp_path):
    from types import SimpleNamespace
    from scripts.deployment.probe_arx_grasp_snapshot import replay_target_history

    journal, target, _ = make_history(tmp_path, past_stamp=25)
    with pytest.raises(ValueError, match="future state"):
        replay_target_history(SimpleNamespace(), journal, target, "a" * 64)


def test_offline_failure_writes_audit_without_motion(tmp_path, monkeypatch):
    import json
    from scripts.deployment import probe_arx_grasp_snapshot as module

    output = tmp_path / "failure.json"
    argv = ["probe", "--hardware-sha256", "a" * 64]
    for name in ("snapshot", "observation", "grasp-config", "hardware-config"):
        argv += ["--" + name, str(tmp_path / "unused")]
    argv += ["--output", str(output)]
    monkeypatch.setattr("sys.argv", argv)

    def fail(args):
        raise ValueError("target identity unavailable")

    monkeypatch.setattr(module, "probe", fail)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 1
    result = json.loads(output.read_text())
    assert result["status"] == "failed"
    assert result["robot_commands_sent"] is False
    assert result["live_motion_eligible"] is False


def test_rotation_rate_is_frozen_and_preserves_joint_increment_cap():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from tests.test_arx_grasp_recovery import Observer
    from robots.arx.gateway.grasp_recovery import GraspRecovery

    class Kinematics:
        def fk(self, q):
            return q[:3].copy(), Rotation.from_rotvec(q[3:]).as_matrix(), None

        def solve(self, q, p, r):
            return np.r_[p, Rotation.from_matrix(r).as_rotvec()]

    observer = Observer()
    observer.provider = SimpleNamespace(tool_fk=Kinematics())
    state = np.zeros(14)
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_rotvec([0.0, 0.0, 1.0]).as_matrix()
    cloud = observer.cloud({"observation_id": "obs-rate"}, {})
    slow = GraspRecovery(
        GraspRecoveryConfig(), observer, closed_policy=0.0, open_policy=-3.4
    )
    with pytest.raises(ValueError, match="physical step budget"):
        slow._plan(state, state, goal, cloud, "pregrasp", 100)
    fast = GraspRecovery(
        GraspRecoveryConfig(angular_speed_rad_s=0.30),
        observer,
        closed_policy=0.0,
        open_policy=-3.4,
    )
    plan, _ = fast._plan(state, state, goal, cloud, "pregrasp", 100)
    assert len(plan) == 80
    assert np.max(np.abs(np.diff(plan[:, 7:13], axis=0))) <= 0.035
    with pytest.raises(ValueError):
        GraspRecoveryConfig(angular_speed_rad_s=0.31)


def test_frozen_learned_commission_package_has_no_normal_runner_permission(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from robots.arx.gateway import real_factory as module
    from zetta.evolution.jsonio import file_sha256

    root = Path(__file__).resolve().parents[1]
    experiment = root / "docs/experiments/arx-graspgen-transfer-20261008"
    hardware = (
        root
        / "docs/experiments/arx-target-visibility-20261007/hardware-wrist-observer.json"
    )
    provider = root / "robots/arx/deployment/picktube_rgbd_provider.py"
    contract = experiment / "frozen/real-input-contract.json"
    config = experiment / "grasp-config-commission.json"
    from scripts.deployment.freeze_arx_picktube_inputs import freeze

    fresh = tmp_path / "frozen"
    freeze(
        experiment / "bundle-pregrasp.json",
        fresh,
        grasp_config=config,
        depth_cameras=("front_depth_mm", "right_depth_mm"),
        hardware_config=hardware,
    )
    contract = fresh / "real-input-contract.json"

    def forbid_backend(**kw):
        raise AssertionError("preflight must not acquire hardware")

    monkeypatch.setattr(module, "build_real_backend", forbid_backend)
    original_load = module.load_real_hardware_config

    def relocated_load(path, sha):
        config = original_load(path, sha)
        cameras = []
        for camera in config.cameras:
            changes = {}
            for field in ("calibration_file", "robot_mount_calibration_file"):
                value = getattr(camera, field, None)
                if value is not None:
                    changes[field] = root / value.relative_to(
                        "/home/dodo/chenfu/Agentic-Embodied"
                    )
            cameras.append(camera.model_copy(update=changes))
        return config.model_copy(update={"cameras": cameras})

    monkeypatch.setattr(module, "load_real_hardware_config", relocated_load)
    factory = module.RealCoreFactory(
        hardware_config=str(hardware),
        hardware_sha256=file_sha256(hardware),
        task=str(root / "robots/arx/manifests/pickup_test_tube.yaml"),
        model_contract=str(root / "robots/arx/manifests/task7_model_a.yaml"),
        output=str(tmp_path),
        episode_id="static-transfer-test",
        limits=dict(
            max_steps=600,
            max_decisions=16,
            max_recoveries=1,
            operation_timeout_s=120.0,
            critic_timeout_s=5.0,
            idle_agent_timeout_s=120.0,
            lease_timeout_s=120.0,
            shutdown_timeout_s=10.0,
        ),
        zeva_host="127.0.0.1",
        zeva_port=5583,
        kinematics_calibration=str(
            root / "robots/arx/manifests/real/dodo_right_controller_ee_fk.json"
        ),
        bundle=str(experiment / "bundle-pregrasp.json"),
        tool_catalog=str(fresh / "tool-catalog.json"),
        real_input_contract=str(contract),
        expected_real_input_sha256=file_sha256(contract),
        feature_provider=str(provider),
        expected_feature_provider_sha256=file_sha256(provider),
        preflight_only=True,
        grasp_config=str(config),
        expected_grasp_config_sha256=file_sha256(config),
    )
    with pytest.raises(ValueError, match="commissioning harness"):
        factory(lambda: False, lambda phase: None)
    factory.allow_learned_pregrasp_commissioning = True
    assert factory(lambda: False, lambda phase: None)["eligible"]
