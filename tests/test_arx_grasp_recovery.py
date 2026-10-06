"""Real grasp admission, model frame mapping, evidence and runner budget tests."""

from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest

from robots.arx.deployment.bundle_program import compile_programs, resolve_call, retain_tool_outputs, verify_call_result
from robots.manipulation.grasp_proposals import TargetCloud, geometry_proposal, rigid_pose
from robots.arx.gateway.grasp_contracts import GraspRecoveryConfig, ProposeGraspArgs, ReviewGraspArgs, ExecuteGraspArgs
from robots.arx.gateway.grasp_recovery import GraspRecovery
from robots.arx.gateway.tools import ApprovedToolContext
from tests.test_arx_bundle_execution import bundle
from zetta.evolution.models import RecoveryStep


class LinearKinematics:
    def fk(self, q):
        return q[:3].copy(), np.eye(3), np.eye(6)

    def solve(self, q, p, r):
        return np.r_[p, q[3:]]


class Observer:
    def __init__(self):
        self.provider = SimpleNamespace(tool_fk=LinearKinematics(),
            observe=lambda o, i: {"privileged.interaction.gripper_contact": o["hardware"].get("contact", False)})
        self.target = np.array([.10, 0., .10])
        self.scene = np.repeat([[0., .2, 0.]], 64, axis=0)

    def cloud(self, observation, images):
        return TargetCloud("pink", np.repeat(self.target[None], 40, axis=0), self.scene,
                           self.target.copy(), np.eye(4), {"observation_id": observation["observation_id"]})


def context(state=None):
    state = np.zeros(14) if state is None else state.copy()
    state[7:10] = [.05, 0., .1]
    state[13] = -3.4
    stamp = time.monotonic_ns()
    observation = {"observation_id": "obs-1", "hardware": {
        "measured_state": state.tolist(), "observation_completed_ns": time.monotonic_ns(),
        "sensor_age_ms": 10., "sensor_skew_ms": 5.,
        "auxiliary_feedback": {"right_gripper_current_native": .02},
        "state_monotonic_ns": stamp,
        "auxiliary_monotonic_ns": {"right_gripper_current_native": stamp},
        "device_health": {"transport_responsive": True, "fault_codes": []},
    }}
    return {"observation": observation, "images": {}, "command": state}


def recovery(**config):
    return GraspRecovery(GraspRecoveryConfig(motion_enabled=True, **config), Observer(),
                         closed_policy=0., open_policy=-3.4)


def proposed_review(rec):
    ctx = context()
    proposal = rec.propose(ProposeGraspArgs(), ctx)
    review = rec.review(ReviewGraspArgs(proposal_id=proposal["proposal_id"]), ctx)
    return ctx, proposal, review


def test_geometry_model_pose_preserves_arx_rotation_and_metric_target():
    cloud = Observer().cloud(context()["observation"], {})
    pose = rigid_pose(geometry_proposal(cloud, np.eye(3))[0]["transform_camera"])
    assert pose[:3, 3] == pytest.approx([.1, 0., .1])
    assert pose[:3, :3] == pytest.approx(np.eye(3))
    invalid = pose.copy()
    invalid[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        rigid_pose(invalid)


def test_review_token_is_single_use_and_plan_completion_requires_measured_pose():
    rec = recovery()
    ctx, proposal, review = proposed_review(rec)
    assert review["eligible"]
    assert review["certificate_level"] == "sensor_tcp_clearance_and_joint_ik"
    args = ExecuteGraspArgs(review_token=review["review_token"])
    ctx["observation"]["hardware"]["observation_completed_ns"] = time.monotonic_ns()
    current = ApprovedToolContext(ctx["command"], ctx["observation"], {})
    plan = rec.prepare(args, current)
    assert len(plan.targets) <= 180
    assert np.max(np.abs(np.diff(plan.targets[:, 7:13], axis=0))) <= .035
    # Merely matching the sent command cannot certify physical arrival.
    plan.on_commit(ApprovedToolContext(plan.targets[-1], ctx["observation"], {}))
    assert not plan.reached
    measured = dict(ctx["observation"], hardware=dict(ctx["observation"]["hardware"],
                                                    measured_state=plan.targets[-1].tolist()))
    plan.on_commit(ApprovedToolContext(plan.targets[-1], measured, {}))
    assert plan.reached
    assert "pregrasp" in rec.completed_phases[proposal["proposal_id"]]
    with pytest.raises(ValueError, match="reused"):
        rec.prepare(args, current)


def test_held_object_rejects_before_open_and_stale_sensor_rejects():
    rec = recovery()
    ctx = context()
    ctx["observation"]["hardware"]["measured_state"][13] = -.1
    ctx["observation"]["hardware"]["auxiliary_feedback"]["right_gripper_current_native"] = .20
    with pytest.raises(ValueError, match="held object"):
        rec.propose(ProposeGraspArgs(), ctx)
    ctx = context()
    ctx["observation"]["hardware"]["observation_completed_ns"] -= 3_000_000_000
    with pytest.raises(ValueError, match="fresh"):
        rec.propose(ProposeGraspArgs(), ctx)


def test_target_drift_and_new_obstacle_and_expired_token_reject_execution():
    rec = recovery()
    ctx, _, review = proposed_review(rec)
    args = ExecuteGraspArgs(review_token=review["review_token"])
    rec.observer.target += [.02, 0., 0.]
    with pytest.raises(ValueError, match="target moved"):
        rec.prepare(args, ApprovedToolContext(ctx["command"], ctx["observation"], {}))
    rec = recovery()
    ctx, _, review = proposed_review(rec)
    args = ExecuteGraspArgs(review_token=review["review_token"])
    rec.observer.scene[:] = [.06, 0., .1]
    with pytest.raises(ValueError, match="intersects"):
        rec.prepare(args, ApprovedToolContext(ctx["command"], ctx["observation"], {}))
    rec = recovery()
    ctx, _, review = proposed_review(rec)
    rec.clock = lambda: time.monotonic() + 10.
    with pytest.raises(ValueError, match="stale"):
        rec.prepare(ExecuteGraspArgs(review_token=review["review_token"]),
                    ApprovedToolContext(ctx["command"], ctx["observation"], {}))


def test_learned_panda_pose_cannot_execute_without_verified_arx_transfer():
    rec = recovery()
    rec.engines["graspgen"] = SimpleNamespace(propose=lambda *a, **k: geometry_proposal(a[0], np.eye(3)))
    ctx = context()
    proposal = rec.propose(ProposeGraspArgs(engine="graspgen"), ctx)
    review = rec.review(ReviewGraspArgs(proposal_id=proposal["proposal_id"]), ctx)
    assert not review["eligible"]
    assert "transfer" in review["checks"][-1]["reason"]


def test_engage_and_lift_require_measured_phase_and_target_contact():
    rec = recovery()
    ctx, proposal, _ = proposed_review(rec)
    engage = rec.review(ReviewGraspArgs(proposal_id=proposal["proposal_id"], phase="engage"), ctx)
    assert not engage["eligible"]
    rec.completed_phases[proposal["proposal_id"]] = {"pregrasp", "engage"}
    lift = rec.review(ReviewGraspArgs(proposal_id=proposal["proposal_id"], phase="lift"), ctx)
    assert not lift["eligible"] and "contact" in lift["checks"][-1]["reason"]


def grasp_bundle():
    original = bundle()
    rule = replace(original.recovery_rules[0], steps=(
        RecoveryStep("arx.propose_grasp", {"engine": "tube_geometry"}, "target proposal"),
        RecoveryStep("arx.set_gripper", {"opening": 1., "max_steps": 60}, "measured opening"),
        RecoveryStep("arx.review_grasp", {"proposal_id": "proposal-from-last", "phase": "pregrasp", "max_steps": 180}, "eligible"),
        RecoveryStep("arx.execute_grasp", {"review_token": "grasp-token-from-review", "phase": "pregrasp", "max_steps": 180}, "measured target pose"),
        RecoveryStep("arx.zeva", {"max_chunks": 1}, "fresh policy"),
    ))
    return replace(original, recovery_rules=(rule,))


def test_bundle_reserves_motion_budget_and_binds_only_reviewed_outputs():
    program = compile_programs(grasp_bundle())["recover"]
    assert program.binding.max_recovery_steps == 256
    assert program.calls[-2].tool == "arx.review_reentry"
    with pytest.raises(ValueError, match="budget"):
        compile_programs(grasp_bundle(), max_physical_steps=255)
    outputs = {}
    retain_tool_outputs(program.calls[0], {"result": {"proposal_id": "grasp-1"}}, outputs)
    assert resolve_call(program.calls[2], "obs-2", None, outputs)["proposal_id"] == "grasp-1"
    retain_tool_outputs(program.calls[2], {"result": {"review_token": "grasp-review-1"}}, outputs)
    assert resolve_call(program.calls[3], "obs-3", None, outputs)["review_token"] == "grasp-review-1"
    retain_tool_outputs(program.calls[3], {"result": {}}, outputs)
    with pytest.raises(ValueError, match="no eligible"):
        resolve_call(program.calls[3], "obs-3", None, outputs)
    result = {"status": "completed", "result": {"command_target_reached": True, "physical_arrival_verified": False}}
    with pytest.raises(ValueError, match="physical arrival"):
        verify_call_result(program.calls[3], result, real=True)


def test_remote_model_endpoints_and_improper_transfer_configuration_rejected():
    with pytest.raises(ValueError, match="loopback"):
        GraspRecoveryConfig(graspgen_endpoint="http://192.168.1.1:18093")
    with pytest.raises(ValueError, match="identified gripper"):
        GraspRecoveryConfig(learned_gripper_transfer_verified=True)


def test_runner_and_gateway_bind_grasp_outputs_and_preserve_physical_step_count(tmp_path):
    from robots.arx.gateway.backend import HardwareEvidence
    from robots.arx.gateway.tools import default_registry, PolicyGripperPlanner
    from tests.test_arx_gateway import FakeBackend, ScriptCritic, make_core, call, limits, Review
    from tests.test_arx_deployment import runner

    class Sensors(FakeBackend):
        reads = 0

        def __init__(self):
            super().__init__(terminal=100)
            self.command = context()["command"].astype(np.float32)

        def commit(self):
            commit = super().commit()
            ctx = context()
            hardware = ctx["observation"]["hardware"]
            hardware["measured_state"] = self.command.tolist()
            return replace(commit, hardware=HardwareEvidence(hardware, None, True, {}))

        def observe(self):
            self.reads += 1
            return self.commit()

    backend = Sensors()
    core, _, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}), backend=backend,
                           config=limits(max_steps=100, max_decisions=50))
    rec = recovery()
    zeva = core.registry.resolve("arx.zeva").handler
    core.registry = default_registry(zeva=zeva, gripper=PolicyGripperPlanner(closed_policy=0., open_policy=-3.4),
                                     reentry=Review(), grasp=rec)
    program = compile_programs(grasp_bundle())["recover"]
    core.bindings, core.programs = (program.binding,), {"recover": program}
    first, _ = call(core)
    assert first["status"] == "interrupted"
    r = runner(tmp_path / "run", core)
    r.trial.candidate = SimpleNamespace(package_sha256=grasp_bundle().sha256)
    r._structured_bundle = r._real_bundle = True
    r.bundle_programs = {"recover": program}
    r.trial.runner_limits = r.trial.runner_limits.model_copy(update={"max_tool_attempts": 50})
    assert r.loop() == "environment_ended"
    assert backend.steps == core.step_index == 100
    assert backend.reads == 3
    assert len(list((tmp_path / "core/grasp-sensors").glob("*.npz"))) == 3
    assert r.outcome.reentry_completed


def test_reentry_requires_this_recoverys_target_and_measured_pregrasp():
    from robots.arx.gateway.grasp_recovery import GraspReentry
    rec = recovery()
    ctx, proposal, _ = proposed_review(rec)
    base = SimpleNamespace(inspect=lambda a, c: {"status": "eligible", "checks": []})
    wrapped = GraspReentry(base, rec)
    review_context = {"observations": [ctx["observation"]], "images": [{}],
                      "tool_outputs": {"proposal_id": proposal["proposal_id"]}}
    assert wrapped.inspect(None, review_context)["status"] == "ineligible"
    rec.completed_phases[proposal["proposal_id"]] = {"pregrasp"}
    ctx["observation"]["hardware"]["measured_state"][7] = .07
    assert wrapped.inspect(None, review_context)["status"] == "eligible"
    rec.observer.target += [.1, 0., 0.]
    assert wrapped.inspect(None, review_context)["status"] == "ineligible"


def test_target_cloud_uses_only_pink_depth_and_correct_right_base_transform():
    from robots.arx.deployment.picktube_grasp_observer import PickTubeGraspObserver
    from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider, FRONT_SERIAL, FRONT_INTRINSICS_SHA256
    provider = PickTubeRgbdProvider()
    # Test colour selection/frame conversion independently of the hardware SDK's distortion routine.
    provider._deproject = lambda x, y, z: np.array([(x - 160) * z / 200, (y - 120) * z / 200, z])
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[70:105, 110:220] = [240, 240, 10]
    rgb[40:52, 150:164] = [230, 70, 150]
    rgb[40:52, 180:194] = [20, 120, 220]
    depth = np.full((240, 320), 500, np.uint16)
    depth[40:52, 180:194] = 400
    ctx = context()
    state = np.zeros(14)
    tcp, _, _ = provider.controller_fk.fk(state[7:13])
    hardware = ctx["observation"]["hardware"]
    stamp = hardware["state_monotonic_ns"]
    hardware.update(measured_state=state.tolist(), right_tcp_xyz_m=tcp.tolist(),
        right_tcp_monotonic_ns=stamp, right_tcp_frame="right_arm_local_base",
        camera_health={"front_rgb": {"device_id": FRONT_SERIAL}},
        camera_calibration_sha256={"front_rgb": FRONT_INTRINSICS_SHA256},
        camera_monotonic_ns={"front_rgb": stamp}, depth_monotonic_ns={"front_depth_mm": stamp})
    cloud = PickTubeGraspObserver(provider).cloud(ctx["observation"], {"front_rgb": rgb, "front_depth_mm": depth})
    assert len(cloud.object_camera_m) == 168
    assert np.all(cloud.object_camera_m[:, 2] == .5)
    expected = (provider.transform @ np.r_[cloud.target_camera_m, 1.])[:3] + [0., .5, 0.]
    assert cloud.evidence["target_base_xyz_m"] == pytest.approx(expected)
    hardware["depth_monotonic_ns"]["front_depth_mm"] += 1
    with pytest.raises(ValueError, match="timestamp"):
        PickTubeGraspObserver(provider).cloud(ctx["observation"], {"front_rgb": rgb, "front_depth_mm": depth})


def test_local_graspgen_protocol_centres_only_target_points_and_checks_model_identity(monkeypatch):
    import io
    import json
    from robots.manipulation.grasp_proposals import LocalGraspService
    cloud = Observer().cloud(context()["observation"], {})
    sent = []
    health = {"ready": True, "gripper_id": "panda", "model_sha256": "a" * 64}

    def request(value, timeout):
        if isinstance(value, str):
            return io.BytesIO(json.dumps(health).encode())
        payload = json.loads(value.data)
        sent.append(payload)
        assert np.asarray(payload["point_cloud"]).mean(axis=0) == pytest.approx([0., 0., 0.], abs=1e-8)
        return io.BytesIO(json.dumps({"ok": True, "grasps": [{"transform_model": np.eye(4).tolist(), "score": .8}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    service = LocalGraspService("graspgen", "http://127.0.0.1:18093", expected_gripper="panda", expected_model_sha256="a" * 64)
    proposal = service.propose(cloud, max_candidates=8)
    assert np.asarray(proposal[0]["transform_camera"])[:3, 3] == pytest.approx(cloud.target_camera_m)
    assert len(sent[0]["point_cloud"]) == 40
    assert not sent[0]["filter_collisions"]
    health["model_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="SHA"):
        service.propose(cloud, max_candidates=8)


def test_frozen_real_candidate_catalog_and_contract_pass_static_preflight():
    import json
    from pathlib import Path
    from robots.arx.deployment.real_input import RealInputContract, LiveCapabilities, preflight_real_bundle
    from zetta.evolution.jsonio import file_sha256

    root = Path(__file__).resolve().parents[1]
    directory = root / "docs/experiments/arx-grasp-recovery-20261006"
    path = directory / "real-input-contract.json"
    contract = RealInputContract.model_validate_json(path.read_text())
    live = LiveCapabilities(schema_version="arx.real.capabilities.v1", robot_id="static-test-only",
        cameras=contract.cameras, depth_cameras=contract.depth_cameras,
        joint_channels=contract.joint_channels, auxiliary_channels=contract.auxiliary_channels,
        feature_sources=contract.feature_sources, tool_catalog_sha256=contract.tool_catalog_sha256)
    result = preflight_real_bundle(bundle_path=directory / "candidate.json",
        task_manifest_path=root / "robots/arx/manifests/pickup_test_tube.yaml",
        model_contract_path=root / "robots/arx/manifests/task7_model_a.yaml",
        tool_catalog_path=directory / "tool-catalog.json", real_contract_path=path,
        live_capabilities=live, expected_real_contract_sha256=file_sha256(path))
    assert result["eligible"]
    assert result["recovery_plans"][0]["tool_calls"] == 6
