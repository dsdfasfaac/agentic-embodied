"""Bundle calls are executable constraints, including order and reentry evidence."""

import json
import time
from dataclasses import replace

import pytest

from robots.arx.deployment.bundle_program import compile_programs, verify_call_result
from robots.arx.deployment.runner import RunnerError
from robots.arx.gateway.bundle_runtime import RealBundleReentry, RealFeatureProvider
from robots.arx.gateway.tools import RegisteredTool
from tests.test_arx_deployment import runner
from tests.test_arx_gateway import ScriptCritic, call, limits, make_core
from zetta.evolution.models import CandidateBundle, CriticRule, RecoveryRule, RecoveryStep
from zetta.evolution.jsonio import file_sha256


def bundle():
    return CandidateBundle(
        candidate_id="test", generation=0, parent_sha256=None,
        diagnosis_sha256="a" * 64, causal_hypothesis="test failure",
        mechanism_change="hold then review", validation_plan="bounded trial",
        critic_rules=(CriticRule(
            rule_id="a", title="test", feature="real_error", operator="gt",
            threshold=0.1, dwell_steps=1, cooldown_steps=0,
            proposal="interrupt", evidence_ids=("obs",),
        ),),
        recovery_rules=(RecoveryRule(
            recovery_id="recover", title="test", trigger_rule_ids=("a",),
            precondition="error", steps=(
                RecoveryStep("arx.hold", {"steps": 1}, "hold observed"),
                RecoveryStep("arx.review_reentry", {"observation_ids": ["post-recovery"]}, "eligible"),
                RecoveryStep("arx.zeva", {"max_chunks": 1, "reentry_token": "token-from-review"}, "resumed"),
            ), safety_constraints=("named tools only",), stop_condition="complete",
            fallback="stop_all_motion", evidence_ids=("obs",),
        ),),
    )


def test_bundle_program_enforces_order_and_runner_records_steps(tmp_path):
    program = compile_programs(bundle())["recover"]
    core, backend, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}), config=limits(max_steps=8))
    core.bindings = (program.binding,)
    core.programs = {"recover": program}
    first, _ = call(core)
    assert first["status"] == "interrupted"
    rejected, _ = call(core, "arx.review_reentry", {"observation_ids": [core.current["observation_id"]]})
    assert rejected["status"] == "rejected"
    assert rejected["error"]["code"] == "BUNDLE_STEP_MISMATCH"
    r = runner(tmp_path / "run", core)
    r.trial.candidate = type("Candidate", (), {"package_sha256": bundle().sha256})()
    r._structured_bundle = True
    r.bundle_programs = {"recover": program}
    assert r.loop() == "environment_ended"
    records = sorted((tmp_path / "run/recovery").glob("*.json"))
    assert len(records) == 3
    assert [json.loads(path.read_text())["tool"] for path in records] == [
        "arx.hold", "arx.review_reentry", "arx.zeva",
    ]
    assert r.outcome.reentry_completed
    assert r.counts.agent_calls == 0
    assert backend.steps == 8


def test_reentry_uses_real_health_arrival_and_rule_clearance():
    class Provider:
        def augment(self, observation, images):
            return dict(observation, real_error=observation["real_error"])

    reviewer = RealBundleReentry(
        bundle(), Provider(), max_sensor_age_ms=100, max_sensor_skew_ms=30,
    )
    observation = {
        "observation_id": "obs-2", "real_error": 0.0,
        "hardware": {
            "arrival_verified": True, "sensor_age_ms": 5.0,
            "sensor_skew_ms": 2.0,
            "device_health": {"transport_responsive": True, "fault_codes": []},
            "camera_health": {name: {"responsive": True} for name in
                              ("front_rgb", "left_rgb", "right_rgb")},
        },
    }
    context = {"observations": [observation], "images": [{}],
               "recovery_id": "incident-1", "policy_id": "recover"}
    args = type("Args", (), {"observation_ids": ["obs-2"]})()
    assert reviewer.inspect(args, context)["status"] == "eligible"
    observation["hardware"]["arrival_verified"] = False
    assert reviewer.inspect(args, context)["status"] == "ineligible"
    observation["hardware"]["arrival_verified"] = True
    observation["real_error"] = .2
    assert reviewer.inspect(args, context)["status"] == "ineligible"


def test_review_denial_does_not_grant_reentry_token():
    program = compile_programs(bundle())["recover"]
    review = program.calls[1]
    with pytest.raises(ValueError, match="not satisfied"):
        verify_call_result(review, {
            "status": "completed", "result": {
                "assessment": {"status": "ineligible"}, "reentry_token": None,
            },
        }, real=True)


def test_denied_reentry_stops_runner_and_records_failure_observation(tmp_path):
    class Denied:
        def inspect(self, args, context):
            return {
                "recovery_id": context["recovery_id"],
                "observation_id": args.observation_ids[-1],
                "policy_id": context["policy_id"], "status": "ineligible",
                "checks": [{"check_id": "real-clearance", "status": "fail",
                            "evidence_ids": args.observation_ids,
                            "reason_code": "not_clear"}],
            }

    program = compile_programs(bundle())["recover"]
    core, backend, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}), config=limits(max_steps=8))
    core.bindings = (program.binding,)
    core.programs = {"recover": program}
    old = core.registry._tools["arx.review_reentry"]
    core.registry._tools["arx.review_reentry"] = RegisteredTool(old.spec, Denied())
    r = runner(tmp_path / "run", core)
    r.trial.candidate = type("Candidate", (), {"package_sha256": bundle().sha256})()
    r._structured_bundle = True
    r.bundle_programs = {"recover": program}
    with pytest.raises(RunnerError, match="recovery_step_failed"):
        r.loop()
    failures = list((tmp_path / "run/recovery").glob("*-failure.json"))
    assert len(failures) == 1
    evidence = json.loads(failures[0].read_text())
    assert evidence["observation_before"]["observation_id"] == "obs-2"
    assert evidence["observation_after"]["observation_id"] == "obs-2"
    assert evidence["tool_result"]["status"] == "failed"
    assert backend.steps == 2


def test_eef_expansion_reserves_planner_physical_budget():
    original = bundle()
    recovery = original.recovery_rules[0]
    steps = (
        RecoveryStep("arx.set_gripper", {"opening": 1.0, "max_steps": 15}, "open"),
        RecoveryStep("arx.move_eef", {
            "delta_xyz_m": [0.0, 0.0, 0.02], "frame": "tool", "speed_m_s": 0.01,
        }, "move"),
        *recovery.steps[1:],
    )
    program = compile_programs(replace(original, recovery_rules=(replace(recovery, steps=steps),)))["recover"]
    assert [call.tool for call in program.calls] == [
        "arx.set_gripper", "arx.move_eef", "arx.move_eef", "arx.review_reentry", "arx.zeva",
    ]
    assert program.binding.max_recovery_steps == 75
    assert program.binding.max_agent_decisions == 5
    with pytest.raises(ValueError, match="budget"):
        compile_programs(replace(original, recovery_rules=(replace(recovery, steps=steps),)),
                         max_physical_steps=74)


def test_sha_pinned_real_feature_provider_requires_fresh_joint_feedback(tmp_path):
    module = tmp_path / "provider.py"
    module.write_text('''
from pathlib import Path
from zetta.evolution.jsonio import file_sha256
class Provider:
    def feature_sources(self):
        return [{"name": "real_error", "provider_id": "test",
                 "provider_sha256": file_sha256(Path(__file__)),
                 "source_kind": "joint_feedback", "source_ids": ["right_joint_1"],
                 "scalar_type": "number", "units": "rad", "max_age_ms": 100}]
    def observe(self, observation, images):
        return {"real_error": float(observation["hardware"]["measured_state"][7])}
def create_provider():
    return Provider()
''')
    provider = RealFeatureProvider(module, file_sha256(module))
    stamp = time.monotonic_ns()
    observation = {"hardware": {"state_monotonic_ns": stamp,
                                "measured_state": [0.] * 14}}
    assert provider.augment(observation, {})["real_error"] == 0.
    observation["hardware"]["state_monotonic_ns"] = stamp - 500_000_000
    with pytest.raises(ValueError, match="stale"):
        provider.augment(observation, {})


def test_real_feature_freshness_uses_acquisition_time_after_journaling(tmp_path):
    module = tmp_path / "provider.py"
    module.write_text('''
from pathlib import Path
from zetta.evolution.jsonio import file_sha256
class Provider:
    def feature_sources(self):
        return [{"name": "joint", "provider_id": "test",
                 "provider_sha256": file_sha256(Path(__file__)),
                 "source_kind": "joint_feedback", "source_ids": ["right_joint_1"],
                 "scalar_type": "number", "units": "rad", "max_age_ms": 100}]
    def observe(self, observation, images):
        return {"joint": 0.0}
def create_provider():
    return Provider()
''')
    provider = RealFeatureProvider(module, file_sha256(module))
    now = time.monotonic_ns()
    hardware = {"state_monotonic_ns": now - 200_000_000,
                "observation_completed_ns": now - 190_000_000}
    assert provider.augment({"hardware": hardware}, {})["joint"] == 0.0
    hardware["observation_completed_ns"] = now - 90_000_000
    with pytest.raises(ValueError, match="stale"):
        provider.augment({"hardware": hardware}, {})
