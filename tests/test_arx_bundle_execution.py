"""Bundle calls are executable constraints, including order and reentry evidence."""

import json

import pytest

from robots.arx.deployment.bundle_program import compile_programs, verify_call_result
from robots.arx.gateway.bundle_runtime import RealBundleReentry
from tests.test_arx_deployment import runner
from tests.test_arx_gateway import ScriptCritic, call, limits, make_core
from zetta.evolution.models import CandidateBundle, CriticRule, RecoveryRule, RecoveryStep


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
    first = call(core)
    assert first["status"] == "interrupted"
    rejected = call(core, "arx.review_reentry", {"observation_ids": [core.current["observation_id"]]})
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
