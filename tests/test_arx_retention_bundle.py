"""Recorded real failure must interrupt before the target disappears."""
import json
from pathlib import Path

from robots.arx.deployment.real_input import _load_bundle
from robots.arx.deployment.bundle_program import compile_programs
from zetta.evolution.critic import TemporalCritic

EXPERIMENT = Path(__file__).resolve().parents[1] / 'docs/experiments/arx-retention-recovery-20261007'


def test_recorded_rim_contact_and_separation_trigger_before_final_occlusion():
    bundle, _ = _load_bundle(EXPERIMENT / 'bundle.json')
    rules = tuple(r for r in bundle.critic_rules if r.rule_id != 'commission_far_open')
    critic = TemporalCritic(rules)
    frames = json.loads((EXPERIMENT / 'evidence/attempt10-retention-frames.json').read_text())['frames']
    events = []
    for frame in frames:
        step = int(frame['observation_id'].split('-')[1])
        events.extend(critic.evaluate(frame['features'], step_index=step,
            unavailable_features=set(frame.get('feature_observation', {}).get('unavailable_features', []))))
    assert [(e['rule_id'], e['step_index']) for e in events] == [
        ('closed_contact_no_lift', 244), ('closed_target_separated', 251)]
    assert all(e['environment_write'] is False for e in events)


def test_candidate_compiles_engage_close_lift_and_audited_vla_reentry_within_budget():
    bundle, _ = _load_bundle(EXPERIMENT / 'bundle.json')
    program = compile_programs(bundle, max_physical_steps=600, max_tool_calls=16)['pink_engage_retention_then_vla']
    assert program.binding.max_recovery_steps == 561 and len(program.calls) == 15
    assert program.calls[0].tool == 'arx.set_gripper' and program.calls[0].arguments['opening'] == 1.
    phases = [c.arguments['phase'] for c in program.calls if c.tool == 'arx.execute_grasp']
    assert phases == ['pregrasp', 'pregrasp', 'engage', 'lift']
    assert program.calls[9].tool == 'arx.set_gripper' and program.calls[9].arguments['opening'] == 0.
    assert [c.tool for c in program.calls[-3:]] == ['arx.hold', 'arx.review_reentry', 'arx.zeva']
