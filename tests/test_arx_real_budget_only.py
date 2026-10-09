"""Budget adaptation must not change the simulation candidate's learned policy."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.arx.deployment.bundle_program import compile_programs
from robots.arx.deployment.real_input import _load_bundle
from robots.arx.gateway.bundle_runtime import BundleMonitor
from tests.test_arx_gateway import call, limits, make_core
from zetta.evolution.jsonio import file_sha256

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'docs/experiments/arx-real2sim2real-budgetonly-20261009'
ORIGINAL = ROOT / 'docs/experiments/arx-real2sim2real-zeroshot-20261009/bundle.json'
PREFIX = 'privileged.interaction.'
DISTANCE = 'privileged.selected.target_gripper_distance_m'


def candidate():
    return _load_bundle(EXP / 'bundle.json')[0]


def observation(step, **extra):
    return dict(observation_id=f'obs-{step}', step_index=step, **extra)


def monitored(distance=.42):
    values = {PREFIX+'gripper_closed': True, PREFIX+'gripper_contact': False,
              PREFIX+'grasped': False, PREFIX+'lift_m': 0., PREFIX+'success': False,
              DISTANCE: distance}

    class Provider:
        sources = [SimpleNamespace(name=k, provider_sha256='a'*64,
                                   scalar_type='boolean' if type(v) is bool else 'number')
                   for k, v in values.items()]

        def augment(self, obs, images):
            return dict(obs, **values)

    monitor = BundleMonitor(candidate(), Provider(), terminal_feature=PREFIX+'success')
    monitor.reset(observation(0), {})
    return monitor, values


def test_only_actuator_budget_and_candidate_lineage_change():
    original = json.loads(ORIGINAL.read_text())
    adapted = json.loads((EXP / 'bundle.json').read_text())
    assert adapted['critic_rules'] == original['critic_rules']
    assert adapted['parent_sha256'] == file_sha256(ORIGINAL)
    assert adapted['generation'] == original['generation'] + 1
    assert adapted['candidate_id'] != original['candidate_id']
    restored = copy.deepcopy(adapted)
    assert restored['recovery_rules'][0]['steps'][0]['parameters']['max_steps'] == 60
    restored['recovery_rules'][0]['steps'][0]['parameters']['max_steps'] = 20
    for key in ('candidate_id', 'generation', 'parent_sha256', 'mechanism_change'):
        restored[key] = original[key]
    assert restored == original


@pytest.mark.parametrize('distance', [.42, .06, None])
def test_original_rule_interrupts_after_twelve_actions_without_distance_gate(distance):
    monitor, _ = monitored(distance)
    for step in range(1, 12):
        assert not monitor.observe(observation(step), {}).events
    events = monitor.observe(observation(12), {}).events
    assert [event.rule_id for event in events] == [candidate().critic_rules[0].rule_id]


def test_unknown_required_feature_breaks_dwell_but_readonly_frames_do_not_count():
    monitor, values = monitored()
    for step in range(1, 8):
        assert not monitor.observe(observation(step), {}).events
    for _ in range(20):
        assert not monitor.observe(observation(7), {}).events
    values[PREFIX+'gripper_contact'] = None
    quality = {'status': 'unknown', 'unavailable_features': [PREFIX+'gripper_contact']}
    assert not monitor.observe(observation(7, feature_observation=quality), {}).events
    values[PREFIX+'gripper_contact'] = False
    for step in range(8, 19):
        assert not monitor.observe(observation(step), {}).events
    assert monitor.observe(observation(19), {}).events


def test_original_fifty_action_cooldown_survives_opening_and_readonly_frames():
    monitor, values = monitored()
    for step in range(1, 13):
        events = monitor.observe(observation(step), {}).events
    assert events
    values[PREFIX+'gripper_closed'] = False
    for step in range(13, 33):
        assert not monitor.observe(observation(step), {}).events
    for _ in range(100):
        assert not monitor.observe(observation(32), {}).events
    values[PREFIX+'gripper_closed'] = True
    for step in range(33, 74):
        assert not monitor.observe(observation(step), {}).events
    assert monitor.observe(observation(74), {}).events  # 12 + 50 cooldown + 12 dwell


def test_compilation_preserves_motion_order_and_reserves_real_action_budget():
    program = next(iter(compile_programs(candidate()).values()))
    assert [c.tool for c in program.calls] == [
        'arx.set_gripper', 'arx.hold', 'arx.review_reentry', 'arx.zeva']
    assert [c.step_index for c in program.calls] == [0, 1, 2, 2]
    assert program.calls[0].arguments == {'opening': 1., 'max_steps': 60}
    assert program.calls[1].arguments == {'steps': 5}
    assert program.calls[-1].arguments['max_chunks'] == 4
    assert program.binding.max_recovery_steps == 60 + 5 + 4*16
    assert program.binding.max_agent_decisions == 4
    with pytest.raises(ValueError, match='exceeds frozen budget'):
        compile_programs(candidate(), max_physical_steps=128)
    protocol = json.loads((EXP/'protocol.json').read_text())
    frozen = json.loads((EXP/'frozen/real-input-contract.json').read_text())
    assert frozen['max_critic_cooldown_steps'] == 50
    assert protocol['bundle_file_sha256'] == file_sha256(EXP/'bundle.json')
    assert protocol['real_input_sha256'] == file_sha256(EXP/'frozen/real-input-contract.json')
    assert protocol['recovery_budget']['total'] == program.binding.max_recovery_steps


def test_active_recovery_suppresses_same_rule_after_original_cooldown_expires(tmp_path):
    # Exercise gateway suppression with a deliberately persistent failure.
    # Holds here simulate elapsed recovery actions; this is not a grasp trial.
    monitor, _ = monitored()
    core, backend, _ = make_core(tmp_path, monitor, config=limits(max_steps=200))
    # Generic holds have a 15-action cap; allow five test calls to cover 63
    # elapsed actions. The actual compiled program remains four calls above.
    binding = next(iter(compile_programs(candidate()).values())).binding
    core.bindings = (binding.model_copy(update={'max_agent_decisions': 5}),)
    try:
        assert call(core, args={'max_chunks': 3})[0]['status'] == 'interrupted'
        assert backend.steps == 12
        for _ in range(4):
            assert call(core, 'arx.hold', {'steps': 15})[0]['status'] == 'completed'
        assert call(core, 'arx.hold', {'steps': 3})[0]['status'] == 'completed'
        assert backend.steps == 75 and core.state == 'RECOVERING'
        suppressed = [json.loads(row[0]) for row in core.journal.db.execute(
            "SELECT payload FROM records WHERE kind='suppressed_proposal' ORDER BY sequence")]
        assert len(suppressed) == 1
        assert suppressed[0]['evidence_observation_ids'] == ['obs-74']
    finally:
        core.close()
