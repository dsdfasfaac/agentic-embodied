"""Simulation encoding is adapted without changing learned bundle bytes."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.arx.deployment.bundle_program import compile_programs, resolve_call
from robots.arx.deployment.real_input import _check_threshold, _load_bundle
from robots.arx.gateway.bundle_runtime import BundleMonitor, RealBundleReentry

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'docs/experiments/arx-real2sim2real-zeroshot-20261009/bundle.json'


def test_original_bundle_label_is_bound_to_fresh_review():
    bundle, _ = _load_bundle(BUNDLE)
    program = next(iter(compile_programs(bundle).values()))
    assert [c.tool for c in program.calls] == [
        'arx.set_gripper', 'arx.hold', 'arx.review_reentry', 'arx.zeva']
    assert program.binding.max_recovery_steps == 89
    assert program.calls[-1].arguments['reentry_token'] == 'token-preflight'
    assert bundle.recovery_rules[0].steps[-1].parameters['reentry_token'] == 'alignment_open_settle_once'
    with pytest.raises(ValueError, match='did not grant'):
        resolve_call(program.calls[-1], 'obs-live', None)
    assert resolve_call(program.calls[-1], 'obs-live', 'live-review-token')['reentry_token'] == 'live-review-token'


@pytest.mark.parametrize('operator,value', [('le', .5), ('ge', 1.), ('eq', 0.), ('eq', False)])
def test_boolean_sources_accept_simulation_binary_thresholds(operator, value):
    _check_threshold(SimpleNamespace(name='binary', scalar_type='boolean'), operator, value)


@pytest.mark.parametrize('operator,value', [('le', -1), ('ge', 2), ('le', float('nan')), ('stagnant', 0), ('ge', True)])
def test_binary_thresholds_reject_invalid_types_and_ranges(operator, value):
    with pytest.raises(ValueError):
        _check_threshold(SimpleNamespace(name='binary', scalar_type='boolean'), operator, value)


def test_numeric_binary_dwell_unknown_and_measured_reentry():
    bundle, _ = _load_bundle(BUNDLE)
    prefix = 'privileged.interaction.'
    values = {prefix+'gripper_closed': True, prefix+'gripper_contact': False,
              prefix+'grasped': False, prefix+'lift_m': 0., prefix+'success': False}
    class Provider:
        sources = [SimpleNamespace(name=name, scalar_type='number' if name.endswith('lift_m') else 'boolean',
                                   provider_sha256='a'*64) for name in values]
        def augment(self, observation, images):
            return dict(observation, **values)
    monitor = BundleMonitor(bundle, Provider(), terminal_feature=prefix+'success')
    def obs(step, **extra):
        return dict(observation_id=f'obs-{step}', step_index=step, **extra)
    for step in range(1, 12):
        assert not monitor.observe(obs(step), {}).events
    assert monitor.observe(obs(12), {}).events
    assert monitor.last_feature_evidence['features'][prefix+'gripper_closed'] is True
    assert monitor.last_feature_evidence['critic_numeric_binary_features'][prefix+'gripper_closed'] == 1.
    assert monitor.completion_evidence() is None
    monitor.temporal.reset()
    for step in range(1, 8):
        monitor.observe(obs(step), {})
    values[prefix+'gripper_contact'] = None
    quality = {'status': 'unknown', 'unavailable_features': [prefix+'gripper_contact']}
    assert not monitor.observe(obs(8, feature_observation=quality), {}).events
    values[prefix+'gripper_contact'] = False
    for step in range(9, 20):
        assert not monitor.observe(obs(step), {}).events
    assert monitor.observe(obs(20), {}).events
    reviewer = RealBundleReentry(bundle, Provider(), require_hardware=False)
    context = {'observations': [obs(21)], 'images': [{}], 'recovery_id': 'incident',
               'policy_id': bundle.recovery_rules[0].recovery_id}
    assert reviewer.inspect(None, context)['status'] == 'ineligible'
    values[prefix+'gripper_closed'] = False
    assert reviewer.inspect(None, context)['status'] == 'eligible'
    values[prefix+'success'] = True
    monitor.observe(obs(22), {})
    assert monitor.completion_evidence()['observation_id'] == 'obs-22'
