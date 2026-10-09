"""Adaptation targets measured failures and retains actual recovery postconditions."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.arx.contracts import load_task_manifest
from robots.arx.control import ActionProcessor
from robots.arx.deployment.bundle_program import compile_programs
from robots.arx.deployment.real_input import _load_bundle
from robots.arx.gateway.bundle_runtime import BundleMonitor, RealBundleReentry
from robots.arx.gateway.contracts import GripperArgs
from robots.arx.gateway.tools import ApprovedToolContext, PolicyGripperPlanner

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'docs/experiments/arx-real2sim2real-adapted-20261009/bundle.json'
PREFIX = 'privileged.interaction.'
DISTANCE = 'privileged.selected.target_gripper_distance_m'


def candidate():
    return _load_bundle(PATH)[0]


def monitored():
    values = {PREFIX+'gripper_closed': True, PREFIX+'gripper_contact': False,
              PREFIX+'grasped': False, PREFIX+'lift_m': 0., PREFIX+'success': False,
              DISTANCE: .42}
    class Provider:
        sources = [SimpleNamespace(name=k, provider_sha256='a'*64,
                                   scalar_type='boolean' if type(v) is bool else 'number')
                   for k, v in values.items()]
        def augment(self, obs, images):
            return dict(obs, **values)
    provider = Provider()
    monitor = BundleMonitor(candidate(), provider, terminal_feature=PREFIX+'success')
    monitor.reset(obs(0), {})
    return monitor, provider, values


def obs(step, **extra):
    return dict(observation_id=f'obs-{step}', step_index=step, **extra)


def test_far_empty_closure_does_not_interrupt_but_near_12_actions_does():
    monitor, _, values = monitored()
    for step in range(1, 38):
        assert not monitor.observe(obs(step), {}).events
    values[DISTANCE] = .075
    for step in range(38, 49):
        assert not monitor.observe(obs(step), {}).events
    assert monitor.observe(obs(49), {}).events
    program = next(iter(compile_programs(candidate()).values()))
    assert program.binding.max_recovery_steps == 129
    assert program.calls[0].arguments['max_steps'] == 60
    assert candidate().parent_sha256 == '7d922aeff6d64b77a7c0db938aad7052d4a80c29baa2f76a21b350ef306d6f7c'


def test_near_rule_counts_physical_actions_and_unknown_breaks_dwell():
    monitor, _, values = monitored()
    values[DISTANCE] = .06
    for step in range(1, 8):
        assert not monitor.observe(obs(step), {}).events
    for _ in range(20):
        assert not monitor.observe(obs(7), {}).events
    values[DISTANCE] = None
    quality = {'status': 'unknown', 'unavailable_features': [DISTANCE]}
    assert not monitor.observe(obs(7, feature_observation=quality), {}).events
    values[DISTANCE] = .06
    for step in range(8, 19):
        assert not monitor.observe(obs(step), {}).events
    assert monitor.observe(obs(19), {}).events


def test_cooldown_survives_open_guard_and_is_not_consumed_by_readonly_frames():
    monitor, _, values = monitored()
    values[DISTANCE] = .06
    for step in range(1, 13):
        event = monitor.observe(obs(step), {}).events
    assert event
    values[PREFIX+'gripper_closed'] = False
    for step in range(13, 73):
        assert not monitor.observe(obs(step), {}).events
    for _ in range(150):
        assert not monitor.observe(obs(72), {}).events
    values[PREFIX+'gripper_closed'] = True
    for step in range(73, 153):
        assert not monitor.observe(obs(step), {}).events
    assert monitor.observe(obs(153), {}).events  # 12 + 129 cooldown + 12 new actions


@pytest.mark.parametrize('initial_gripper', [-.93717, 0., .05])
def test_sixty_opening_actions_converge_under_real_filter_and_step_limit(initial_gripper):
    task = load_task_manifest(ROOT/'robots/arx/manifests/pickup_test_tube.yaml')
    start = np.asarray(task.start_state, dtype=np.float32)
    start[13] = initial_gripper
    processor = ActionProcessor(task, start)
    context = ApprovedToolContext(start, {})
    planner = PolicyGripperPlanner(closed_policy=0., open_policy=-3.4,
                                   max_policy_step=task.control.max_gripper_step)
    plan = planner.prepare(GripperArgs(opening=1., max_steps=60), context)
    count = 0
    for raw in plan.next_targets(context):
        command, info = processor.process(raw)
        assert info.output_gripper_step <= .080001
        assert command[13] <= context.command[13]
        context = ApprovedToolContext(command, {})
        plan.on_commit(context)
        count += 1
        if plan.reached:
            break
    assert plan.reached and count <= 60
    assert abs(context.command[13] + 3.4) <= .001
    # +0.9 is applied by the device, so this model-space endpoint maps to -2.5.
    assert abs(context.command[13] + .9 + 2.5) <= .001


def test_reentry_denies_far_closed_or_possible_held_target_even_when_rule_clear():
    _, provider, values = monitored()
    reviewer = RealBundleReentry(candidate(), provider, require_hardware=False)
    context = {'observations': [obs(1)], 'images': [{}], 'recovery_id': 'incident',
               'policy_id': candidate().recovery_rules[0].recovery_id}
    assert reviewer.inspect(None, context)['status'] == 'ineligible'
    values[PREFIX+'gripper_closed'] = False
    assert reviewer.inspect(None, context)['status'] == 'eligible'
    values[PREFIX+'grasped'] = True
    assert reviewer.inspect(None, context)['status'] == 'ineligible'
    values[PREFIX+'grasped'] = None
    assert reviewer.inspect(None, context)['status'] == 'ineligible'
