#!/usr/bin/env python3
"""Verify bounded empty-jaw close/open at the frozen home pose; never call VLA.

Requires an onsite supervised session and an already staged, enabled controller.
No reentry capability is granted. Results are commissioning evidence, not a
candidate rescue or task success. On failure, keep controller enabled for review.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.gateway.contracts import ReviewArgs, ToolRequest
from robots.arx.gateway.real_factory import RealCoreFactory

PREFIX = 'privileged.interaction.'
DISTANCE = 'privileged.selected.target_gripper_distance_m'


def require_empty(features, *, closed):
    if features.get('feature_observation', {}).get('status') != 'observed':
        raise ValueError('empty-jaw commissioning requires observed features')
    values = features['features']
    if values.get(PREFIX+'gripper_closed') is not closed or any(
        values.get(PREFIX+name) is not False for name in ('gripper_contact', 'grasped', 'success')
    ):
        raise ValueError('empty jaw state is not verified')
    distance = values.get(DISTANCE)
    if type(distance) not in (int, float) or not np.isfinite(distance) or distance < .25:
        raise ValueError('commissioning requires a target-clear home area')


def commission(experiment: Path, hardware: Path, output: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((experiment/'protocol.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    private = output/'private/gateway'
    private.mkdir(parents=True)
    result = {'schema_version': 'arx.real.gripper.commissioning.v1',
              'status': 'checking', 'termination_reason': 'gripper_commissioning_failed',
              'candidate_sha256': protocol['bundle_semantic_sha256'],
              'task_success': False, 'recovery_attempted': False,
              'reentry_token_granted': False, 'vla_calls': 0, 'tools': []}
    core = None
    try:
        core = RealCoreFactory(
            hardware_config=str(hardware), hardware_sha256=protocol['hardware_sha256'],
            task=str(root/'robots/arx/manifests/pickup_test_tube.yaml'),
            model_contract=str(root/'robots/arx/manifests/task7_model_a.yaml'),
            output=str(private), episode_id=uuid.uuid4().hex,
            limits=json.loads((experiment/'runtime-limits.json').read_text()),
            zeva_host='127.0.0.1', zeva_port=5583, kinematics_calibration=None,
            bundle=str(experiment/'bundle.json'), tool_catalog=str(experiment/'frozen/tool-catalog.json'),
            real_input_contract=str(experiment/'frozen/real-input-contract.json'),
            expected_real_input_sha256=protocol['real_input_sha256'],
            feature_provider=str(root/'robots/arx/deployment/picktube_rgbd_provider.py'),
            expected_feature_provider_sha256=protocol['feature_provider_sha256'],
        )(lambda: False, lambda _: None)
        core.reset()  # Existing backend enforces measured frozen home before admission.
        require_empty(core.critic.last_feature_evidence, closed=False)
        initial_right = core.commit.command[7:13].copy()
        for tool, arguments, closed in (
            ('arx.set_gripper', {'opening': 0., 'max_steps': 60}, True),
            ('arx.set_gripper', {'opening': 1., 'max_steps': 60}, False),
            ('arx.hold', {'steps': 5}, False),
        ):
            identity = uuid.uuid4().hex
            request = ToolRequest(request_id=identity, decision_ref=identity,
                                  observation_id=core.current['observation_id'],
                                  control_epoch=core.epoch, tool=tool, arguments=arguments,
                                  reason='Supervised empty-jaw commissioning at measured home; no VLA')
            core.journal.register_decision(request, source='runner', evidence={'scope': 'commissioning'})
            value = core.execute(request)
            result['tools'].append({'request': request.model_dump(), 'result': value,
                                    'feature_evidence': core.critic.last_feature_evidence})
            (output/'progress.json').write_text(json.dumps(result, indent=2)+'\n')
            evidence = value.get('result') or {}
            if value['status'] != 'completed' or evidence.get('physical_arrival_verified') is not True:
                raise ValueError('commissioning tool did not physically complete: '+tool)
            if tool == 'arx.set_gripper' and evidence.get('command_target_reached') is not True:
                raise ValueError('gripper endpoint not reached')
            require_empty(core.critic.last_feature_evidence, closed=closed)
            if np.max(np.abs(core.commit.command[7:13]-initial_right)) > 1e-6:
                raise ValueError('in-place commissioning issued an arm displacement')
        # Inspect the real reviewer without manufacturing a recovery incident or token.
        handler = core.registry.resolve('arx.review_reentry').handler
        assessment = handler.inspect(ReviewArgs(observation_ids=[core.current['observation_id']]), {
            'observations': [core.current], 'images': [core._images()],
            'recovery_id': 'commissioning-open-check', 'policy_id': next(iter(core.programs)),
        })
        result['readonly_reentry_assessment'] = assessment
        if assessment['status'] != 'eligible':
            raise ValueError('open/unheld reentry state not verified')
        commands = [json.loads(row[0]) for row in core.journal.db.execute(
            'SELECT payload FROM records WHERE kind=? ORDER BY sequence', ('command_sent',))]
        if any(np.max(np.abs(np.asarray(row['target'])[7:13]-initial_right)) > 1e-6 for row in commands):
            raise ValueError('in-place command audit found an arm displacement')
        core._record('commissioning_reentry_assessment', assessment)
        result.update(status='completed', termination_reason='gripper_commissioning_complete',
                      physical_steps=core.step_index, right_arm_command_unchanged=True)
    except Exception as exc:
        result.update(status='failed_keep_enabled', error=str(exc))
    finally:
        if core is not None:
            result['final_observation'] = core.current
            core.close()
            core.journal.close()
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--hardware-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required for supervised empty-jaw motion')
    result = commission(args.experiment, args.hardware_config, args.output)
    print(json.dumps({'status': result['status'], 'error': result.get('error'),
                      'physical_steps': result.get('physical_steps'),
                      'tool_steps': [v['result']['executed_steps'] for v in result['tools']]}))
    raise SystemExit(0 if result['status'] == 'completed' else 2)


if __name__ == '__main__':
    main()
