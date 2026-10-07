"""Read-only verification of an interrupted bundle closing checkpoint.

No review token is restored. Measurements restore target identity and lift
reference only; every subsequent write uses a new synchronized observation.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import numpy as np

from .bundle_program import resolve_call, retain_tool_outputs, verify_call_result
from .feature_observation import FeatureObservationUnavailable
from zetta.evolution.jsonio import file_sha256


def read_checkpoint(trial, expected_journal_sha256, program, package_sha256):
    trial = Path(trial); gateway = trial / 'private/gateway'
    journal = gateway / 'journal.sqlite3'
    if file_sha256(journal) != expected_journal_sha256:
        raise ValueError('checkpoint journal SHA differs')
    outcome = json.loads((trial / 'result.json').read_text())
    if (outcome['termination_reason'] != 'observation_unavailable'
            or outcome['package_sha256'] != package_sha256
            or outcome['counts']['recoveries'] != 1):
        raise ValueError('continuation requires one interrupted recovery of the same bundle')
    with sqlite3.connect(f'file:{journal}?mode=ro', uri=True) as db:
        records = [(kind, json.loads(payload)) for kind, payload in db.execute(
            'select kind,payload from records order by sequence')]
        requests = {rid: json.loads(raw) for rid, raw in db.execute('select request_id,request from operations')}
        snapshot = json.loads(db.execute('select payload from snapshot').fetchone()[0])
    commands = [v for k,v in records if k == 'command_sent']
    arrivals = [v for k,v in records if k == 'arrival_observed']
    steps = outcome['counts']['physical_steps']
    if len(commands) != steps or len(arrivals) != steps or not all(v['verified'] is True for v in arrivals):
        raise ValueError('checkpoint has unverified physical writes')
    start = next(i for i,(k,v) in enumerate(records) if k == 'interrupt' and v.get('recovery_context'))
    recovery = deepcopy(records[start][1]['recovery_context'])
    if recovery['binding_id'] != program.binding.binding_id:
        raise ValueError('checkpoint recovery binding differs')
    acknowledged = next(i for i,(k,v) in enumerate(records[start+1:], start+1)
                        if k == 'recovery_acknowledged' and v['binding_id'] == recovery['binding_id'])
    nominal = [v for k,v in records[start+1:acknowledged] if k == 'tool_result']
    if (len(nominal) != 1 or nominal[0]['tool'] != 'arx.zeva'
            or nominal[0].get('result',{}).get('completion') != 'critic_interrupted'):
        raise ValueError('checkpoint has no verified nominal-to-recovery boundary')
    calls = [v for k,v in records[acknowledged+1:] if k == 'tool_result' and v['tool'] != 'arx.finish']
    outputs, phases, proposals = {}, {}, {}
    cursor, spent, close_spent = 0, 0, None
    for index,result in enumerate(calls):
        if cursor >= len(program.calls): raise ValueError('checkpoint exceeds bundle program')
        entry = program.calls[cursor]; request = requests[result['request_id']]
        expected = resolve_call(entry, request['observation_id'], None, outputs)
        # Compare the same typed arguments used by gateway admission.
        from robots.arx.deployment.real_input import _TOOL_MODELS
        model = _TOOL_MODELS[entry.tool]
        if (request['tool'] != entry.tool or model.model_validate(request['arguments']).model_dump()
                != model.model_validate(expected).model_dump()):
            raise ValueError('checkpoint diverges from frozen recovery program')
        spent += result['executed_steps']
        if result['status'] != 'completed':
            if (index != len(calls)-1 or entry.tool != 'arx.set_gripper'
                    or entry.arguments.get('opening') != 0.
                    or result.get('result',{}).get('completion') != 'observation_unavailable'):
                raise ValueError('only an interrupted closing call can continue')
            close_spent = result['executed_steps']; break
        verify_call_result(entry, result, real=True)
        value = result.get('result') or {}
        if entry.tool == 'arx.propose_grasp': proposals[value['proposal_id']] = value
        if entry.tool == 'arx.execute_grasp':
            phases.setdefault(outputs['proposal_id'], set()).add(request['arguments']['phase'])
        retain_tool_outputs(entry, result, outputs)
        cursor += 1
    if close_spent is None or 'engage' not in phases.get(outputs.get('proposal_id'), set()):
        raise ValueError('checkpoint has no verified engage before closing')
    remaining = program.calls[cursor].arguments['max_steps'] - close_spent
    if remaining <= 0: raise ValueError('closing call budget exhausted')
    updated = list(program.calls)
    updated[cursor] = replace(updated[cursor], arguments={**updated[cursor].arguments, 'max_steps':remaining})
    recovery['remaining_steps'] -= spent
    recovery['remaining_decisions'] -= len(calls)
    recovery['last_execution_status'] = 'interrupted'
    if recovery['remaining_steps'] <= 0 or recovery['remaining_decisions'] <= 0:
        raise ValueError('continuation recovery budget exhausted')
    return dict(gateway=gateway, records=records, recovery=recovery, program=replace(program,calls=tuple(updated)),
        cursor=cursor, outputs=outputs, phases=phases, proposals=proposals,
        expected_feedback=snapshot['observation']['hardware']['measured_state'], steps=steps,
        decisions=64-snapshot['budget_remaining']['decisions'], journal_sha256=expected_journal_sha256,
        source_trial=str(trial), closing_steps_consumed=close_spent, closing_steps_remaining=remaining)


def restore_sensor_history(checkpoint, grasp):
    provider = grasp.observer.provider
    seen, clouds = set(), {}
    wanted = {v['observation_id']:pid for pid,v in checkpoint['proposals'].items()}
    history = [(gateway,kind,item) for gateway,records in checkpoint.get('histories', [(checkpoint['gateway'],checkpoint['records'])])
               for kind,item in records]
    for gateway,kind,item in history:
        if kind != 'grasp_sensor_evidence': continue
        observation = item['observation']; oid = observation['observation_id']
        if (gateway,oid) in seen: continue
        seen.add((gateway,oid)); path = gateway / item['path']
        if not path.resolve().is_relative_to(gateway.resolve()):
            raise ValueError('checkpoint sensor path escapes evidence directory')
        if file_sha256(path) != item['sha256']: raise ValueError('checkpoint sensor SHA differs')
        with np.load(path, allow_pickle=False) as archive: images = {k:archive[k] for k in archive.files}
        try: provider.observe(observation, images)
        except FeatureObservationUnavailable: pass
        if oid in wanted: clouds[wanted[oid]] = grasp.observer.cloud(observation, images)
    for pid, phases in checkpoint['phases'].items():
        if pid not in clouds: raise ValueError('proposal sensor evidence is missing')
        grasp.proposals[pid] = dict(output=deepcopy(checkpoint['proposals'][pid]), cloud=clouds[pid], created=grasp.clock())
        grasp.completed_phases[pid] = set(phases)
    # Review capabilities expire across interruption. A fresh review is mandatory.
    grasp.reviews.clear(); checkpoint['outputs'].pop('grasp_review_token', None)
    provider.success_hold_frames = 0; provider.success_latched = False


def extend_closing_checkpoint(checkpoint, segment, expected_sha256):
    """Chain one interrupted closing segment; charge its commands and decision."""
    segment=Path(segment); gateway=segment/'private/gateway'; journal=gateway/'journal.sqlite3'
    if file_sha256(journal)!=expected_sha256: raise ValueError('closing segment SHA differs')
    report=json.loads((segment/'result.json').read_text())
    if (report['schema_version']!='arx.grasp.bundle-continuation.v1'
            or report['termination_reason']!='critic_interrupted'
            or report['source_journal_sha256']!=checkpoint['journal_sha256']
            or report['checkpoint_steps']!=checkpoint['steps']):
        raise ValueError('closing segment provenance differs')
    with sqlite3.connect(f'file:{journal}?mode=ro',uri=True) as db:
        records=[(k,json.loads(v)) for k,v in db.execute('select kind,payload from records order by sequence')]
        operations=[json.loads(v) for v, in db.execute('select request from operations')]
        snapshot=json.loads(db.execute('select payload from snapshot').fetchone()[0])
    calls=[v for k,v in records if k=='tool_result']
    if len(calls)!=1 or len(operations)!=1 or calls[0]['tool']!='arx.set_gripper' or calls[0]['status']!='interrupted':
        raise ValueError('closing segment contains other operations')
    entry=checkpoint['program'].calls[checkpoint['cursor']]
    from robots.arx.gateway.contracts import GripperArgs
    if GripperArgs.model_validate(operations[0]['arguments'])!=GripperArgs.model_validate(entry.arguments):
        raise ValueError('closing segment exceeded frozen closing arguments')
    interrupted=[v for k,v in records if k=='interrupt']
    if (len(interrupted)!=1 or interrupted[0].get('code')!='RECOVERY_ESCALATION_REQUIRED'
            or {v['rule_id'] for v in interrupted[0]['proposals']}!={'closed_contact_no_lift'}):
        raise ValueError('closing segment has an unrelated interruption')
    commands=[v for k,v in records if k=='command_sent']; arrivals=[v for k,v in records if k=='arrival_observed']
    spent=calls[0]['executed_steps']
    if (len(commands)!=spent or len(arrivals)!=spent or not all(v['verified'] is True for v in arrivals)
            or report['total_physical_steps']!=checkpoint['steps']+spent
            or 64-snapshot['budget_remaining']['decisions']!=checkpoint['decisions']+1):
        raise ValueError('closing segment physical/decision counts are unverified')
    remaining=entry.arguments['max_steps']-spent
    if remaining<=0: raise ValueError('closing call budget exhausted')
    calls=list(checkpoint['program'].calls);calls[checkpoint['cursor']]=replace(entry,arguments={**entry.arguments,'max_steps':remaining})
    checkpoint['program']=replace(checkpoint['program'],calls=tuple(calls))
    checkpoint['histories']=[(checkpoint['gateway'],checkpoint['records']),(gateway,records)]
    checkpoint['steps']+=spent; checkpoint['decisions']+=1
    checkpoint['closing_steps_consumed']+=spent;checkpoint['closing_steps_remaining']=remaining
    checkpoint['recovery']['remaining_steps']-=spent;checkpoint['recovery']['remaining_decisions']-=1
    checkpoint['expected_feedback']=snapshot['observation']['hardware']['measured_state']
    checkpoint['closing_segment']=str(segment);checkpoint['closing_segment_sha256']=expected_sha256
    if checkpoint['recovery']['remaining_decisions']<1 or checkpoint['recovery']['remaining_steps']<1:
        raise ValueError('closing segment exhausted recovery budget')


def adopt_checkpoint(core, checkpoint):
    """Initialize a new segment with consumed counts and an observed hold."""
    if core.current is not None or core.journal.snapshot() is not None:
        raise ValueError('continuation must use a fresh segment journal')
    core.backend.set_event_sink(lambda k,v:core._record(k,v))
    commit = core.backend.adopt_observed_hold(np.asarray(checkpoint['expected_feedback']),
                                             checkpoint.get('committed_command'))
    core.step_index = checkpoint['steps']; core.decisions = checkpoint['decisions']; core.incidents = 1
    core.programs[checkpoint['recovery']['binding_id']] = checkpoint['program']
    core.recovery = deepcopy(checkpoint['recovery'])
    core.binding = checkpoint['program'].binding
    core.program_cursor = checkpoint['cursor']; core.program_outputs = deepcopy(checkpoint['outputs'])
    core.state = 'INTERRUPTED'; core.epoch += 1; core.commit = commit
    core._publish(commit, lifecycle='audited-continuation', observation_id=f"obs-{core.step_index}-continuation-0")
    core.critic.reset(core._critic_observation(), core._images()); core._record_feature_evidence()
    evidence = core.critic.last_feature_evidence
    distance = evidence['features']['privileged.selected.target_gripper_distance_m']
    if evidence['feature_observation']['status'] != 'observed' or distance is None or distance > .02:
        raise ValueError('continuation requires fresh same-target depth within 20mm of gripper')
    if checkpoint.get('closing_complete'):
        values=evidence['features']
        if not (values['privileged.interaction.gripper_closed'] is True and
                values['privileged.interaction.gripper_contact'] is True):
            raise ValueError('completed closing requires fresh target-specific closed contact')
        core._record('closing_completion_reverified',dict(observation_id=core.current['observation_id'],
            command_tolerance_policy=1e-3,source_journal_sha256=checkpoint['completed_closing_sha256'],
            prior_result='BUNDLE_STEP_FAILED at stricter command convergence; original result preserved'))
    core._record('closing_continuation_admitted', {k:checkpoint[k] for k in (
        'source_trial','journal_sha256','closing_steps_consumed','closing_steps_remaining','steps','decisions')})
    core._retain_grasp_sensors(); core._save()


def extend_completed_closing(checkpoint, segment, expected_sha256):
    """Accept measured closing rejected solely by the former1e-4 tolerance.

    No command is reissued and no budget is refunded. Fresh closed contact is
    independently mandatory in adopt_checkpoint before lift review.
    """
    segment=Path(segment);gateway=segment/'private/gateway';journal=gateway/'journal.sqlite3'
    if file_sha256(journal)!=expected_sha256:raise ValueError('completed closing SHA differs')
    report=json.loads((segment/'result.json').read_text())
    if (report['checkpoint_steps']!=checkpoint['steps'] or report['closing_segment_sha256']!=checkpoint['closing_segment_sha256']
            or report['source_journal_sha256']!=checkpoint['journal_sha256']):
        raise ValueError('completed closing provenance differs')
    with sqlite3.connect(f'file:{journal}?mode=ro',uri=True) as db:
        records=[(k,json.loads(v)) for k,v in db.execute('select kind,payload from records order by sequence')]
        operations=[json.loads(v) for v, in db.execute('select request from operations')]
        snapshot=json.loads(db.execute('select payload from snapshot').fetchone()[0])
    calls=[v for k,v in records if k=='tool_result'];failed=[v for k,v in records if k=='bundle_step_failed']
    entry=checkpoint['program'].calls[checkpoint['cursor']]
    from robots.arx.gateway.contracts import GripperArgs
    if (len(calls)!=1 or len(operations)!=1 or calls[0]['tool']!='arx.set_gripper'
            or calls[0]['error']['code']!='BUNDLE_STEP_FAILED'
            or GripperArgs.model_validate(operations[0]['arguments'])!=GripperArgs.model_validate(entry.arguments)
            or len(failed)!=1 or failed[0]['reason']!='recovery target not reached: arx.set_gripper'
            or failed[0]['output']['physical_arrival_verified'] is not True):
        raise ValueError('closing was rejected for reasons other than command convergence')
    commands=[v for k,v in records if k=='command_sent'];arrivals=[v for k,v in records if k=='arrival_observed']
    spent=calls[0]['executed_steps'];command=np.asarray(commands[-1]['target']) if commands else np.empty(0)
    if (spent!=entry.arguments['max_steps'] or len(commands)!=spent or len(arrivals)!=spent
            or not all(v['verified'] is True for v in arrivals) or command.shape!=(14,)
            or abs(command[13])>1e-3 or entry.arguments['opening']!=0.
            or report['total_physical_steps']!=checkpoint['steps']+spent
            or 64-snapshot['budget_remaining']['decisions']!=checkpoint['decisions']+1):
        raise ValueError('closing command or measured arrivals fail completion review')
    checkpoint['histories'].append((gateway,records));checkpoint['steps']+=spent;checkpoint['decisions']+=1
    checkpoint['closing_steps_consumed']+=spent;checkpoint['closing_steps_remaining']=0
    checkpoint['recovery']['remaining_steps']-=spent;checkpoint['recovery']['remaining_decisions']-=1
    checkpoint['cursor']+=1;checkpoint['closing_complete']=True;checkpoint['completed_closing_sha256']=expected_sha256
    checkpoint['expected_feedback']=snapshot['observation']['hardware']['measured_state']
    checkpoint['committed_command']=command


def extend_read_only_lift_review(checkpoint, segment, expected_sha256):
    """Charge an expired review segment with zero writes; issue no old token."""
    segment=Path(segment);gateway=segment/'private/gateway';journal=gateway/'journal.sqlite3'
    if file_sha256(journal)!=expected_sha256:raise ValueError('lift review segment SHA differs')
    report=json.loads((segment/'result.json').read_text())
    if (report['source_journal_sha256']!=checkpoint['journal_sha256'] or report['checkpoint_steps']!=checkpoint['steps']
            or report['closing_segment_sha256']!=checkpoint['closing_segment_sha256']):
        raise ValueError('lift review segment provenance differs')
    with sqlite3.connect(f'file:{journal}?mode=ro',uri=True) as db:
        records=[(k,json.loads(v)) for k,v in db.execute('select kind,payload from records order by sequence')]
        snapshot=json.loads(db.execute('select payload from snapshot').fetchone()[0])
    calls=[v for k,v in records if k=='tool_result'];rejected=[v for k,v in records if k=='attempt_rejected']
    if (any(k in ('command_sent','command_dispatch_started') for k,v in records)
            or len(calls)!=1 or calls[0]['tool']!='arx.review_grasp' or calls[0]['status']!='completed'
            or calls[0]['result']['phase']!='lift' or not calls[0]['result']['eligible']
            or len(rejected)!=1 or rejected[0]['tool']!='arx.execute_grasp'
            or rejected[0]['executed_steps']!=0 or rejected[0]['write_certainty']!='none'
            or rejected[0]['error']['code']!='VALIDATION_ERROR'
            or report['total_physical_steps']!=checkpoint['steps']
            or 64-snapshot['budget_remaining']['decisions']!=checkpoint['decisions']+1):
        raise ValueError('lift review segment has writes, faults or inconsistent budgets')
    checkpoint['histories'].append((gateway,records));checkpoint['decisions']+=1
    checkpoint['recovery']['remaining_decisions']-=1
    checkpoint['lift_review_segment_sha256']=expected_sha256
    if checkpoint['recovery']['remaining_decisions']<2:
        raise ValueError('remaining recovery budget cannot admit fresh review plus execution')
