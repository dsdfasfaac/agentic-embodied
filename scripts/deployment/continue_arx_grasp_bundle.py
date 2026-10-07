#!/usr/bin/env python3
"""Continue an audited interrupted closing call through the frozen bundle suffix.

Preserve the original episode budgets. Source files are never changed. Live
state, target identity and depth must pass before the first new command.
Controllers stay enabled on termination, for supervised unloading/home.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.deployment.serve_arx_real_gateway import RealCoreFactory
from robots.arx.deployment.grasp_continuation import read_checkpoint, restore_sensor_history, adopt_checkpoint, extend_closing_checkpoint
from robots.arx.deployment.bundle_program import resolve_call, compile_programs
from robots.arx.gateway.contracts import ToolRequest
from zetta.evolution.jsonio import file_sha256


def run(args):
    root=Path(__file__).resolve().parents[2]; provider=root/'robots/arx/deployment/picktube_rgbd_provider.py'
    args.output.mkdir(exist_ok=False,parents=True); gateway=args.output/'private/gateway';gateway.mkdir(parents=True)
    limits=json.loads(args.runtime_config.read_text()); core=None; calls=[]
    report=dict(schema_version='arx.grasp.bundle-continuation.v1',status='initializing',task_success=False,
        controller_disabled=False,source_trial=str(args.source_trial),source_journal_sha256=args.source_journal_sha256)
    factory=RealCoreFactory(hardware_config=str(args.hardware_config),hardware_sha256=args.hardware_sha256,
        task=str(root/'robots/arx/manifests/pickup_test_tube.yaml'),model_contract=str(root/'robots/arx/manifests/task7_model_a.yaml'),
        output=str(gateway),episode_id='continuation-'+uuid.uuid4().hex,limits=limits,zeva_host='127.0.0.1',zeva_port=5583,
        kinematics_calibration=str(root/'robots/arx/manifests/real/dodo_right_controller_ee_fk.json'),
        bundle=str(args.bundle),tool_catalog=str(args.frozen/'tool-catalog.json'),real_input_contract=str(args.frozen/'real-input-contract.json'),
        expected_real_input_sha256=file_sha256(args.frozen/'real-input-contract.json'),feature_provider=str(provider),
        expected_feature_provider_sha256=file_sha256(provider),grasp_config=str(args.grasp_config),expected_grasp_config_sha256=file_sha256(args.grasp_config))
    try:
        core=factory(lambda:False,lambda phase:None)
        if len(core.programs)!=1: raise ValueError('continuation requires one frozen program')
        program=next(iter(core.programs.values()))
        source_package=core.package_sha256
        if args.source_bundle:
            from robots.arx.deployment.real_input import _load_bundle
            source,_=_load_bundle(args.source_bundle);current,_=_load_bundle(args.bundle)
            previous=compile_programs(source,max_physical_steps=limits['max_steps'])
            if (current.parent_sha256!=source.sha256 or previous!=core.programs):
                raise ValueError('critic revision must retain exactly the source recovery program')
            source_package=source.sha256
        checkpoint=read_checkpoint(args.source_trial,args.source_journal_sha256,program,source_package)
        if args.closing_segment:
            if not args.closing_segment_sha256:raise ValueError('closing segment requires SHA')
            extend_closing_checkpoint(checkpoint,args.closing_segment,args.closing_segment_sha256)
        if limits['max_steps']!=600 or limits['max_decisions']!=64:
            raise ValueError('source episode budgets must remain 600 steps and 64 decisions')
        grasp=core.registry.resolve('arx.execute_grasp').handler
        restore_sensor_history(checkpoint,grasp); adopt_checkpoint(core,checkpoint)
        report.update(status='admitted',checkpoint_steps=checkpoint['steps'],remaining_recovery_steps=core.recovery['remaining_steps'],
                      package_sha256=core.package_sha256,source_package_sha256=source_package,
                      closing_segment=None if not args.closing_segment else str(args.closing_segment),
                      closing_segment_sha256=args.closing_segment_sha256)
        if args.check_only:
            report.update(status='read_only_verified',termination_reason='check_only',robot_commands_sent=False)
        while not args.check_only and core.recovery is not None and not core.closed:
            program=core.programs[core.recovery['binding_id']]
            if core.program_cursor>=len(program.calls): raise ValueError('bundle suffix exhausted without terminal success')
            entry=program.calls[core.program_cursor]
            request=ToolRequest(request_id='continuation-'+uuid.uuid4().hex,decision_ref='decision-'+uuid.uuid4().hex,
                observation_id=core.current['observation_id'],control_epoch=core.epoch,tool=entry.tool,
                arguments=resolve_call(entry,core.current['observation_id'],core.program_token,core.program_outputs),
                evidence_ids=[core.current['observation_id']],reason='Continue audited bundle suffix with consumed budgets')
            core.journal.register_decision(request,source='runner',evidence=request.evidence_ids)
            result=core.execute(request);calls.append(result)
            (args.output/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
            print(json.dumps(dict(tool=entry.tool,status=result['status'],steps=result['executed_steps'],result=result['result'],error=result['error'])),flush=True)
            if result.get('result',{}).get('completion')=='task_success':
                report.update(status='completed',task_success=True,termination_reason='task_success');break
            if result['status']!='completed':
                report.update(status='paused_keep_enabled',termination_reason=result.get('result',{}).get('completion','gateway_rejected'));break
        if not report['task_success'] and report['status']=='admitted':
            report.update(status='completed_without_success',termination_reason='bundle_suffix_completed')
    except Exception as exc:
        report.update(status='failed_keep_enabled',termination_reason='continuation_rejected',error=str(exc))
        raise
    finally:
        if core is not None:
            if core.current is not None:
                report.update(total_physical_steps=core.step_index,new_physical_steps=core.step_index-report.get('checkpoint_steps',core.step_index),
                    final_features=getattr(core.critic,'last_feature_evidence',None),final_observation=core.current)
                if not core.closed: core.close()
            else: core.backend.close()
            core.journal.close()
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-trial','bundle','grasp-config','frozen','output','hardware-config','runtime-config'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-journal-sha256',required=True);p.add_argument('--hardware-sha256',required=True)
    p.add_argument('--source-bundle',type=Path);p.add_argument('--closing-segment',type=Path);p.add_argument('--closing-segment-sha256')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true');mode.add_argument('--check-only',action='store_true');a=p.parse_args()
    report=run(a);print(json.dumps({k:report[k] for k in ('status','task_success','termination_reason')}))
if __name__=='__main__':main()
