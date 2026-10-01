#!/usr/bin/env python3
"""Interactive Agent tool executor: one JSON decision per stdin line.

The real external Agent sees RGB snapshots and proposals, then chooses a tool.
There is no automatic recovery choice, reference regrasp, or reverse-path tool.
Physics is paused while waiting for an Agent decision.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import shutil

import imageio.v2 as imageio
import numpy as np

from early_critic import EarlyRgbCritic
from eef_tools import CommandKinematics, set_gripper
from robots.arx.environment import ArxMujocoEnv
from robots.arx.control import ActionProcessor, prepare_model_state
from robots.arx.contracts import load_model_contract
from robots.arx.cosmos_edge_client import CosmosEdgeClient

CAMERAS=('front_rgb','left_rgb','right_rgb')
ROOT=Path('/data4/zhengyikai/Agentic-Embodied')
SCENE=ROOT/'runs/arx_pickup_test_tube/pickup_test_tube_initial_scene'


class Session:
    critic_type=EarlyRgbCritic

    def __init__(self,args):
        self.args=args
        self.output=args.output
        self.output.mkdir(parents=True,exist_ok=False)
        source_dir=self.output/'executed_source'; source_dir.mkdir()
        hashes={}
        for name in ('eef_tools.py','early_critic.py','run_agent_session.py'):
            source=Path(__file__).with_name(name)
            shutil.copy2(source,source_dir/name)
            hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        (self.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
        self.env=ArxMujocoEnv(prepared_scene_bundle=str(SCENE),
            mapping_path=str(SCENE/'mapping.json'),task_manifest=str(SCENE/'task.yaml'),
            camera_names={n:n for n in CAMERAS})
        self.kin=CommandKinematics.from_scene_xml(SCENE/'scene.xml')
        (self.output/'robot_calibration.json').write_text(json.dumps(self.kin.calibration,indent=2))
        self.obs,self.info=self.env.reset(seed=args.seed)
        self.initial_hash=hashlib.sha256(np.ascontiguousarray(self.obs['state']).tobytes()).hexdigest()
        # Initial command is static task configuration; later commands are
        # predicted with the same pure filter used by the hardware adapter.
        self.command=np.array(self.env.task.start_state,dtype=np.float32)
        self.processor=ActionProcessor(self.env.task,self.command)
        # Subclasses select their critic before any reset-image validation.
        self.critic=self.critic_type(self.rgb())
        self.frames=0; self.finished=False; self.ids=set(); self.active='original_prefix'
        self.audit_rows=[]; self.raw=[]; self.sources=[]; self.proposals=[]
        opts=dict(fps=15,codec='libx264',pixelformat='yuv420p',macro_block_size=1)
        self.writers={n:imageio.get_writer(self.output/f'{n}.mp4',**opts) for n in (*CAMERAS,'three_view')}
        self.record_frame()

    def rgb(self):
        return {n:self.obs[n] for n in CAMERAS}

    def log(self,name,row):
        with (self.output/name).open('a') as f:
            f.write(json.dumps(row,default=str)+'\n')

    def record_frame(self):
        for n in CAMERAS: self.writers[n].append_data(self.obs[n])
        self.writers['three_view'].append_data(np.concatenate([self.obs[n] for n in CAMERAS],axis=1))
        # Private evaluator recorder: never serialized in live Agent responses.
        self.audit_rows.append({'frame':self.frames,'measured_state':self.obs['state'].tolist(),
                                'command':self.command.tolist(),'evaluation':self.info['evaluation']})

    def snapshot(self,reason):
        folder=self.output/f'frame_{self.frames:04d}'; folder.mkdir(exist_ok=True)
        paths={}
        for n in CAMERAS:
            path=folder/f'{n}.png'; imageio.imwrite(path,self.obs[n]); paths[n]=str(path)
        imageio.imwrite(folder/'three_view.png',np.concatenate([self.obs[n] for n in CAMERAS],axis=1))
        reply={'frame':self.frames,'reason':reason,'rgb':paths,'active_tool':self.active,
               'environment_ended':self.finished,'latest_proposals':self.proposals[-2:],
               'observation_boundary':'RGB only; no measured joints/contact/object pose/evaluator in Agent response'}
        self.log('agent_observations.jsonl',reply)
        print(json.dumps(reply),flush=True)

    def step(self,target,source):
        self.command,_=self.processor.process(target)
        self.obs,_,term,trunc,self.info=self.env.step(target)
        self.finished=bool(term or trunc); self.frames+=1
        self.raw.append(np.array(target)); self.sources.append(source)
        self.record_frame()
        proposal=self.critic.observe(self.rgb())
        self.log('rgb_features.jsonl',{'frame':self.frames,
            'views':self.critic.confirmed._json_features(self.critic.confirmed._features(self.rgb()))})
        self.log('actions.jsonl',{'frame':self.frames,'tool':source,'raw_target':np.asarray(target).tolist(),
                                 'command_prediction':self.command.tolist()})
        if proposal:
            row={'frame':self.frames,'proposal':dataclasses.asdict(proposal)}
            self.proposals.append(row); self.log('critic_proposals.jsonl',row)
        return proposal

    def bootstrap(self):
        with np.load(self.args.prefix) as data:
            targets=data['vla_action_targets']
            for target in targets[:160]:
                if self.step(target,'original_prefix') or self.finished:
                    break
        self.snapshot('original prefix replay stopped for Agent review; subsequent recovery is live')

    def execute(self,request):
        identifier=request['decision_id']; tool=request['tool']; params=request.get('args',{})
        if identifier in self.ids: raise ValueError('duplicate decision id')
        if request['evidence_frame'] != self.frames: raise ValueError('stale RGB frame')
        if not request.get('reason'): raise ValueError('Agent visual reasoning required')
        self.ids.add(identifier)
        self.log('agent_decisions.jsonl',dict(request,decision_owner='external_multimodal_agent'))
        if tool=='finish': return False
        if self.finished: raise ValueError('environment has ended; only finish allowed')
        if tool not in ('move_eef','set_gripper','hold','resume_vla'):
            raise ValueError('unknown tool')
        # Validate the complete primitive before changing the active controller.
        if tool=='move_eef':
            targets=self.kin.move_eef(self.command,**params)
            # Settle the command filter only; visual arrival still needs review.
            targets += [targets[-1].copy() for _ in range(8)]
        elif tool=='set_gripper':
            target=set_gripper(self.command,**params)
            count=min(55,int(np.ceil(abs(target[13]-self.command[13])/.08))+12)
            targets=[target.copy() for _ in range(count)]
        elif tool=='hold':
            count=params.get('frames',3)
            if not isinstance(count,int) or not 1<=count<=15: raise ValueError('hold 1..15 frames')
            targets=[self.command.copy() for _ in range(count)]
        else:
            contract=load_model_contract(ROOT/'robots/arx/manifests/task7_model_a.yaml')
            with CosmosEdgeClient('127.0.0.1',self.args.port,contract,timeout_sec=60) as client:
                pred=client.predict(self.rgb(),prepare_model_state(self.obs['state'],self.env.task),
                    self.env.task.instruction,seed=int(params.get('seed',self.args.seed)))
            targets=pred.actions[:self.env.task.execution_steps]
        self.log('tool_switches.jsonl',{'frame':self.frames,'from':self.active,'to':tool,'approved_by':identifier})
        self.active=tool
        for target in targets:
            if self.step(target,tool) or self.finished:
                self.snapshot('new critic proposal or environment end; Agent decision required')
                return True
        self.snapshot('bounded primitive ended; review RGB before next action')
        return True

    def close(self):
        for w in self.writers.values(): w.close()
        np.savez_compressed(self.output/'trajectory.npz',raw_targets=np.array(self.raw),
                            command_sources=np.array(self.sources))
        prefix_count=self.sources.count('original_prefix')
        with np.load(self.args.prefix) as reference:
            command_error=float(np.max(np.abs(np.array([r['command'] for r in self.audit_rows[1:prefix_count+1]])
                    -reference['processed_action_targets'][:prefix_count]))) if prefix_count else 0.
            tracked_error=float(np.max(np.abs(np.array([r['measured_state'] for r in self.audit_rows[:prefix_count+1]])
                    -reference['robot_states'][:prefix_count+1])))
        result={'seed':self.args.seed,'initial_state_sha256':self.initial_hash,'frames':self.frames,
                'prefix_frames':prefix_count,'prefix_command_max_abs_error':command_error,
                'prefix_measured_state_max_abs_error_posthoc_only':tracked_error,
                'final_evaluation_posthoc_only':self.info['evaluation'],
                'agent_decisions':len(self.ids),'stage2_pass':False,
                'scope':'primitive experiment; no automatic Stage2 pass claim',
                'runtime_recovery_inputs':['RGB','static robot geometry','own command history'],
                'measured_joints_available_to_recovery':False,
                'critic_contact_or_object_pose_used':False}
        (self.output/'audit.json').write_text(json.dumps(result,indent=2))
        (self.output/'posthoc_trace.json').write_text(json.dumps(self.audit_rows))
        self.env.close()
        print(json.dumps({'saved':str(self.output)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--seed',type=int,default=941132); p.add_argument('--port',type=int,default=15581)
    s=Session(p.parse_args())
    try:
        s.bootstrap()
        for line in sys.stdin:
            try:
                if not s.execute(json.loads(line)): break
            except Exception as exc:
                s.log('rejections.jsonl',{'frame':s.frames,'reason':str(exc)})
                print(json.dumps({'rejected':str(exc),'frame':s.frames,'no_automatic_fallback':True}),flush=True)
    finally: s.close()


if __name__=='__main__': main()
