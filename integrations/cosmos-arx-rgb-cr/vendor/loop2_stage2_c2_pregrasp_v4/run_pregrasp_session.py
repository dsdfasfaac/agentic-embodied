"""Agent EEF restaging, evidence-gated pregrasp handoff, then fresh Cosmos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import dataclasses
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'loop2_stage2_c2_phase_v3'))
from run_closed_loop import ClosedLoopSession
import run_agent_session as base
from pregrasp_gate import PregraspGate
from rgb_geometry import estimate, estimate_temporal


class PregraspSession(ClosedLoopSession):
    def __init__(self,args):
        super().__init__(args)
        self.env.task=dataclasses.replace(self.env.task,max_steps=args.max_steps)
        self.gate=PregraspGate(self.rgb())
        self.needs_pregrasp=False;self.restaged=False
        for source in (Path(__file__),Path(__file__).with_name('pregrasp_gate.py')):
            shutil.copy2(source,self.output/'executed_source'/source.name)
        (self.output/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (self.output/'executed_source').glob('*.py')},indent=2))

    def step(self,target,source):
        result=super().step(target,source)
        if result and result.kind=='attempted_grasp_target_left_behind':
            self.needs_pregrasp=True;self.restaged=False
        _,rotation,_=self.kin.fk(self.command[7:13])
        self.gate.observe(self.rgb(),self.frames,self.command,rotation,source)
        self.log('pregrasp_features.jsonl',self.gate.history[-1])
        return result

    def bootstrap(self):
        if self.args.mode == 'vla-first':
            self.snapshot('fresh initial scene; awaiting external Agent decision for first VLA request')
            return
        if not self.args.approved_replay:return super().bootstrap()
        if not self.args.replay_approval:raise ValueError('Agent replay approval required')
        folder=self.args.approved_replay
        # Development reproduction of explicitly Agent-approved prior actions.
        # Only own actions and RGB images are read, NEVER audit or simulator state.
        data=np.load(folder/'trajectory.npz');targets=data['raw_targets'];sources=data['command_sources']
        if len(targets)>=self.args.max_steps:raise ValueError('replay exceeds trial budget')
        frames={json.loads(line)['frame'] for line in (folder/'agent_observations.jsonl').read_text().splitlines()}
        reviewed=[]
        for target,source in zip(targets,sources):
            self.active=str(source)
            self.step(target,str(source))
            if self.frames in frames:
                for camera in base.CAMERAS:
                    expected=base.imageio.imread(folder/f'frame_{self.frames:04d}'/f'{camera}.png')
                    if not np.array_equal(expected,self.rgb()[camera]):
                        raise ValueError(f'RGB reproduction diverged at {self.frames}; stop before continuation')
                self.snapshot('verified exact RGB reproduction of previously Agent-reviewed state')
                reviewed.append(self.frames)
            if self.finished:raise ValueError('episode ended during approved reproduction')
        self.recovery_active=True;self.recovery_ever=True;self.needs_pregrasp=True;self.restaged=True
        self.critic.begin_recovery();self.active='hold'
        self.log('approved_replay.jsonl',{'source':str(folder),'frames':len(targets),
            'approval':self.args.replay_approval,'exact_rgb_review_frames':reviewed,
            'source_trajectory_sha256':hashlib.sha256((folder/'trajectory.npz').read_bytes()).hexdigest(),
            'interpretation':'development state reproduction, not an independent recovery trial'})
        shutil.copy2(folder/'agent_decisions.jsonl',self.output/'replayed_agent_decisions.jsonl')
        self.snapshot('approved RGB-verified state reproduction complete; fresh gate/Agent decision required')

    def execute(self,request):
        if request['evidence_frame']!=self.frames:raise ValueError('stale RGB frame')
        if request['decision_id'] in self.ids:raise ValueError('duplicate decision id')
        if not request.get('reason'):raise ValueError('Agent visual reasoning required')
        tool=request['tool'];params=request.get('args',{})
        if tool=='review_pregrasp':
            if self.finished:raise ValueError('episode ended')
            self.ids.add(request['decision_id'])
            self.log('agent_decisions.jsonl',dict(request,decision_owner='external_multimodal_agent'))
            # Snapshot exists from last primitive. Geometry is recomputed by the
            # trusted executor, never accepted as an Agent-supplied pass flag.
            mode=params.get('measurement_mode','simultaneous')
            try:
                if mode=='simultaneous':geometry=estimate(base.SCENE/'scene.xml',self.output,self.frames)
                elif mode=='temporal_rgb':
                    frames=params['observation_frames']
                    if not isinstance(frames,list) or len(frames)<5 or frames!=sorted(set(frames)) or \
                        frames[-1]!=self.frames or frames[0]<self.frames-150:
                        raise ValueError('need >=5 ordered unique RGB snapshots within 150 frames, including current')
                    geometry=estimate_temporal(base.SCENE/'scene.xml',self.output,frames)
                    split_a=sorted(set(frames[::2]+[frames[-1]]))
                    split_b=sorted(set(frames[1::2]+[frames[-1]]))
                    a=estimate_temporal(base.SCENE/'scene.xml',self.output,split_a)
                    b=estimate_temporal(base.SCENE/'scene.xml',self.output,split_b)
                    disagreement=float(np.linalg.norm(np.asarray(a['rgb_band_to_commanded_tcp_tool_m'])-
                        b['rgb_band_to_commanded_tcp_tool_m']))
                    geometry.update(measurement_mode=mode,split_fit_disagreement_m=disagreement,
                                    fresh_window_verified=True)
                else:raise ValueError('unknown measurement mode; no automatic fallback')
            except (ValueError,KeyError) as e:geometry={'status':'unknown','reason':str(e)}
            proposal=self.gate.review(self.frames,geometry)
            self.log('pregrasp_proposals.jsonl',proposal)
            print(json.dumps({'pregrasp_proposal':proposal,'frame':self.frames}),flush=True)
            return True
        guarded=tool=='resume_vla' and (self.recovery_active or self.needs_pregrasp)
        if guarded:
            if not self.restaged:raise ValueError('resume blocked: Agent EEF restaging required after failed acquisition')
            proposal=self.gate.require(self.frames,params.get('pregrasp_proposal_id'),params.get('visual_checks'))
            if not params.get('reentry_reason'):raise ValueError('Agent reentry reason required')
            self.log('validated_reentries.jsonl',{'frame':self.frames,'proposal':proposal,
                'approved_by':request['decision_id'],'visual_checks':params['visual_checks']})
        before=self.frames
        result=super().execute(request)
        if tool=='move_eef' and self.frames>before:self.restaged=True
        if guarded and self.frames>before:
            # A new failure during the nominal chunk must remain latched.
            self.needs_pregrasp=bool(self.proposals and self.proposals[-1]['frame']>before and
                self.proposals[-1]['proposal']['kind']=='attempted_grasp_target_left_behind')
        return result

    def close(self):
        super().close()
        path=self.output/'audit.json';a=json.loads(path.read_text())
        a.update({'pregrasp_gate_version':'4.1','max_steps':self.args.max_steps,
                  'evaluation_mode':f'legacy success continues; safety failures terminate; max {self.args.max_steps} steps',
                  'stage2_pass':False})
        path.write_text(json.dumps(a,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--port',type=int,default=15591)
    p.add_argument('--prefix-limit',type=int,default=512);p.add_argument('--max-steps',type=int,default=1400)
    p.add_argument('--approved-replay',type=Path);p.add_argument('--replay-approval')
    args=p.parse_args()
    if not 600<=args.max_steps<=1600:raise ValueError('bounded evaluation budget 600..1600')
    s=PregraspSession(args)
    try:
        s.bootstrap()
        for line in sys.stdin:
            try:
                if not s.execute(json.loads(line)):break
            except Exception as exc:
                s.log('rejections.jsonl',{'frame':s.frames,'reason':str(exc)})
                s.snapshot(f'request failed: {exc}; no fallback')
    finally:s.close()


if __name__=='__main__':main()
