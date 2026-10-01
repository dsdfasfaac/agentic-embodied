"""RGB critic -> external Agent recovery -> fresh receding-horizon Cosmos.

No recovery action is selected automatically. Agent must explicitly approve a
current-frame reentry_reason after recovery. No buffered pre-recovery action can
be resumed. Requests contain nominal proprioception as in the original policy;
critic and Agent recovery observations still contain RGB only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'loop2_stage2_c2_eef_v2'))
from run_full_pickup_eval import FullSession
import run_agent_session as base
from phase_critic import PhaseAwareRgbCritic


class ClosedLoopSession(FullSession):
    def __init__(self,args):
        super().__init__(args)
        self.critic=PhaseAwareRgbCritic(self.rgb())
        self.recovery_active=False;self.recovery_ever=False
        self.fresh_requests=0;self.handoffs=[]
        for source in (Path(__file__),Path(__file__).with_name('phase_critic.py')):
            shutil.copy2(source,self.output/'executed_source'/source.name)
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (self.output/'executed_source').glob('*.py')}
        (self.output/'source_hashes.json').write_text(json.dumps(manifest,indent=2))

    def step(self,target,source):
        proposal=super().step(target,source)
        self.log('phase_trace.jsonl',dict(frame=self.frames,**self.critic.last_features))
        return proposal

    def execute(self,request):
        tool=request['tool'];params=request.get('args',{})
        # Reject stale/duplicate requests before any protocol state mutation.
        if request['evidence_frame']!=self.frames:raise ValueError('stale RGB frame')
        if request['decision_id'] in self.ids:raise ValueError('duplicate decision id')
        if not request.get('reason'):raise ValueError('Agent visual reasoning required')
        if tool!='resume_vla':
            if tool in ('move_eef','set_gripper') and not self.recovery_active:
                # Validate before clearing the critic's phase latch.
                if self.finished:raise ValueError('environment ended')
                if tool=='move_eef':self.kin.move_eef(self.command,**params)
                else:base.set_gripper(self.command,**params)
                self.critic.begin_recovery()
                self.recovery_active=True;self.recovery_ever=True
                self.log('protocol.jsonl',{'frame':self.frames,'event':'agent_started_recovery',
                    'approved_by':request['decision_id']})
            return super().execute(request)
        if self.finished:raise ValueError('environment ended')
        count=params.get('max_chunks',1)
        if not isinstance(count,int) or not 1<=count<=32:raise ValueError('max_chunks 1..32')
        if self.recovery_active and not params.get('reentry_reason'):
            raise ValueError('Agent current-RGB reentry_reason required after recovery')
        self.ids.add(request['decision_id'])
        self.log('agent_decisions.jsonl',dict(request,decision_owner='external_multimodal_agent'))
        handoff=self.recovery_active
        if handoff:
            row={'frame':self.frames,'approved_by':request['decision_id'],
                 'rgb_reentry_reason':params['reentry_reason'],'old_chunk_discarded':True,
                 'first_fresh_request_index':self.fresh_requests}
            self.handoffs.append(row);self.log('handoffs.jsonl',row)
            self.recovery_active=False
        # New visual attempt must establish itself; never carry stale alert into
        # resumed policy. This resets protocol memory, not the simulation.
        if handoff or self.critic.sent:
            self.critic.begin_recovery()
        source='cosmos_after_recovery' if self.recovery_ever else 'cosmos_before_recovery'
        self.log('tool_switches.jsonl',{'frame':self.frames,'from':self.active,'to':source,
                                      'approved_by':request['decision_id']})
        self.active=source
        contract=base.load_model_contract(base.ROOT/'robots/arx/manifests/task7_model_a.yaml')
        with base.CosmosEdgeClient('127.0.0.1',self.args.port,contract,timeout_sec=120) as client:
            for _ in range(count):
                request_frame=self.frames
                seed=int(params.get('seed',self.args.seed))+self.fresh_requests
                t=time.monotonic()
                pred=client.predict(self.rgb(),base.prepare_model_state(self.obs['state'],self.env.task),
                                    self.env.task.instruction,seed=seed)
                self.log('cosmos_requests.jsonl',{'index':self.fresh_requests,'frame':request_frame,
                    'seed':seed,'source':source,'latency_s':time.monotonic()-t,
                    'returned_actions':len(pred.actions),'executed_chunk_limit':self.env.task.execution_steps,
                    'fresh_current_observation':True,'approved_by':request['decision_id']})
                self.fresh_requests+=1
                for target in pred.actions[:self.env.task.execution_steps]:
                    if self.step(target,source) or self.finished:
                        self.snapshot('critic proposal or environment end; remaining chunk discarded')
                        return True
        self.snapshot('nominal review interval reached; fresh policy may continue after Agent RGB review')
        return True

    def close(self):
        super().close()
        path=self.output/'audit.json';report=json.loads(path.read_text())
        post=sum(s=='cosmos_after_recovery' for s in self.sources)
        report.update({'fresh_cosmos_requests':self.fresh_requests,'agent_approved_handoffs':self.handoffs,
            'post_recovery_cosmos_steps':post,'vla_handoff_executed':bool(self.handoffs and post),
            'stage2_pass':False,'scope':'handoff execution is NOT downstream success; independent full-task audit required'})
        path.write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--port',type=int,default=15591)
    p.add_argument('--prefix-limit',type=int,default=512)
    s=ClosedLoopSession(p.parse_args())
    try:
        s.bootstrap()
        for line in sys.stdin:
            try:
                if not s.execute(json.loads(line)):break
            except Exception as exc:
                s.log('rejections.jsonl',{'frame':s.frames,'reason':str(exc)})
                s.snapshot(f'request failed: {exc}; no automatic fallback; inspect before retry')
    finally:s.close()


if __name__=='__main__':main()
