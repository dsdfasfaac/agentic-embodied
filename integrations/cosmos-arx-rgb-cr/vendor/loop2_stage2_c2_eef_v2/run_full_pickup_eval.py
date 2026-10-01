"""Evaluation-only continuation; frozen critic/tools, RGB-only external Agent.

Legacy success is recorded, not used to terminate. Safety failures and explicit
900-step budget still terminate. Privileged clearance is private until close.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
import run_agent_session as base


class ContinueSuccessEnv(base.ArxMujocoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task=dataclasses.replace(self.task,max_steps=900)

    def step(self, action):
        obs,reward,term,trunc,info=super().step(action)
        if info['evaluation']['success'] and not info['evaluation']['failure']:
            term=False
            trunc=self._steps>=self.task.max_steps
            self._finished=trunc
        return obs,reward,term,trunc,info


base.ArxMujocoEnv=ContinueSuccessEnv


class FullSession(base.Session):
    def __init__(self,args):
        super().__init__(args)
        for source in (Path(__file__),Path(base.__file__).with_name('rgb_geometry.py'),
                       Path(base.__file__).parent.parent/'loop2_stage2_c2_rgb_v1/rgb_c2_critic.py'):
            shutil.copy2(source,self.output/'executed_source'/source.name)
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (self.output/'executed_source').glob('*.py')}
        (self.output/'source_hashes.json').write_text(json.dumps(manifest,indent=2))

    def record_frame(self):
        super().record_frame()
        # Private audit: never returned by snapshot, critic, or tool planner.
        try:
            model,data=self.env.model,self.env.data
            target=self.env._evaluator.body_id
            # Scene bundles use both the legacy ``prop_rack`` name and the
            # generated ``yellow_test_tube_rack`` name.  Resolve by semantic
            # aliases instead of treating a missing legacy name as body -1.
            rack=-1
            for name in ('prop_rack','yellow_test_tube_rack'):
                rack=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
                if rack >= 0:
                    break
            def vertical_bounds(body):
                ids=np.flatnonzero(model.geom_bodyid==body)
                lows=[]; highs=[]
                for g in ids:
                    rotation=data.geom_xmat[g].reshape(3,3)
                    center=data.geom_xpos[g]+rotation@model.geom_aabb[g,:3]
                    extent=np.abs(rotation)@model.geom_aabb[g,3:]
                    lows.append(float(center[2]-extent[2]));highs.append(float(center[2]+extent[2]))
                if not lows:
                    raise ValueError(f'body {body} has no collision/render geometry')
                return min(lows),max(highs)
            if rack < 0:
                raise ValueError('scene has no supported rack body (expected prop_rack or yellow_test_tube_rack)')
            target_low,_=vertical_bounds(target);_,rack_high=vertical_bounds(rack)
            others=set()
            fingers=set(self.env._evaluator.finger_body_ids)
            for contact in data.contact[:data.ncon]:
                a,b=int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
                if target in (a,b):
                    other=b if a==target else a
                    if other!=target and other not in fingers: others.add(other)
            self.audit_rows[-1]['full_pickup_private']={
                'conservative_vertical_clearance_m':target_low-rack_high,
                'target_low_z':target_low,'rack_high_z':rack_high,
                'nonfinger_contact_bodies':sorted(others),
                'target_position':data.xpos[target].tolist(),
                'active_tool':self.active}
        except Exception as exc:
            # Audit data is diagnostic only; never let it stop the session.
            self.audit_rows[-1]['full_pickup_private'] = None
            self.audit_rows[-1]['full_pickup_audit_error'] = str(exc)

    def bootstrap(self):
        with np.load(self.args.prefix) as data:
            for target in data['vla_action_targets'][:self.args.prefix_limit]:
                if self.step(target,'original_prefix') or self.finished: break
        self.snapshot('prefix stopped at proposal, budget or end; no automatic recovery')

    def close(self):
        super().close()
        path=self.output/'audit.json';report=json.loads(path.read_text())
        any_legacy=False;run=0;longest=0;hold_run=0;max_hold=0
        audit_rows=[row for row in self.audit_rows if row.get('full_pickup_private') is not None]
        for row in audit_rows:
            e=row['evaluation'];p=row['full_pickup_private']
            any_legacy|=e['success']
            good=(p['conservative_vertical_clearance_m']>=.003
                  and all(e['finger_contacts']) and not p['nonfinger_contact_bodies'])
            run=run+1 if good else 0;longest=max(longest,run)
            hold_run=hold_run+1 if good and p['active_tool']=='hold' else 0
            max_hold=max(max_hold,hold_run)
        report.update({'legacy_environment_success_ever':any_legacy,
            'full_clearance_consecutive_steps':longest,'full_clearance_stationary_hold_steps':max_hold,
            'full_pickup_physical_audit_pass':max_hold>=15 and len(audit_rows)==len(self.audit_rows),
            'full_pickup_audit_frames':len(audit_rows),
            'full_pickup_audit_errors':len(self.audit_rows)-len(audit_rows),
            'visual_full_pickup_verdict':'external Agent review required',
            'vla_handoff_validated':False,'stage2_pass':False,
            'evaluation_mode':'legacy success continues; safety failures terminate; max 900 steps',
            'clearance_definition':'conservative target geom AABB minimum Z exceeds rack geom AABB maximum Z by 3 mm; both fingers; no nonfinger contact; 15 hold-tool frames',
            'prefix_limit':self.args.prefix_limit})
        path.write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--port',type=int,default=15581)
    p.add_argument('--prefix-limit',type=int,default=512)
    s=FullSession(p.parse_args())
    try:
        s.bootstrap()
        for line in sys.stdin:
            try:
                if not s.execute(json.loads(line)):break
            except Exception as exc:
                s.log('rejections.jsonl',{'frame':s.frames,'reason':str(exc)})
                print(json.dumps({'rejected':str(exc),'frame':s.frames,'no_automatic_fallback':True}),flush=True)
    finally:s.close()


if __name__=='__main__':main()
