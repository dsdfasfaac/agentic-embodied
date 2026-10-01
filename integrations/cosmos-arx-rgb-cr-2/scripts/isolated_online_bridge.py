"""Transport-only bridge: fresh Agent capsule, whitelisted RGB outputs, no policy.

The parent launches this process and independently spawns a no-history Agent.
This is an input boundary, not a sandbox for the Agent's other tools.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[1]
CAMERAS=('front_rgb','left_rgb','right_rgb')
TOOLS={'hold','set_gripper','move_eef','geometry','review_pregrasp','resume_vla','finish'}
EVIDENCE_KEYS=('rgb_only','critic_cameras','phase','attempt_frame','finger_visual_match',
    'finger_inward_pixels','peak_wrist_area_ratio','wrist_centroid_shift',
    'departure_persistence','target_on_fixture','target_stable','limitation')
VIEW_KEYS=('pink_pixels','pink_area_ratio','target_visible','target_xy','target_rack_delta',
    'baseline_delta_error','dark_dynamic_near_target','global_rgb_change')


def public_proposal(row):
    p=row.get('proposal',{});e=p.get('evidence',{})
    evidence={k:e[k] for k in EVIDENCE_KEYS if k in e}
    evidence['views']={n:{k:v[k] for k in VIEW_KEYS if k in v}
        for n,v in e.get('views',{}).items() if n in CAMERAS}
    return {'frame':row.get('frame'),'proposal':{
        k:p[k] for k in ('proposal_id','kind','candidate_actions','expected_success_signal',
                         'escalation_condition','nominal_vla_reentry_state') if k in p},
        'evidence':evidence}


def validate_decision(d,state,ids,seen_frames,count):
    if state['pending'] or state['closed']:raise ValueError('worker pending or closed')
    if not isinstance(d,dict) or d.get('tool') not in TOOLS:raise ValueError('unsupported tool')
    if d.get('evidence_frame')!=state['observation']['frame']:raise ValueError('stale evidence frame')
    if not isinstance(d.get('decision_id'),str) or not d['decision_id'] or d['decision_id'] in ids:
        raise ValueError('unique decision_id required')
    if not isinstance(d.get('reason'),str) or not d['reason'].strip():raise ValueError('reason required')
    if not isinstance(d.get('args',{}),dict):raise ValueError('args must be an object')
    if count>=100 and d['tool']!='finish':raise ValueError('decision budget exhausted; finish required')
    frames=d.get('args',{}).get('observation_frames')
    if frames is not None:
        if not isinstance(frames,list) or not frames or any(type(f) is not int for f in frames):
            raise ValueError('observation_frames must be a nonempty integer list')
        if frames!=sorted(set(frames)) or not set(frames)<=seen_frames or frames[-1]!=d['evidence_frame']:
            raise ValueError('only own ordered snapshots ending at current frame are allowed')


class Bridge:
    def __init__(self,a):
        self.a=a;self.lock=threading.RLock();self.ids=set();self.seen=set();self.count=0
        self.state={'pending':True,'closed':False,'observation':None,'tool_result':None,
                    'budget':{'total_simulation_steps':1400,'max_online_decisions':100}}
        self.token=secrets.token_urlsafe(32)
        self.private=a.private.resolve();self.private.mkdir(parents=True,exist_ok=False)
        self.capsule=a.capsule.resolve()
        shutil.copytree(ROOT/'online-skill/cosmos-arx-online-recovery',self.capsule,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (self.capsule/'connection.json').write_text(json.dumps(
            {'url':f'http://127.0.0.1:{a.port}','token':self.token}))
        hashes={str(p.relative_to(self.capsule)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.capsule.rglob('*') if p.is_file() and p.name!='connection.json'}
        (self.private/'frozen_input_manifest.json').write_text(json.dumps(
            {'skill_sha256':hashes,'context':'no parent turn history; new Agent required',
             'allowed':'skill + current RGB/proposals + own within-episode actions and feedback',
             'not_os_sandbox':True,'seed_held_out':False,'recorded_recovery_replay':False},indent=2))
        self.err=(self.private/'worker_stderr.log').open('w')
        command=['python3',str(Path(a.remote_root)/'cr.py'),'live','--seed',str(a.seed),
            '--output',a.remote_run,'--port',str(a.cosmos_port),'--max-steps','1400']
        self.worker=subprocess.Popen(['ssh','-o','BatchMode=yes',a.host,shlex.join(command)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1)
        threading.Thread(target=self.read_worker,daemon=True).start()

    def log(self,name,row):
        with (self.private/name).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

    def snapshot(self):
        with self.lock:return copy.deepcopy(self.state)

    def read_worker(self):
        try:
            for line in self.worker.stdout:
                try:event=json.loads(line)
                except json.JSONDecodeError:continue
                self.log('backend_events.jsonl',event)
                if 'rgb' in event:
                    frame=int(event['frame'])
                    folder=self.capsule/'observations'/f'frame_{frame:04d}'
                    folder.mkdir(parents=True,exist_ok=True)
                    files=[str(Path(self.a.remote_run)/f'frame_{frame:04d}'/(n+'.png')) for n in CAMERAS]
                    subprocess.run(['scp','-q',*[self.a.host+':'+p for p in files],str(folder)],check=True)
                    obs={k:event[k] for k in ('frame','reason','active_tool','environment_ended') if k in event}
                    obs['rgb']={n:str(folder/(n+'.png')) for n in CAMERAS}
                    obs['latest_proposals']=[public_proposal(r) for r in event.get('latest_proposals',[])]
                    with self.lock:
                        self.seen.add(frame);self.state.update(observation=obs,pending=False)
                        self.state['budget'].update(online_decisions=self.count,
                            simulation_steps_remaining=max(0,1400-frame))
                        self.log('agent_exposure.jsonl',self.state)
                elif 'pregrasp_proposal' in event:
                    with self.lock:
                        self.state.update(tool_result=event,pending=False)
                        self.log('agent_exposure.jsonl',self.state)
                # 'saved' can precede CLOSED.json; wait for actual process exit.
            code=self.worker.wait()
            with self.lock:
                self.state.update(pending=False,closed=True,exit_code=code)
                self.log('agent_exposure.jsonl',self.state)
        except Exception as exc:
            with self.lock:self.state.update(pending=False,transport_error=str(exc))

    def geometry(self,d):
        command=['python3',str(Path(self.a.remote_root)/'cr.py'),'geometry',self.a.remote_run,
                 '--frame',str(d['evidence_frame'])]
        frames=d.get('args',{}).get('observation_frames')
        if frames:command+=['--temporal-frames',*map(str,frames)]
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes',self.a.host,shlex.join(command)],
                capture_output=True,text=True,timeout=60,check=True)
            value=json.loads(result.stdout)
        except Exception as exc:value={'status':'unknown','reason':str(exc),'proposal_only':True}
        with self.lock:
            self.state.update(tool_result={'geometry':value},pending=False)
            self.log('agent_exposure.jsonl',self.state)

    def submit(self,d):
        with self.lock:
            validate_decision(d,self.state,self.ids,self.seen,self.count)
            self.ids.add(d['decision_id']);self.count+=1
            self.log('agent_decisions.jsonl',d)
            self.state.update(pending=True,tool_result=None)
            self.state['budget']['online_decisions']=self.count
            if d['tool']=='geometry':threading.Thread(target=self.geometry,args=(d,),daemon=True).start()
            else:
                self.worker.stdin.write(json.dumps(d,ensure_ascii=False)+'\n');self.worker.stdin.flush()
            return {'accepted':True,'pending':True}


def handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            data=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def authorized(self):
            return secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+bridge.token)
        def do_GET(self):
            if not self.authorized():return self.reply(403,{'error':'unauthorized'})
            if self.path!='/state':return self.reply(404,{'error':'unknown endpoint'})
            self.reply(200,bridge.snapshot())
        def do_POST(self):
            if not self.authorized():return self.reply(403,{'error':'unauthorized'})
            if self.path!='/action':return self.reply(404,{'error':'unknown endpoint'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=32768:raise ValueError('invalid request length')
                value=json.loads(self.rfile.read(length));result=bridge.submit(value)
            except Exception as exc:return self.reply(400,{'error':str(exc)})
            self.reply(200,result)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('host','remote-root','remote-run'):p.add_argument('--'+name,required=True)
    for name in ('capsule','private'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--port',type=int,default=18791)
    p.add_argument('--cosmos-port',type=int,default=15591)
    a=p.parse_args();b=Bridge(a)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(b))
    print(json.dumps({'capsule':str(b.capsule),'client':str(b.capsule/'scripts/client.py')}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if b.worker.poll() is None:
            b.worker.stdin.close();b.worker.wait(timeout=180)
        server.server_close();b.err.close()


if __name__=='__main__':main()
