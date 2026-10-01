"""Small durable coordinator for one-shot ARX rollout subprocesses."""
from __future__ import annotations
import json, subprocess, sys, uuid, os, socket, time
from pathlib import Path

class CampaignCoordinator:
 def __init__(self, root, contract): self.root=Path(root); self.contract=contract; self.root.mkdir(parents=True,exist_ok=True); self.zeva=None
 def _start_zeva(self, cfg):
  if self.zeva and self.zeva.poll() is None: return
  env=os.environ.copy(); env.update({k:str(v) for k,v in cfg.get('zeva_env',{}).items()})
  env.setdefault('HOST',cfg.get('vla_host','127.0.0.1')); env.setdefault('PORT',str(cfg.get('vla_port',5581)))
  log=(self.root/'zeva-server.log').open('ab')
  self.zeva=subprocess.Popen(['bash','scripts/deployment/start_zeva_arx_task7_server.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.time()+float(cfg.get('zeva_startup_timeout_s',180))
  while time.time()<deadline:
   if self.zeva.poll() is not None: raise RuntimeError('Zeva server exited during startup')
   try:
    with socket.create_connection((env['HOST'],int(env['PORT'])),timeout=1): return
   except OSError: time.sleep(1)
  self.zeva.terminate(); raise RuntimeError('Zeva server readiness timeout')
 def close(self):
  if self.zeva and self.zeva.poll() is None: self.zeva.terminate(); self.zeva.wait(timeout=20)
 def run(self, request, job):
  blocks=self.contract.trial_registry.get(request.split_block_id, [])
  if not blocks: raise ValueError(f'unknown split block: {request.split_block_id}')
  item=blocks[int(job.job_id.rsplit('-',1)[-1]) % len(blocks)]
  cfg=self.contract.rollout; trial_dir=self.root/'trials'; trial_dir.mkdir(exist_ok=True)
  attempt=self.root/'attempts'/job.job_id; attempt.parent.mkdir(exist_ok=True)
  limits=cfg.get('runtime_limits'); python=cfg.get('python',sys.executable); self._start_zeva(cfg)
  if not limits: raise ValueError('campaign rollout.runtime_limits is required')
  trial={'schema_version':'arx.rollout.trial.v1','trial_id':job.job_id,'mode':'baseline','environment':{k:item[k] for k in ('scene','mapping','task','model_contract','seed')},'vla':{'host':cfg.get('vla_host','127.0.0.1'),'port':int(cfg.get('vla_port',5581)),'expected_identity':{}},'gateway':{'python':python,'host':'127.0.0.1','port':int(cfg.get('gateway_port',8091))+(int(job.job_id.rsplit('-',1)[-1])%1000),'runtime_limits':str(Path(limits).resolve())},'runner_limits':cfg.get('runner_limits',{'startup_timeout_s':60,'episode_timeout_s':1800,'reconciliation_timeout_s':360,'shutdown_timeout_s':15,'heartbeat_interval_s':10,'max_tool_attempts':100,'max_agent_calls':10,'max_recovery_agent_calls':5,'max_contract_retries':1,'max_event_records':10000}),'evaluation':'none'}
  active=getattr(self, 'active_package', None)
  if request.mode == 'candidate':
   if not active or active['sha256'] != request.package_sha256: raise ValueError('requested candidate package is not the active sealed package')
   d=cfg.get('deployment',{}); trial['mode']='candidate'; trial['candidate']={'package':active['path'],'package_sha256':active['sha256'],'contract_sha256':d['contract_sha256'],'catalog_sha256':d['catalog_sha256'],'bootstrap_sha256':d['bootstrap_sha256'],'critic_runtime_limits':d['critic_runtime_limits']}; trial['agent']=d['agent']
  path=trial_dir/(job.job_id+'.json'); path.write_text(json.dumps(trial,indent=2))
  subprocess.run([python,'-m','scripts.deployment.run_arx_evolution_rollout','--trial-config',str(path),'--output',str(attempt)],cwd=Path.cwd(),check=False)
  result=json.loads((attempt/'result.json').read_text()); return result
