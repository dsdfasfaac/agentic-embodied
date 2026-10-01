# Copyright (c) 2026 Zetta Contributors
"""Typed top-level learner tools with request-id idempotency."""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .contracts import *


class LearningTools:
 def __init__(self,store,contract,runner=None): self.store,self.contract,self.runner=store,contract,runner; self.requests={}; self.groups={}; self.drafts={}
 def read_contract(self, _input=None): return self.contract.model_dump()
 def request_rollouts(self,request):
  if request.request_id in self.requests:return self.requests[request.request_id]
  jobs=[]
  for i in range(request.count):
   jid=f'{request.request_id}-{i:03d}'; jobs.append(Job(job_id=jid,request_id=request.request_id,status='pending',arm=request.mode))
  group=JobGroup(group_id='group-'+uuid.uuid4().hex,jobs=jobs); self.requests[request.request_id]=group; self.groups[group.group_id]=group
  if self.runner:
   workers=min(len(jobs), int(self.contract.rollout.get('max_parallel_jobs', 1)))
   def execute(job):
    try: return job, self.runner(request,job), None
    except Exception as exc: return job, None, exc
   with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
    futures=[pool.submit(execute,j) for j in jobs]
    for future in as_completed(futures):
     job, result, error = future.result()
     if error: self._update_job(group,job,status='failed',error=type(error).__name__)
     else: self._update_job(group,job,status='completed',result_content_id=self.store.put('result',result))
  return group
 def _update_job(self, group, job, **updates):
  updated=job.model_copy(update=updates); group.jobs=[updated if x.job_id==job.job_id else x for x in group.jobs]; self.groups[group.group_id]=group
 def job_status(self,request): return self.groups[request.group_id]
 def read_evidence(self,request):
  evidence=self.store.read(request.content_id,offline=request.offline); return evidence.model_dump()
 def save_analysis(self,analysis): return self.store.put('analysis',analysis.model_dump())
 def write_candidate_file(self,request, relative_path=None, contents=None):
  # Keep the original learner API usable while typed tool dispatch migrates.
  if isinstance(request, str):
   draft_id, relative_path, contents = request, relative_path, contents
  else:
   draft_id,relative_path,contents=request.draft_id,request.relative_path,request.contents
  if not isinstance(draft_id, str) or not isinstance(relative_path, str) or not isinstance(contents, str):
   raise TypeError('write_candidate_file requires draft_id, relative_path, and contents')
  p=Path(relative_path)
  if p.is_absolute() or '..' in p.parts or not relative_path.startswith(('critic/','skill/','reentry/','tests/','evidence/')): raise ValueError('path not allowed')
  root=self.drafts.setdefault(draft_id,Path(self.store.root)/'drafts'/draft_id); target=root/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(contents);return {'sha256':__import__('hashlib').sha256(contents.encode()).hexdigest()}
 def submit_candidate(self,request):
  draft_id,manifest=request.draft_id,request.manifest
  from robots.arx.critics.packages import seal_candidate
  root=self.drafts[draft_id]
  package=seal_candidate(root,manifest)
  package_dir=self.store.root/'packages'/package.sha256; package_dir.mkdir(parents=True,exist_ok=False)
  for name, data in package.payloads: target=package_dir/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
  (package_dir/'manifest.json').write_bytes(package.manifest_bytes)
  cid=self.store.put('candidate',{'package_sha256':package.sha256,'draft_id':draft_id,'package':str(package_dir)})
  self.store.active_package={'sha256':package.sha256,'path':str(package_dir)}
  if self.runner is not None and hasattr(self.runner, '__self__'):
   self.runner.__self__.active_package=self.store.active_package
  return {'package_sha256':package.sha256,'content_id':cid,'package':str(package_dir)}
 def evaluate_candidate(self,request): return {'evaluation_job_id':'evaluation-'+uuid.uuid4().hex,'package_sha256':request.package_sha256,'gate':request.gate}
 def request_owner_change(self,request): return self.store.put('feedback',request.model_dump())
 def checkpoint_learning(self,checkpoint): return self.store.checkpoint(checkpoint)
