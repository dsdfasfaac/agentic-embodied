# Copyright (c) 2026 Zetta Contributors
"""Strict ARX learning-session contracts and public evidence records."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from robots.arx.gateway.contracts import ID, StrictModel

SHA=Annotated[str,Field(pattern=r'^[0-9a-f]{64}$')]
class CampaignContract(StrictModel):
 schema_version: Literal['arx.learning.contract.v1','arx.learning.contract.v2']
 contract_id: ID
 task: str
 evaluator_digest: SHA
 gateway_digest: SHA
 critic_abi_digest: SHA
 deployment_bootstrap_sha256: SHA
 trial_budgets: dict[str,int]
 learning_budgets: dict[str,int]
 training_split: list[ID]
 regression_split: list[ID]
 heldout_split: list[ID]
 gate_policy: dict[str,Any]
 artifact_visibility: dict[str,Literal['public','private','offline_authorized']]
 trial_registry: dict[str, list[dict[str, Any]]] = {}
 rollout: dict[str, Any] = {}
 learner: dict[str, Any] = {}
 deployment: dict[str, Any] = {}
 budgets: dict[str, int] = {}

class RolloutRequest(StrictModel):
 schema_version: Literal['arx.learning.rollout_request.v1']='arx.learning.rollout_request.v1'
 request_id: ID
 mode: Literal['baseline','parent','candidate']
 package_sha256: SHA|None=None
 split_block_id: ID
 trial_config: str
 count: Annotated[int,Field(ge=1,le=128)]=1
 @model_validator(mode='after')
 def package(self):
  if self.mode=='candidate' and not self.package_sha256: raise ValueError('candidate package required')
  if self.mode!='candidate' and self.package_sha256 is not None: raise ValueError('package only for candidate')
  return self
class EvidenceReadRequest(StrictModel):
 content_id: ID
 offline: bool = False
 frame_selector: dict[str, Any] | None = None
class JobStatusRequest(StrictModel):
 group_id: ID
class CandidateFileRequest(StrictModel):
 draft_id: ID
 relative_path: str
 contents: str
class CandidateSubmissionRequest(StrictModel):
 draft_id: ID
 manifest: dict[str, Any]
class CandidateEvaluationRequest(StrictModel):
 package_sha256: SHA
 gate: Literal['replay','contract','development','paired','regression','heldout']
class OwnerChangeRequest(StrictModel):
 limitation: str
 evidence_ids: list[ID] = []
 proposed_scope: str
class Job(StrictModel):
 job_id: ID
 request_id: ID
 status: Literal['pending','running','completed','invalid','failed']
 arm: str
 result_content_id: str|None=None
 error: str|None=None
class JobGroup(StrictModel):
 schema_version: Literal['arx.learning.job_group.v1']='arx.learning.job_group.v1'
 group_id: ID
 jobs: list[Job]
class Evidence(StrictModel):
 content_id: ID
 kind: Literal['result','event','tool','invocation','frame','attempt','analysis','feedback']
 payload: dict[str,Any]
 visibility: Literal['public','offline_authorized']
class AttemptRecord(StrictModel):
 schema_version: Literal['arx.learning.attempt.v1']='arx.learning.attempt.v1'
 attempt_id: ID
 job_id: ID
 arm: str
 package_sha256: SHA|None
 result_content_id: ID
 status: str
 termination_reason: str
 task_success: bool|None
 write_certainty: str
 physical_steps: int
 tool_attempts: int
 agent_calls: int
 recoveries: int
 event_content_ids: list[ID]=[]
 tool_content_ids: list[ID]=[]
 invocation_content_ids: list[ID]=[]
 critic_incident_ids: list[ID]=[]
 feedback_ids: list[ID]=[]
 visibility_policy: Literal['public']='public'
class Hypothesis(StrictModel):
 claim: str
 support_ids: list[ID]
 counterevidence_ids: list[ID]
 distinguishing_test: str
class AnalysisCluster(StrictModel):
 cluster_id: ID
 member_incident_ids: list[ID]
 failure_axis: Literal['execution','physical']
 visual_or_error_signature: str
 evidence_ids: list[ID]
 successful_control_ids: list[ID]
 onset: int|Literal['unknown']
 hypotheses: list[Hypothesis]
 selected_hypothesis: str|None=None
 unresolved_reason: str|None=None
 proposed_change_surface: Literal['critic','skill','reentry','owner_runtime']
class Analysis(StrictModel):
 schema_version: Literal['arx.learning.analysis.v1']
 analysis_id: ID
 clusters: list[AnalysisCluster]

class Segment(StrictModel):
 schema_version: Literal['arx.learning.segment.v1']='arx.learning.segment.v1'
 segment_id: ID
 failure_axis: Literal['execution','physical']
 failure_class: str
 onset: int|Literal['unknown']
 summary: str
 state_error_signature: str
 evidence_ids: list[ID]=[]
 success_control_ids: list[ID]=[]

class ClusterReview(StrictModel):
 schema_version: Literal['arx.learning.cluster_review.v1']='arx.learning.cluster_review.v1'
 assignments: dict[ID, ID]
 unresolved_segment_ids: list[ID]=[]
class Checkpoint(StrictModel):
 schema_version: Literal['arx.learning.checkpoint.v1']='arx.learning.checkpoint.v1'
 checkpoint_id: ID
 phase: Literal['PREPARE','COLLECT','ANALYZE','AUTHOR','PREFLIGHT','REPLAY','CONTRACT','DEVELOPMENT','REFINE','PAIRED_GATE','REGRESSION','HELDOUT','PROMOTE','COMPLETE']
 evidence_ids: list[ID]
 hypotheses: list[str]
 remaining_budget: dict[str,int]
 next_action: str
 logical_session_id: ID
