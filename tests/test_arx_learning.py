# Copyright (c) 2026 Zetta Contributors
from pathlib import Path
import json
from zetta.evolution.arx.contracts import CampaignContract,RolloutRequest,Analysis,Checkpoint
from zetta.evolution.arx.store import LearningStore
from zetta.evolution.arx.tools import LearningTools
from zetta.evolution.arx.session import LearningSession

def contract():
 return CampaignContract(schema_version='arx.learning.contract.v1',contract_id='c',task='pickup',evaluator_digest='a'*64,gateway_digest='b'*64,critic_abi_digest='c'*64,deployment_bootstrap_sha256='d'*64,trial_budgets={'steps':10},learning_budgets={'trials':2},training_split=['train'],regression_split=['reg'],heldout_split=['hold'],gate_policy={'min':0.5},artifact_visibility={'result':'public'})

def test_store_public_visibility_and_checkpoint(tmp_path):
 s=LearningStore(tmp_path); cid=s.put('result',{'ok':True}); assert s.read(cid).payload['ok']; c=contract();s.save_contract(c);assert s.contract().contract_id=='c'; session=LearningSession(tmp_path,c);cp=session.checkpoint('PREPARE',[],{'trials':1},'next');assert s.load_checkpoint().checkpoint_id==cp.checkpoint_id

def test_tools_request_id_idempotency_and_paths(tmp_path):
 s=LearningStore(tmp_path);t=LearningTools(s,contract()); req=RolloutRequest(request_id='r',mode='baseline',split_block_id='train',trial_config='/tmp/t',count=2); a=t.request_rollouts(req); assert t.request_rollouts(req).group_id==a.group_id and len(a.jobs)==2
 assert t.write_candidate_file('d','critic/features.py','x')['sha256'];
 try:t.write_candidate_file('d','../x','x');assert False
 except ValueError:pass

def test_automated_and_interactive_payload(tmp_path):
 s=LearningSession(tmp_path,contract()); out=s.automated(('COLLECT',)); assert out[0]['phase']=='COLLECT'; assert s.store.load_checkpoint().phase=='COLLECT'
