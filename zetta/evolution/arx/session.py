"""Stateful automated and interactive ARX learning orchestration."""
from __future__ import annotations
import json, uuid
from zetta.planner.base import build_planner
from .tool_adapter import build_toolkit
from .contracts import Checkpoint
from .store import LearningStore
from .tools import LearningTools

LEARNING_SYSTEM_PROMPT = """You are the top-level ARX learning agent. Produce an immutable observation-only critic and recovery skill for fresh deployment agents. Read the frozen contract, collect paired baseline/parent/candidate rollouts, inspect public events, tool results, invocations, RGB windows, execution errors, and successful controls. Keep execution/interface failures separate from physical-task failures; represent missed failures and unknown onset explicitly. Compare competing causal hypotheses with evidence and a discriminating test.
Author feature_schema.json, executable RGB-only critic/features.py, temporal rules, a self-contained skill, reentry/recovery bindings, replay cases, and claims. Every online condition must use permitted observations. Never use simulator state, contact/reward, seeds, private paths, hidden memory, or prerecorded actions. Run package preflight, isolation, mock-gateway contract tests, and replay on failures and successes. Do not change gateway contracts, budgets, splits, labels, or promotion policy; request an owner change when needed. Submit immutable packages, checkpoint evidence/hypotheses/budgets/next action, and stop at harness terminal decision."""

class LearningSession:
    def __init__(self, root, contract, planner=None, runner=None):
        self.store=LearningStore(root); self.contract=contract; self.planner=planner
        self.tools=LearningTools(self.store, contract, runner); self.session_id=uuid.uuid4().hex
    def prepare(self): self.store.save_contract(self.contract); return self.checkpoint('PREPARE', [], {}, 'collect baseline')
    def checkpoint(self, phase, evidence, budget, next_action):
        c=Checkpoint(checkpoint_id='checkpoint-'+uuid.uuid4().hex, phase=phase, evidence_ids=evidence, hypotheses=[], remaining_budget=budget, next_action=next_action, logical_session_id=self.session_id); self.tools.checkpoint_learning(c); return c
    def turn(self, phase, context):
        cp=self.store.load_checkpoint(); payload={'schema_version':'arx.learning.turn.v1','phase':phase,'contract_id':self.contract.contract_id,'parent_package_id':context.get('parent_package_id'),'checkpoint_id':cp.checkpoint_id if cp else None,'new_feedback_ids':context.get('new_feedback_ids',[]),'allowed_actions':context.get('allowed_actions',[]),'remaining_budget':context.get('remaining_budget',self.contract.learning_budgets),'required_output':context.get('required_output','')}
        if not self.planner: return payload
        toolkit=build_toolkit(self.tools, context.get('allowed_actions',[]))
        planner=self.planner() if callable(self.planner) and not hasattr(self.planner,'solve') else self.planner
        result=planner.solve(system_prompt=LEARNING_SYSTEM_PROMPT,user_message=json.dumps(payload),toolkit=toolkit,max_turns=int(self.contract.learner.get('max_turns',4)))
        return {'finish_result':result.finish_result,'stats':result.stats,'error':result.error}
    def automated(self, phases=('COLLECT','ANALYZE','AUTHOR')):
        self.prepare(); out=[]; actions=['read_contract','request_rollouts','job_status','read_evidence','save_analysis','write_candidate_file','submit_candidate','evaluate_candidate','checkpoint_learning']
        for phase in phases: out.append(self.turn(phase, {'allowed_actions':actions,'required_output':'evidence-backed next action'})); self.checkpoint(phase,[],self.contract.learning_budgets,'continue')
        return out
    def interactive(self):
        self.prepare(); print(LEARNING_SYSTEM_PROMPT)
        while True:
            line=input('learning> ').strip()
            if line in ('quit','exit'): return
            try:
                # The interactive shell accepts a small, auditable JSON RPC envelope.
                # Planner/provider code is never given direct filesystem or gateway access.
                if line.startswith('{'):
                    request = json.loads(line)
                    tool = request.pop('tool', None)
                    if not tool or not hasattr(self.tools, tool):
                        raise ValueError('unknown learner tool')
                    result = getattr(self.tools, tool)(request) if request else getattr(self.tools, tool)()
                    print(json.dumps(result.model_dump() if hasattr(result, 'model_dump') else result, indent=2, default=str))
                else:
                    print(json.dumps(self.turn('REFINE', {'new_feedback_ids':[line] if line else [],'allowed_actions':['read_contract','read_evidence','write_candidate_file','submit_candidate','request_owner_change','checkpoint_learning'],'required_output':'candidate submission or evidence-backed owner change'}), indent=2))
            except Exception as exc: print(json.dumps({'error':str(exc)}))
