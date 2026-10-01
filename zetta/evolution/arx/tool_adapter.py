"""Strict adapter between planner JSON calls and ARX learner handlers."""
from __future__ import annotations
from typing import Any, Callable
from pydantic import BaseModel, create_model
from zetta.tools.toolkit import Toolkit

def _empty_model(name: str):
    return create_model(name, __config__={'extra':'forbid'})

def register_typed_tool(toolkit: Toolkit, name: str, model: type[BaseModel], handler: Callable[..., Any], description: str):
    schema = model.model_json_schema()
    def invoke(**kwargs):
        value = model.model_validate(kwargs)
        result = handler(value)
        return result.model_dump() if hasattr(result, 'model_dump') else result
    toolkit.add_tool(name, {'name': name, 'description': description, 'input_schema': schema}, invoke)

def build_toolkit(tools: Any, names: list[str]) -> Toolkit:
    from .contracts import RolloutRequest, Analysis, Checkpoint, EvidenceReadRequest, JobStatusRequest, CandidateFileRequest, CandidateSubmissionRequest, CandidateEvaluationRequest, OwnerChangeRequest
    models = {'read_contract': _empty_model('ReadContractInput'), 'request_rollouts': RolloutRequest, 'job_status': JobStatusRequest, 'read_evidence': EvidenceReadRequest, 'save_analysis': Analysis, 'write_candidate_file': CandidateFileRequest, 'submit_candidate': CandidateSubmissionRequest, 'evaluate_candidate': CandidateEvaluationRequest, 'request_owner_change': OwnerChangeRequest, 'checkpoint_learning': Checkpoint}
    tk = Toolkit(); tk.retain_tools(set())
    for name in names: register_typed_tool(tk, name, models[name], getattr(tools, name), f'ARX learner tool {name}; arguments are strictly validated.')
    return tk
