# Copyright (c) 2026 Zetta Contributors
"""Strict, public ARX gateway contracts (no simulator telemetry)."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
Positive = Annotated[int, Field(gt=0)]
Vector = Annotated[list[float], Field(min_length=3, max_length=3)]
State = Literal[
    "READY",
    "RUNNING_NOMINAL",
    "INTERRUPTED",
    "RECOVERING",
    "EXECUTION_UNCERTAIN",
    "ENDED",
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )


class ZevaArgs(StrictModel):
    max_chunks: Annotated[int, Field(ge=1, le=4)]
    reentry_token: ID | None = None


class HoldArgs(StrictModel):
    steps: Annotated[int, Field(ge=1, le=15)]


class GripperArgs(StrictModel):
    opening: Annotated[float, Field(ge=0, le=1)]
    max_steps: Annotated[int, Field(ge=1, le=60)]


class EefArgs(StrictModel):
    delta_xyz_m: Vector
    delta_rotvec_rad: Vector = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    frame: Literal["world", "tool"] = "tool"
    speed_m_s: Annotated[float, Field(gt=0, le=0.03)] = 0.01

    @model_validator(mode="after")
    def norms(self):
        if sum(x * x for x in self.delta_xyz_m) > 0.01**2 + 1e-15:
            raise ValueError("translation norm exceeds 0.01 m")
        if sum(x * x for x in self.delta_rotvec_rad) > 0.1**2 + 1e-15:
            raise ValueError("rotation norm exceeds 0.1 rad")
        return self


class ReviewArgs(StrictModel):
    observation_ids: Annotated[list[ID], Field(min_length=1, max_length=30)]

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("duplicate observations")
        return self


class FinishArgs(StrictModel):
    reason: Annotated[str, Field(min_length=1, max_length=2048)]


class ToolRequest(StrictModel):
    schema_version: Literal["arx.tool.request.v1"] = "arx.tool.request.v1"
    request_id: ID
    decision_ref: ID
    observation_id: ID
    control_epoch: Annotated[int, Field(ge=0)]
    tool: Annotated[str, Field(max_length=128)]
    arguments: dict[str, Any]
    evidence_ids: Annotated[list[ID], Field(max_length=90)] = Field(
        default_factory=list
    )
    reason: Annotated[str, Field(max_length=2048)] = ""


class Error(StrictModel):
    code: str
    phase: str
    retry_class: Literal["correct_request", "never", "query_same_request"]
    public_message: str


class ExecutionOutput(StrictModel):
    completion: Literal[
        "plan_exhausted",
        "budget_exhausted",
        "critic_interrupted",
        "environment_ended",
        "cancelled",
        "error",
    ]
    last_committed_step: Annotated[int, Field(ge=0)]
    command_target_reached: bool | None
    physical_arrival_verified: Literal[False] = False


class Proposal(StrictModel):
    detector_id: ID
    failure_mode: ID
    rule_id: ID
    evidence_observation_ids: Annotated[list[ID], Field(min_length=1, max_length=60)]
    reason_code: ID
    summary: Annotated[str, Field(max_length=2048)]
    limitations: Annotated[list[str], Field(max_length=20)] = Field(
        default_factory=list
    )
    proposal: Literal["interrupt"] = "interrupt"


class Assessment(StrictModel):
    schema_version: Literal["arx.critic.assessment.v1"] = "arx.critic.assessment.v1"
    critic_id: ID
    observation_id: ID
    step_index: Annotated[int, Field(ge=0)]
    status: Literal["clear", "unknown", "failure"]
    events: Annotated[list[Proposal], Field(max_length=64)] = Field(
        default_factory=list
    )
    features: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "failure") != bool(self.events):
            raise ValueError("only failure assessments contain proposals")
        return self


class ReentryCheck(StrictModel):
    check_id: ID
    status: Literal["pass", "fail", "unknown"]
    evidence_ids: list[ID]
    reason_code: ID


class ReentryAssessment(StrictModel):
    schema_version: Literal["arx.reentry.assessment.v1"] = "arx.reentry.assessment.v1"
    recovery_id: ID
    observation_id: ID
    policy_id: ID
    status: Literal["eligible", "ineligible", "unknown"]
    checks: Annotated[list[ReentryCheck], Field(min_length=1)]


class ReviewOutput(StrictModel):
    assessment: ReentryAssessment
    reentry_token: ID | None


class FinishOutput(StrictModel):
    closed: Literal[True] = True
    finalization: Literal["complete", "incomplete"]


class RecoveryBinding(StrictModel):
    binding_id: ID
    failure_modes: Annotated[list[ID], Field(min_length=1)]
    skill_entrypoint: str
    allowed_tools: list[str]
    max_recovery_steps: Annotated[int, Field(ge=0)]
    max_agent_decisions: Positive
    monitor_policy: Literal["recovery_local", "stop_all_motion"]
    reentry_policy_id: ID

    @model_validator(mode="after")
    def motion_budget(self):
        if self.max_recovery_steps == 0 and self.monitor_policy != "stop_all_motion":
            raise ValueError("zero motion budget is only valid for stop-only bindings")
        return self


class RuntimeLimits(StrictModel):
    max_steps: Positive
    max_decisions: Positive
    max_recoveries: Positive
    operation_timeout_s: Annotated[float, Field(gt=0)]
    critic_timeout_s: Annotated[float, Field(gt=0)]
    idle_agent_timeout_s: Annotated[float, Field(gt=0)]
    lease_timeout_s: Annotated[float, Field(gt=0)]
    shutdown_timeout_s: Annotated[float, Field(gt=0)]


class GatewayError(Exception):
    def __init__(self, code: str, message: str = "Request cannot be executed"):
        super().__init__(message)
        self.code = code


class ToolResult(StrictModel):
    schema_version: Literal["arx.tool.result.v1"] = "arx.tool.result.v1"
    request_id: ID
    operation_id: ID
    event_sequence: Annotated[int, Field(ge=0)]
    status: Literal[
        "accepted",
        "running",
        "completed",
        "interrupted",
        "rejected",
        "failed",
        "cancelled",
        "unknown",
    ]
    tool: str
    observation_id_before: ID
    observation_id_after: ID
    control_epoch: Annotated[int, Field(ge=0)]
    executed_steps: Annotated[int, Field(ge=0)]
    planned_steps: Annotated[int, Field(ge=0)] | None
    write_certainty: Literal["none", "known_partial", "completed", "unknown"]
    error: Error | None
    critic_event_ids: list[ID]
    result: dict[str, Any] | None
    budget_remaining: dict[str, int]
    recovery_context: dict[str, Any] | None
