# Copyright (c) 2026 Zetta Contributors
"""Frozen single-trial inputs and learner-facing results."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from robots.arx.critics.contracts import SHA
from robots.arx.gateway.contracts import ID, StrictModel

Positive = Annotated[int, Field(gt=0)]
Seconds = Annotated[float, Field(gt=0)]


class Environment(StrictModel):
    scene: str
    mapping: str
    task: str
    model_contract: str
    calibration: str | None = None
    privileged: bool = False
    seed: int


class Vla(StrictModel):
    host: str
    port: Annotated[int, Field(ge=1, le=65535)]
    expected_identity: dict[str, str] = Field(default_factory=dict)


class Gateway(StrictModel):
    python: str
    host: Literal["127.0.0.1", "::1"]
    port: Annotated[int, Field(ge=1, le=65535)]
    runtime_limits: str


class Candidate(StrictModel):
    package: str
    package_sha256: SHA
    contract_sha256: SHA
    catalog_sha256: SHA
    bootstrap_sha256: SHA
    critic_runtime_limits: str


class AgentSettings(StrictModel):
    planner_type: Literal["api"]
    model: str
    reasoning_effort: Literal["low", "medium", "high", "xhigh"]
    max_tokens: Positive
    max_turns: Positive
    timeout_s: Seconds
    credential_env: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]


class RunnerLimits(StrictModel):
    startup_timeout_s: Seconds
    episode_timeout_s: Seconds
    reconciliation_timeout_s: Seconds
    shutdown_timeout_s: Seconds
    heartbeat_interval_s: Seconds
    max_tool_attempts: Positive
    max_agent_calls: Positive
    max_recovery_agent_calls: Positive
    max_contract_retries: Annotated[int, Field(ge=0)]
    max_event_records: Positive = 10000


class Trial(StrictModel):
    schema_version: Literal["arx.rollout.trial.v1"]
    trial_id: ID
    mode: Literal["baseline", "candidate"]
    environment: Environment
    vla: Vla
    gateway: Gateway
    candidate: Candidate | None = None
    agent: AgentSettings | None = None
    runner_limits: RunnerLimits
    evaluation: Literal["none"]

    @model_validator(mode="after")
    def mode_fields(self):
        if self.runner_limits.reconciliation_timeout_s < 30:
            raise ValueError("reconciliation timeout must allow gateway operation completion")
        if self.mode == "candidate" and (self.candidate is None or self.agent is None):
            raise ValueError("candidate mode requires package and agent settings")
        if self.mode == "baseline" and (
            self.candidate is not None or self.agent is not None
        ):
            raise ValueError("baseline cannot configure candidate or agent")
        paths = [
            self.environment.scene,
            self.environment.mapping,
            self.environment.task,
            self.environment.model_contract,
            self.gateway.python,
            self.gateway.runtime_limits,
        ]
        if self.environment.calibration:
            paths.append(self.environment.calibration)
        if self.candidate:
            paths += [self.candidate.package, self.candidate.critic_runtime_limits]
        if any(not Path(p).is_absolute() for p in paths):
            raise ValueError("trial paths must be absolute")
        if self.vla.expected_identity:
            raise ValueError(
                "VLA identity attestation is not supported by the current gateway; use empty expected_identity explicitly"
            )
        return self


class Decision(StrictModel):
    schema_version: Literal["arx.deployment.decision.v1"]
    event_id: ID
    observation_id: ID
    control_epoch: Annotated[int, Field(ge=0)]
    tool: str
    arguments: dict[str, Any]
    evidence_ids: Annotated[list[ID], Field(max_length=90)]
    rationale: Annotated[str, Field(min_length=1, max_length=2048)]


class Outcome(StrictModel):
    task_success: bool | None = None
    evaluator_id: str | None = None
    recovery_attempted: bool = False
    reentry_completed: bool = False


class Counts(StrictModel):
    physical_steps: int = 0
    tool_attempts: int = 0
    rejected_tools: int = 0
    agent_calls: int = 0
    recoveries: int = 0


class RolloutResult(StrictModel):
    schema_version: Literal["arx.rollout.result.v1"] = "arx.rollout.result.v1"
    trial_id: str
    attempt_id: ID
    episode_id: str | None = None
    mode: str
    package_sha256: str | None = None
    identities: dict[str, Any] = Field(default_factory=dict)
    status: Literal[
        "completed",
        "configuration_error",
        "infrastructure_error",
        "agent_error",
        "execution_uncertain",
        "interrupted",
    ]
    termination_reason: str
    outcome: Outcome = Field(default_factory=Outcome)
    counts: Counts = Field(default_factory=Counts)
    last_operation: dict[str, Any] | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    cleanup_status: str
    error: dict[str, str] | None = None


EXIT_CODES = {
    "completed": 0,
    "configuration_error": 2,
    "infrastructure_error": 3,
    "agent_error": 3,
    "execution_uncertain": 3,
    "interrupted": 130,
}
