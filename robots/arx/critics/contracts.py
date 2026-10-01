# Copyright (c) 2026 Zetta Contributors
"""Candidate-authored feature declarations and trusted critic contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, model_validator

from robots.arx.gateway.contracts import ID, StrictModel

SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(pattern=r"^(?:visual|privileged)\.[a-zA-Z0-9_.]+$", max_length=128)]
Camera = Literal["front_rgb", "left_rgb", "right_rgb"]
Scalar = float | int | bool | str


class FeatureValue(StrictModel):
    valid: bool
    value: Scalar | None

    @model_validator(mode="after")
    def validity(self):
        if self.valid == (self.value is None):
            raise ValueError("valid features need a scalar; invalid features need null")
        return self


class FeatureDeclaration(StrictModel):
    name: Name
    scalar_type: Literal["number", "integer", "boolean", "string"]
    units: Annotated[str, Field(min_length=1, max_length=128)]
    source_cameras: Annotated[list[Camera], Field(min_length=0, max_length=3)]
    extractor_version: ID
    validity_condition: Annotated[str, Field(min_length=1, max_length=2048)]
    sampling: Literal["one_per_observation"]
    history_window: Annotated[int, Field(ge=1, le=1024)]
    phase_reset_policy: Literal["episode_reset_only"]
    value_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def valid_range(self):
        if self.value_range and (
            self.scalar_type not in ("number", "integer")
            or self.value_range[0] > self.value_range[1]
        ):
            raise ValueError("invalid feature range")
        if len(set(self.source_cameras)) != len(self.source_cameras):
            raise ValueError("duplicate cameras")
        return self


class FeatureSchema(StrictModel):
    schema_version: Literal["arx.features.schema.v1"]
    features: Annotated[list[FeatureDeclaration], Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def unique(self):
        if len({x.name for x in self.features}) != len(self.features):
            raise ValueError("duplicate features")
        return self


class SourceSpec(StrictModel):
    kind: Literal["package_python", "prm_service"]
    api_version: Literal["arx.features.v1"]


class CriticManifest(StrictModel):
    schema_version: Literal["arx.critic.manifest.v1"]
    critic_id: ID
    input_profile: Literal["arx.rgb_only.v1", "arx.prm_inputs.v1"]
    implementation: Literal["feature_rules"]
    source: SourceSpec
    entrypoint: Literal["features.py:FeatureExtractor"]
    code_sha256: SHA
    config_sha256: SHA
    feature_schema_sha256: SHA
    cameras: Annotated[list[Camera], Field(min_length=1, max_length=3)]
    history_limit: Annotated[int, Field(ge=1, le=1024)]
    evaluation_timeout_ms: Annotated[int, Field(ge=1, le=60000)]
    failure_modes: Annotated[list[ID], Field(min_length=1, max_length=128)]
    recovery_bindings: dict[ID, ID]

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.cameras)) != len(self.cameras) or len(
            set(self.failure_modes)
        ) != len(self.failure_modes):
            raise ValueError("duplicate manifest entries")
        if set(self.recovery_bindings) != set(self.failure_modes):
            raise ValueError("every failure mode requires a binding")
        return self


class Predicate(StrictModel):
    feature: Name
    operator: Literal["lt", "le", "gt", "ge", "eq", "ne"]
    threshold: Scalar


class Rule(StrictModel):
    rule_id: ID
    title: Annotated[str, Field(min_length=1, max_length=512)]
    feature: Name
    operator: Literal["lt", "le", "gt", "ge", "eq", "ne", "stagnant"]
    threshold: Scalar
    dwell_steps: Annotated[int, Field(ge=1, le=1024)]
    cooldown_steps: Annotated[int, Field(ge=0, le=10000)]
    proposal: Literal["interrupt"]
    activation_conditions: Annotated[list[Predicate], Field(max_length=32)]
    evidence_ids: Annotated[list[ID], Field(min_length=1, max_length=64)]


class CriticConfig(StrictModel):
    schema_version: Literal["arx.critic.config.v1"]
    extractor_config: dict[str, Any]
    rules: Annotated[list[Rule], Field(min_length=1, max_length=128)]
    rule_failure_modes: dict[ID, ID]


class ImageReference(StrictModel):
    content_id: ID
    sha256: SHA
    width: Annotated[int, Field(ge=1, le=4096)]
    height: Annotated[int, Field(ge=1, le=4096)]
    encoding: Literal["png"]


class CriticObservation(StrictModel):
    schema_version: Literal["arx.critic.observation.v1"] = "arx.critic.observation.v1"
    episode_nonce: ID
    observation_id: ID
    step_index: Annotated[int, Field(ge=0)]
    simulation_time_s: Annotated[float, Field(ge=0)]
    cameras: dict[Camera, ImageReference]
    lifecycle: Literal["reset", "nominal", "recovery", "reentry"]
    event_sequence: Annotated[int, Field(ge=0)]
    privileged: dict[str, Any] | None = None

    @classmethod
    def from_public(cls, value):
        # Construct a separate object; never forward arbitrary gateway keys.
        fields = {
            name: value[name]
            for name in cls.model_fields
            if name != "schema_version" and name in value
        }
        return cls.model_validate(fields)


class WorkerLimits(StrictModel):
    python: str
    max_history: Annotated[int, Field(ge=1, le=1024)]
    max_evaluation_ms: Annotated[int, Field(ge=1, le=60000)]
    startup_timeout_s: Annotated[float, Field(gt=0, le=120)]
    memory_bytes: Annotated[int, Field(ge=268435456)]
    cpu_seconds: Annotated[int, Field(ge=1)]
    scratch_bytes: Annotated[int, Field(ge=4096, le=67108864)]
    image_width: Annotated[int, Field(ge=1, le=4096)]
    image_height: Annotated[int, Field(ge=1, le=4096)]
    max_message_bytes: Annotated[int, Field(ge=4096, le=268435456)]


class FeatureSource(Protocol):
    def reset(self, observation: dict, images: dict, config: dict) -> None: ...
    def extract(self, observation: dict, images: dict) -> dict: ...
    def close(self) -> None: ...


class PrmRequest(StrictModel):
    """Reserved future public profile, never accepted by the RGB worker."""

    schema_version: Literal["arx.prm_inputs.v1"]
    request_id: ID
    observation_id: ID
    task_instruction: str
    current_images: dict[Camera, ImageReference]
    target_images: dict[Camera, ImageReference]
    model_version: ID


class PrmResponse(StrictModel):
    request_id: ID
    observation_id: ID
    score: float | None
    valid: bool
    model_version: ID


class PrmFeatureSource:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PRM_SOURCE_NOT_IMPLEMENTED: no fallback to RGB or baseline"
        )
