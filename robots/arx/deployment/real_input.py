# Copyright (c) 2026 Zetta Contributors
"""Fail-closed input preflight for a future ARX real-robot deployment."""

from __future__ import annotations

import dataclasses
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from robots.arx.contracts import ARX_ACTION_DIM, ARX_CAMERA_NAMES, load_model_contract, load_task_manifest
from robots.arx.critics.contracts import SHA
from robots.arx.gateway.contracts import (
    EefArgs,
    FinishArgs,
    GripperArgs,
    HoldArgs,
    ReviewArgs,
    StrictModel,
    ZevaArgs,
    digest,
)
from zetta.evolution.jsonio import canonical_sha256, file_sha256
from zetta.evolution.models import (
    CandidateBundle,
    CriticPredicate,
    CriticRule,
    RecoveryRule,
    RecoveryStep,
)

_TOOL_MODELS = {
    "arx.finish": FinishArgs,
    "arx.hold": HoldArgs,
    "arx.move_eef": EefArgs,
    "arx.review_reentry": ReviewArgs,
    "arx.set_gripper": GripperArgs,
    "arx.zeva": ZevaArgs,
}


class RealCamera(StrictModel):
    name: Literal["front_rgb", "left_rgb", "right_rgb"]
    device_id: str = Field(min_length=1, max_length=128)
    calibration_id: str = Field(min_length=1, max_length=128)
    calibration_sha256: SHA
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: Literal[3]
    dtype: Literal["uint8"]
    color_order: Literal["RGB"]


class RealFeatureSource(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    provider_id: str = Field(min_length=1, max_length=128)
    provider_sha256: SHA
    source_kind: Literal["camera_rgb", "camera_rgbd", "joint_feedback", "fused", "rgbd_fused"]
    source_ids: list[str] = Field(min_length=1, max_length=17)
    scalar_type: Literal["number", "integer", "boolean", "string"]
    units: str = Field(min_length=1, max_length=64)
    max_age_ms: int = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def unique_sources(self):
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("feature source IDs must be unique")
        return self


class RealInputContract(StrictModel):
    schema_version: Literal["arx.real.input.v1"]
    candidate_sha256: SHA
    candidate_file_sha256: SHA
    task_manifest_sha256: SHA
    task_id: int = Field(ge=0, lt=7)
    task_name: str = Field(min_length=1)
    model_contract_sha256: SHA
    tool_catalog_sha256: SHA
    cameras: list[RealCamera] = Field(min_length=3, max_length=3)
    depth_cameras: list[str] = Field(default_factory=list)
    joint_channels: list[str] = Field(min_length=ARX_ACTION_DIM, max_length=ARX_ACTION_DIM)
    auxiliary_channels: list[str] = Field(default_factory=list, max_length=16)
    feature_sources: list[RealFeatureSource] = Field(min_length=1, max_length=128)
    max_critic_history_steps: int = Field(ge=1, le=1024)
    max_critic_cooldown_steps: int = Field(ge=0, le=10000)
    max_recovery_tool_calls: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def unique_names(self):
        if tuple(camera.name for camera in self.cameras) != ARX_CAMERA_NAMES:
            raise ValueError("real cameras must match the pinned Task7 camera order")
        if len(set(self.joint_channels)) != ARX_ACTION_DIM or any(
            not name.strip() for name in self.joint_channels
        ):
            raise ValueError("14 distinct joint feedback channels are required")
        if len(set(self.auxiliary_channels)) != len(self.auxiliary_channels):
            raise ValueError("duplicate auxiliary feedback channel")
        if len({feature.name for feature in self.feature_sources}) != len(self.feature_sources):
            raise ValueError("duplicate feature source")
        if len(set(self.depth_cameras)) != len(self.depth_cameras) or not set(self.depth_cameras) <= {
            camera.name.removesuffix("_rgb") + "_depth_mm" for camera in self.cameras
        }:
            raise ValueError("depth cameras must correspond to declared RGB streams")
        return self


class LiveCapabilities(StrictModel):
    """Snapshot returned by the trusted real adapter before robot motion."""

    schema_version: Literal["arx.real.capabilities.v1"]
    robot_id: str = Field(min_length=1, max_length=128)
    cameras: list[RealCamera] = Field(min_length=3, max_length=3)
    depth_cameras: list[str] = Field(default_factory=list)
    joint_channels: list[str] = Field(min_length=ARX_ACTION_DIM, max_length=ARX_ACTION_DIM)
    auxiliary_channels: list[str] = Field(default_factory=list, max_length=16)
    feature_sources: list[RealFeatureSource] = Field(max_length=128)
    tool_catalog_sha256: SHA

    @model_validator(mode="after")
    def unique_names(self):
        if len({camera.name for camera in self.cameras}) != len(self.cameras):
            raise ValueError("duplicate live camera")
        if len(set(self.joint_channels)) != ARX_ACTION_DIM:
            raise ValueError("duplicate live joint channel")
        if len(set(self.auxiliary_channels)) != len(self.auxiliary_channels):
            raise ValueError("duplicate live auxiliary channel")
        if len({feature.name for feature in self.feature_sources}) != len(self.feature_sources):
            raise ValueError("duplicate live feature")
        return self


def _json_object(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"{path}: non-finite JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _exact_keys(value: dict[str, Any], cls: type, context: str) -> None:
    expected = {field.name for field in dataclasses.fields(cls)}
    if set(value) != expected:
        raise ValueError(
            f"{context} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _load_bundle(path: Path) -> tuple[CandidateBundle, dict[str, Any]]:
    payload = _json_object(path)
    _exact_keys(payload, CandidateBundle, "CandidateBundle")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported CandidateBundle schema_version")
    if payload["tool_plugin"] is not None:
        raise ValueError("real deployment does not admit tool_plugin candidates")
    for rule in payload["critic_rules"]:
        _exact_keys(rule, CriticRule, "critic rule")
        for predicate in rule["activation_conditions"]:
            _exact_keys(predicate, CriticPredicate, "critic activation")
    for recovery in payload["recovery_rules"]:
        _exact_keys(recovery, RecoveryRule, "recovery rule")
        for step in recovery["steps"]:
            _exact_keys(step, RecoveryStep, "recovery step")
    bundle = CandidateBundle.from_dict(payload)
    if bundle.sha256 != canonical_sha256(payload):
        raise ValueError("CandidateBundle normalization changes its digest")
    if not bundle.critic_rules or not bundle.recovery_rules:
        raise ValueError("real deployment requires critic and recovery rules")
    return bundle, payload


def _check_sources(contract: RealInputContract, live: LiveCapabilities) -> None:
    if contract.cameras != live.cameras:
        raise ValueError("live camera geometry, device, or calibration differs")
    if contract.depth_cameras != live.depth_cameras:
        raise ValueError("live aligned metric depth streams differ")
    if contract.joint_channels != live.joint_channels:
        raise ValueError("live 14D joint feedback mapping differs")
    if contract.auxiliary_channels != live.auxiliary_channels:
        raise ValueError("live auxiliary feedback mapping differs")
    if contract.tool_catalog_sha256 != live.tool_catalog_sha256:
        raise ValueError("live tool catalog differs")
    available = {feature.name: feature for feature in live.feature_sources}
    cameras = {camera.name for camera in contract.cameras}
    joints = set(contract.joint_channels)
    feedback = joints | set(contract.auxiliary_channels)
    depth = set(contract.depth_cameras)
    for feature in contract.feature_sources:
        if available.get(feature.name) != feature:
            raise ValueError(f"real provider does not attest feature: {feature.name}")
        ids = set(feature.source_ids)
        if feature.source_kind == "camera_rgb" and not ids <= cameras:
            raise ValueError(f"feature has no matching real camera source: {feature.name}")
        if feature.source_kind == "camera_rgbd" and not (
            ids <= cameras | depth and ids & cameras and ids & depth
        ):
            raise ValueError(f"RGBD feature needs RGB and metric depth: {feature.name}")
        if feature.source_kind == "joint_feedback" and not ids <= feedback:
            raise ValueError(f"feature has no matching joint feedback source: {feature.name}")
        if feature.source_kind == "fused" and not (
            ids <= cameras | feedback and ids & cameras and ids & feedback
        ):
            raise ValueError(f"fused feature needs camera and joint sources: {feature.name}")
        if feature.source_kind == "rgbd_fused" and not (
            ids <= cameras | depth | feedback and ids & cameras and ids & depth and ids & feedback
        ):
            raise ValueError(f"RGBD feature needs RGB, metric depth and joint sources: {feature.name}")


def _check_threshold(feature: RealFeatureSource, operator: str, value: Any) -> None:
    if operator not in ("lt", "le", "gt", "ge", "eq", "ne", "stagnant"):
        raise ValueError(f"unsupported critic operator: {operator}")
    kind = feature.scalar_type
    if kind == "number":
        valid = type(value) in (int, float) and math.isfinite(value)
    elif kind == "integer":
        valid = type(value) is int
    elif kind == "boolean":
        valid = type(value) is bool
    else:
        valid = type(value) is str
    if not valid:
        raise ValueError(f"threshold type differs from real feature: {feature.name}")
    if operator not in ("eq", "ne") and kind not in ("number", "integer"):
        raise ValueError(f"numeric operator requires numeric real feature: {feature.name}")
    if operator == "stagnant" and value < 0:
        raise ValueError("stagnant tolerance must be nonnegative")


def _check_rules(bundle: CandidateBundle, contract: RealInputContract) -> list[dict[str, Any]]:
    sources = {source.name: source for source in contract.feature_sources}
    used = set()
    for rule in bundle.critic_rules:
        if rule.dwell_steps > contract.max_critic_history_steps:
            raise ValueError(f"critic dwell exceeds real history budget: {rule.rule_id}")
        if rule.cooldown_steps > contract.max_critic_cooldown_steps:
            raise ValueError(f"critic cooldown exceeds real budget: {rule.rule_id}")
        for predicate in (rule, *rule.activation_conditions):
            feature = sources.get(predicate.feature)
            if feature is None:
                raise ValueError(f"critic feature has no real observation source: {predicate.feature}")
            _check_threshold(feature, predicate.operator, predicate.threshold)
            used.add(feature.name)
    return [source.model_dump(mode="json") for source in contract.feature_sources if source.name in used]


def _catalog_tools(catalog: dict[str, Any], expected_sha256: str) -> dict[str, dict[str, Any]]:
    if set(catalog) != {"schema_version", "catalog_sha256", "tools"}:
        raise ValueError("invalid frozen tool catalog fields")
    if catalog["schema_version"] != "arx.tool.catalog.v1":
        raise ValueError("unsupported tool catalog schema")
    tools = catalog["tools"]
    if not isinstance(tools, list) or catalog["catalog_sha256"] != digest(tools):
        raise ValueError("tool catalog digest mismatch")
    if catalog["catalog_sha256"] != expected_sha256:
        raise ValueError("tool catalog differs from real input contract")
    names = [entry.get("name") for entry in tools if isinstance(entry, dict)]
    if len(names) != len(tools) or len(set(names)) != len(names):
        raise ValueError("invalid or duplicate tool entry")
    return {entry["name"]: entry for entry in tools}


def _contains_reference(value: Any) -> bool:
    if isinstance(value, dict):
        return "$ref" in value or any(_contains_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_reference(item) for item in value)
    return False


def _check_recoveries(
    bundle: CandidateBundle, contract: RealInputContract, catalog: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate the bundle's declarative steps and report bounded tool expansion.

    CandidateBundle steps are plan parameters, not necessarily one gateway call.
    In the existing sample, a 2 cm request needs two calls under the gateway's
    1 cm per-call limit; the real executor must implement this expansion.
    """
    tools = _catalog_tools(catalog, contract.tool_catalog_sha256)
    plans = []
    for recovery in bundle.recovery_rules:
        if not recovery.fallback.strip():
            raise ValueError(f"recovery fallback text is required: {recovery.recovery_id}")
        calls = 0
        steps = []
        review_seen = False
        for step in recovery.steps:
            model = _TOOL_MODELS.get(step.tool)
            entry = tools.get(step.tool)
            if model is None or entry is None:
                raise ValueError(f"recovery tool is absent from real catalog: {step.tool}")
            schema = entry.get("input_schema")
            if (
                entry.get("version") != 1
                or not isinstance(schema, dict)
                or schema.get("type") != "object"
                or schema.get("additionalProperties") is not False
                or set(schema.get("properties", {})) != set(model.model_fields)
            ):
                raise ValueError(f"recovery tool schema differs from runtime: {step.tool}")
            if "RECOVERING" not in entry.get("allowed_states", []):
                raise ValueError(f"tool cannot run in recovery: {step.tool}")
            if not isinstance(step.parameters, dict):
                raise ValueError("recovery parameters must be an object")
            if _contains_reference(step.parameters):
                raise ValueError(f"unsupported recovery parameter reference: {step.tool}")
            arguments = deepcopy(step.parameters)
            if step.tool == "arx.review_reentry":
                # The sample bundle uses a symbolic observation produced after
                # recovery. The real executor must bind it to the live ID.
                if arguments.get("observation_ids") == ["post-recovery"]:
                    arguments["observation_ids"] = ["obs-preflight"]
                review_seen = True
            elif step.tool == "arx.zeva":
                token = arguments.get("reentry_token")
                if token not in (None, "token-from-review"):
                    raise ValueError("Zeva token must come from a fresh review")
                if not review_seen:
                    review_entry = tools.get("arx.review_reentry")
                    if review_entry is None or "RECOVERING" not in review_entry.get("allowed_states", []):
                        raise ValueError("automatic reentry review is absent from real catalog")
                    calls += 1
                    steps.append({"tool": "arx.review_reentry", "tool_calls": 1,
                                  "inserted_before": "arx.zeva"})
                    review_seen = True
                arguments["reentry_token"] = "token-preflight"
            call_count = 1
            if step.tool == "arx.move_eef":
                review_seen = False
                vector = arguments.get("delta_xyz_m")
                if isinstance(vector, list) and len(vector) == 3 and all(
                    type(value) in (int, float) and math.isfinite(value) for value in vector
                ):
                    norm = math.sqrt(sum(value * value for value in vector))
                    call_count = max(1, math.ceil((norm - 1e-12) / 0.01))
                    if call_count > contract.max_recovery_tool_calls:
                        raise ValueError("EEF movement exceeds recovery call budget")
                    arguments["delta_xyz_m"] = [value / call_count for value in vector]
            elif step.tool in ("arx.set_gripper", "arx.hold"):
                review_seen = False
            try:
                model.model_validate(arguments)
            except Exception as exc:
                raise ValueError(f"invalid recovery parameters for {step.tool}: {exc}") from exc
            calls += call_count
            steps.append({"tool": step.tool, "tool_calls": call_count})
        if calls > contract.max_recovery_tool_calls:
            raise ValueError(f"recovery exceeds tool-call budget: {recovery.recovery_id}")
        plans.append({"recovery_id": recovery.recovery_id, "tool_calls": calls, "steps": steps})
    return plans


def preflight_real_bundle(
    *,
    bundle_path: Path,
    task_manifest_path: Path,
    model_contract_path: Path,
    tool_catalog_path: Path,
    real_contract_path: Path,
    live_capabilities: LiveCapabilities,
    expected_real_contract_sha256: str,
) -> dict[str, Any]:
    """Verify frozen bytes against a trusted in-process hardware capability snapshot.

    This function performs no hardware motion. The future real runner must call it
    before opening action admission and retain the verified digests for the run.
    """
    paths = (
        bundle_path,
        task_manifest_path,
        model_contract_path,
        tool_catalog_path,
        real_contract_path,
    )
    (
        bundle_path,
        task_manifest_path,
        model_contract_path,
        tool_catalog_path,
        real_contract_path,
    ) = tuple(map(Path, paths))
    contract_digest = file_sha256(real_contract_path)
    if contract_digest != expected_real_contract_sha256:
        raise ValueError("real input contract SHA-256 mismatch")
    contract = RealInputContract.model_validate(_json_object(real_contract_path))
    if not isinstance(live_capabilities, LiveCapabilities):
        raise TypeError("live capabilities must come from a validated real adapter snapshot")
    if file_sha256(bundle_path) != contract.candidate_file_sha256:
        raise ValueError("CandidateBundle file SHA-256 mismatch")
    bundle, _ = _load_bundle(bundle_path)
    if bundle.sha256 != contract.candidate_sha256:
        raise ValueError("CandidateBundle semantic SHA-256 mismatch")
    if file_sha256(task_manifest_path) != contract.task_manifest_sha256:
        raise ValueError("task manifest SHA-256 mismatch")
    task = load_task_manifest(task_manifest_path)
    if (task.task_id, task.name) != (contract.task_id, contract.task_name):
        raise ValueError("task identity differs from real input contract")
    if file_sha256(model_contract_path) != contract.model_contract_sha256:
        raise ValueError("model contract SHA-256 mismatch")
    model = load_model_contract(model_contract_path)
    for actual, expected in zip(contract.cameras, model.cameras):
        actual_spec = (
            actual.name, actual.width, actual.height, actual.channels,
            actual.dtype, actual.color_order, actual.calibration_id,
        )
        expected_spec = (
            expected.name, expected.width, expected.height, expected.channels,
            expected.dtype, expected.color_order, expected.calibration_id,
        )
        if actual_spec != expected_spec:
            raise ValueError(f"real camera differs from model contract: {actual.name}")
    _check_sources(contract, live_capabilities)
    used_features = _check_rules(bundle, contract)
    catalog = _json_object(tool_catalog_path)
    recovery_plans = _check_recoveries(bundle, contract, catalog)
    return {
        "schema_version": "arx.real.preflight.v1",
        "eligible": True,
        "robot_id": live_capabilities.robot_id,
        "candidate_id": bundle.candidate_id,
        "candidate_sha256": bundle.sha256,
        "candidate_file_sha256": contract.candidate_file_sha256,
        "real_contract_sha256": contract_digest,
        "task_manifest_sha256": contract.task_manifest_sha256,
        "model_contract_sha256": contract.model_contract_sha256,
        "tool_catalog_sha256": contract.tool_catalog_sha256,
        "feature_sources": used_features,
        "recovery_plans": recovery_plans,
    }
