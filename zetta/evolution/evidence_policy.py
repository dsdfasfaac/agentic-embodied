"""Visibility contracts for campaign evidence.

The ARX policy is deliberately fail-closed: private simulator fields are
rejected at publication time instead of being silently removed from prompts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_PRIVATE = re.compile(
    r"(?:qpos|qvel|contact|pose|target|reward|residual|seed|rng|capabilit|"
    r"gateway|evaluator|privileged|simulator|model_path|reset_state|scene_path)",
    re.IGNORECASE,
)
_PUBLIC_FIELDS = frozenset({
    "schema_version", "event_id", "type", "kind", "step", "step_index",
    "action_index", "timestamp", "observation_id", "rgb_frame_id",
    "frame_id", "camera", "content_id", "sha256", "media_type", "width",
    "height", "action_ack", "accepted", "status", "tool", "tool_name",
    "request_id", "operation_id", "write_certainty", "control_epoch",
    "features", "value", "available", "reason", "terminal_reason",
    "authoritative_success", "success", "remaining_steps", "remaining_decisions",
    "remaining_recoveries", "public_history", "events",
})


class EvidencePolicyError(ValueError):
    pass


class EvidencePolicy(Protocol):
    policy_id: str

    def validate_public_event(self, event: Mapping[str, Any]) -> None: ...

    def validate_candidate_feature(self, name: str) -> None: ...

    def publish_event(self, event: Mapping[str, Any]) -> dict[str, Any]: ...


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            current = f"{path}.{name}" if path else name
            yield current, name
            yield from _walk(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ArxRgbPublicEvidencePolicy:
    policy_id: str = "arx_rgb_public_v1"

    def _check(self, value: Any) -> None:
        for path, name in _walk(value):
            if _PRIVATE.search(name):
                raise EvidencePolicyError(f"private ARX evidence field: {path}")

    def reject_private_fields(self, value: Any) -> None:
        """Check an upstream public event before projecting its public ABI."""
        self._check(value)

    def validate_public_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise EvidencePolicyError("public event must be an object")
        self._check(event)
        unknown = {name for _, name in _walk(event) if name not in _PUBLIC_FIELDS}
        if unknown:
            raise EvidencePolicyError(f"unknown ARX public fields: {sorted(unknown)}")

    def validate_candidate_feature(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise EvidencePolicyError("candidate feature name must be non-empty")
        if _PRIVATE.search(name):
            raise EvidencePolicyError(f"private ARX feature: {name}")

    def publish_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_public_event(event)
        return json.loads(json.dumps(dict(event), sort_keys=True))

    def publish_jsonl(self, source: Path, destination: Path) -> int:
        count = 0
        with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                event = json.loads(line)
                dst.write(json.dumps(self.publish_event(event), sort_keys=True) + "\n")
                count += 1
        return count


@dataclass(frozen=True, slots=True)
class PermissiveEvidencePolicy:
    policy_id: str

    def validate_public_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise EvidencePolicyError("event must be an object")

    def validate_candidate_feature(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise EvidencePolicyError("feature name must be non-empty")

    def publish_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_public_event(event)
        return dict(event)


@dataclass(frozen=True, slots=True)
class ArxPrivilegedEvidencePolicy(PermissiveEvidencePolicy):
    """Explicit simulator projection policy; arbitrary private fields remain invalid."""

    policy_id: str = "arx_privileged_v1"

    def validate_candidate_feature(self, name: str) -> None:
        if not isinstance(name, str) or not name.startswith("privileged."):
            raise EvidencePolicyError("privileged ARX features must use privileged.*")
        if not name.strip() or name.count(".") < 2:
            raise EvidencePolicyError("invalid privileged ARX feature")


def get_evidence_policy(policy_id: str) -> EvidencePolicy:
    if policy_id == "arx_rgb_public_v1":
        return ArxRgbPublicEvidencePolicy()
    if policy_id in {"libero_privileged_critic_v1", "robocasa_named_action_v1"}:
        return PermissiveEvidencePolicy(policy_id)
    if policy_id == "arx_privileged_v1":
        return ArxPrivilegedEvidencePolicy()
    raise EvidencePolicyError(f"unknown evidence policy: {policy_id}")
