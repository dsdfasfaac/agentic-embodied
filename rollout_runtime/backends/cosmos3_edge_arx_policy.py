# Copyright (c) 2026 Zetta Contributors
"""Persistent Rollout Runtime backend for remote Zeva Task7 inference."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robots.arx.contracts import load_model_contract, load_task_manifest
from robots.arx.control import prepare_model_state
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.core import payload as payload_module

COSMOS3_EDGE_ARX_POLICY_FAMILY = "cosmos3_edge_arx"
_CAMERAS = ("front_rgb", "left_rgb", "right_rgb")

__all__ = [
    "COSMOS3_EDGE_ARX_POLICY_FAMILY",
    "Cosmos3EdgeArxPolicyConfig",
    "Cosmos3EdgeArxPolicyCore",
]


@dataclasses.dataclass(kw_only=True)
class Cosmos3EdgeArxPolicyConfig:
    host: str
    port: int = 5581
    model_contract: str
    task_manifest: str
    timeout_sec: float = 300.0
    policy_id: str = "zeva_arx_task7"
    policy_family: str = COSMOS3_EDGE_ARX_POLICY_FAMILY
    action_dim: int = 14
    actions_per_chunk: int = 32
    device: str = "remote"
    dtype: str = "float32"
    model_version: str = "cosmos3-edge-arx-task7-s4000"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Cosmos3EdgeArxPolicyConfig:
        payload = dict(value)
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown Cosmos3-Edge ARX policy config keys: {unknown}")
        try:
            result = cls(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid Cosmos3-Edge ARX policy config: {exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        if not self.host.strip() or not 1 <= self.port <= 65535 or self.timeout_sec <= 0:
            raise ValueError("Cosmos3-Edge endpoint configuration is invalid")
        for name in ("model_contract", "task_manifest"):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"{name} must be an existing absolute file")
        if self.policy_family != COSMOS3_EDGE_ARX_POLICY_FAMILY:
            raise ValueError(f"policy_family must be {COSMOS3_EDGE_ARX_POLICY_FAMILY!r}")
        if self.action_dim != 14 or self.actions_per_chunk != 32:
            raise ValueError("Task7 policy backend requires action_dim=14 and actions_per_chunk=32")
        if not self.policy_id.strip() or not self.model_version.strip():
            raise ValueError("policy_id and model_version must not be empty")


class Cosmos3EdgeArxPolicyCore:
    """Keep one ZMQ client resident and process Runtime requests serially."""

    def __init__(self, config: Cosmos3EdgeArxPolicyConfig) -> None:
        self.config = config
        self.contract = load_model_contract(config.model_contract)
        self.task = load_task_manifest(config.task_manifest)
        if self.task.execution_steps > config.actions_per_chunk:
            raise ValueError("task execution prefix exceeds policy horizon")
        self.client: Any = None
        self.loaded = False
        self.closed = False

    @property
    def model_version(self) -> str:
        return self.config.model_version

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def dtype(self) -> str:
        return self.config.dtype

    @property
    def policy_family(self) -> str:
        return self.config.policy_family

    def load(self) -> None:
        if self.loaded:
            return
        from robots.arx.cosmos_edge_client import CosmosEdgeClient

        self.client = CosmosEdgeClient(
            self.config.host,
            self.config.port,
            self.contract,
            self.config.timeout_sec,
        )
        self.loaded = True

    def update_weights(self, model_version: str) -> None:
        if model_version != self.model_version:
            raise ValueError("remote Cosmos3-Edge weights cannot be hot-swapped")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.loaded = False
        self.closed = True

    def _error(self, request: InferenceRequest, error: BaseException | str) -> ActionResponse:
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            model_version=self.model_version,
            error=make_error(
                ErrorCode.POLICY_FAILURE,
                f"Cosmos3-Edge ARX inference failed: {error}",
                policy_id=request.policy_id,
            ),
        )

    @staticmethod
    def _image_ref(observation: Any, slot: str) -> Any:
        if slot == "main_image":
            return observation.main_image
        if slot == "wrist_image":
            return observation.wrist_image
        prefix = "extra_view_images."
        if slot.startswith(prefix) and slot[len(prefix):].isdigit():
            index = int(slot[len(prefix):])
            return observation.extra_view_images[index] if index < len(observation.extra_view_images) else None
        raise ValueError(f"unknown camera payload slot: {slot!r}")

    def _infer_one(self, request: InferenceRequest) -> ActionResponse:
        try:
            if not self.loaded or self.client is None:
                raise RuntimeError("policy backend is not loaded")
            if request.policy_id != self.config.policy_id:
                raise ValueError(f"unexpected policy_id {request.policy_id!r}")
            names = request.observation.extras.get("camera_names")
            if not isinstance(names, dict) or set(names) != set(_CAMERAS):
                raise ValueError("observation camera_names must map all three Task7 views")
            images: dict[str, np.ndarray] = {}
            for name in _CAMERAS:
                ref = self._image_ref(request.observation, str(names[name]))
                if ref is None:
                    raise ValueError(f"observation is missing camera {name}")
                images[name] = payload_module.decode_image(ref)
            state = prepare_model_state(request.observation.state, self.task)
            instruction = (
                request.instruction_override
                if request.instruction_override is not None
                else request.observation.instruction
            )
            if instruction.strip() != self.task.instruction and request.instruction_override is None:
                raise ValueError("observation instruction does not match immutable task manifest")
            prediction = self.client.predict(images, state, instruction)
            actions = np.ascontiguousarray(
                prediction.actions[: self.task.execution_steps], dtype=np.float32
            )
        except BaseException as exc:  # normalize per request; do not poison batch peers
            return self._error(request, exc)
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            actions=payload_module.encode_array(actions),
            model_version=self.model_version,
            auxiliary_outputs={
                "raw_horizon": int(prediction.actions.shape[0]),
                "returned_horizon": int(actions.shape[0]),
                "roundtrip_sec": float(prediction.roundtrip_sec),
                "server_metadata": prediction.metadata,
                "modality_digest": self.client.modality_digest,
                "task": self.task.name,
            },
        )

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        return [self._infer_one(request) for request in requests]
