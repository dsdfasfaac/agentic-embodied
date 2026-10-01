# Copyright (c) 2026 Zetta Contributors
"""Runtime policy bridge tests without a Cosmos checkpoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robots.arx.cosmos_edge_client as client_module
from rollout_runtime.api.ids import EpisodeId, OperationSeq, RequestId, SessionId
from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends.cosmos3_edge_arx_policy import (
    Cosmos3EdgeArxPolicyConfig,
    Cosmos3EdgeArxPolicyCore,
)
from rollout_runtime.backends import build_policy_core
from rollout_runtime.core import payload as payload_module

_ROOT = Path(__file__).resolve().parents[2]
_MANIFESTS = _ROOT / "robots/arx/manifests"


class _FakeClient:
    instances: list[_FakeClient] = []

    def __init__(self, host, port, contract, timeout_sec) -> None:
        self.modality_digest = "d" * 64
        self.calls = []
        self.closed = False
        self.instances.append(self)

    def predict(self, images, state, prompt):
        self.calls.append((images, state.copy(), prompt))
        return SimpleNamespace(
            actions=np.arange(32 * 14, dtype=np.float32).reshape(32, 14),
            metadata={"request_id": "server-request"},
            roundtrip_sec=0.125,
        )

    def close(self) -> None:
        self.closed = True


def _core(monkeypatch: pytest.MonkeyPatch) -> Cosmos3EdgeArxPolicyCore:
    monkeypatch.setattr(client_module, "CosmosEdgeClient", _FakeClient)
    config = Cosmos3EdgeArxPolicyConfig.from_mapping({
        "host": "127.0.0.1",
        "port": 5581,
        "model_contract": str((_MANIFESTS / "task7_model.yaml").resolve()),
        "task_manifest": str((_MANIFESTS / "pickup_test_tube.yaml").resolve()),
    })
    return Cosmos3EdgeArxPolicyCore(config)


def _request(*, instruction_override: str | None = None) -> InferenceRequest:
    values = {
        "front_rgb": np.full((240, 320, 3), 1, np.uint8),
        "left_rgb": np.full((240, 320, 3), 2, np.uint8),
        "right_rgb": np.full((240, 320, 3), 3, np.uint8),
    }
    observation = Observation(
        session_id=SessionId("session"), episode_id=EpisodeId(1), step_index=0,
        main_image=payload_module.encode_image(values["right_rgb"]),
        wrist_image=payload_module.encode_image(values["front_rgb"]),
        extra_view_images=[payload_module.encode_image(values["left_rgb"])],
        state=[1.0] * 14,
        instruction="Pick up test tube with the pink label.",
        extras={"camera_names": {
            "right_rgb": "main_image",
            "front_rgb": "wrist_image",
            "left_rgb": "extra_view_images.0",
        }},
    )
    return InferenceRequest(
        request_id=RequestId("request"), session_id=SessionId("session"),
        episode_id=EpisodeId(1), operation_seq=OperationSeq(1),
        policy_id="zeva_arx_task7", observation=observation,
        instruction_override=instruction_override,
        routing_token="env:0", compat_key="key",
    )


def test_bridge_decodes_semantic_cameras_and_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeClient.instances.clear()
    core = _core(monkeypatch)
    core.load()
    response = core.infer_batch([_request()])[0]
    assert response.error is None
    actions = payload_module.decode_payload(response.actions)
    assert actions.shape == (16, 14)
    images, state, prompt = _FakeClient.instances[-1].calls[0]
    assert [int(images[name][0, 0, 0]) for name in ("front_rgb", "left_rgb", "right_rgb")] == [1, 2, 3]
    assert np.all(state[:7] == 0)  # task's model-only left-side override
    assert np.all(state[7:13] == 1)
    assert state[13] == pytest.approx(1.0 - 2.0 * np.pi)
    assert prompt == "Pick up test tube with the pink label."
    assert response.auxiliary_outputs["modality_digest"] == "d" * 64
    core.close()
    assert _FakeClient.instances[-1].closed


def test_instruction_override_and_per_request_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _core(monkeypatch)
    core.load()
    good = _request(instruction_override="alternate safe instruction")
    bad = InferenceRequest(**{**good.__dict__, "policy_id": "wrong"})
    responses = core.infer_batch([bad, good])
    assert responses[0].error is not None
    assert responses[1].error is None
    assert core.client.calls[-1][2] == "alternate safe instruction"
    with pytest.raises(ValueError, match="hot-swapped"):
        core.update_weights("new-version")
    core.close()


def test_backend_registry_builds_strict_arx_core() -> None:
    core = build_policy_core(
        backend="cosmos3_edge_arx_remote",
        policy_config={
            "host": "127.0.0.1",
            "port": 5581,
            "model_contract": str((_MANIFESTS / "task7_model.yaml").resolve()),
            "task_manifest": str((_MANIFESTS / "pickup_test_tube.yaml").resolve()),
        },
        device="remote",
        dtype="float32",
        policy_family="cosmos3_edge_arx",
        action_dim=14,
        actions_per_chunk=32,
    )
    assert isinstance(core, Cosmos3EdgeArxPolicyCore)
