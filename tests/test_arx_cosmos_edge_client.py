# Copyright (c) 2026 Zetta Contributors
"""In-process protocol tests for the Zeva Task7 ZMQ service boundary."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zmq

from robots.arx.contracts import load_model_contract
from robots.arx.cosmos_edge_client import CosmosEdgeClient, packb, unpackb

_CONTRACT = load_model_contract(
    Path(__file__).resolve().parents[1] / "robots/arx/manifests/task7_model.yaml"
)
_MODALITY = {
    "action_semantic": "raw absolute X5 joint positions; gripper channels 6 and 13 are continuous raw positions",
    "camera_shape_hwc": [240, 320, 3],
    "resolution": "480",
    "domain_name": "arx-task7-x5",
    "domain_id": 17,
    "action_dim": 14,
    "action_horizon": 32,
    "view_mode": "concat_view",
    "action_normalization": "raw",
}


class _Server:
    def __init__(self, replies: list[Any]) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.replies = iter(replies)
        self.requests: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        for reply in self.replies:
            request = unpackb(self.socket.recv())
            self.requests.append(request)
            self.socket.send(reply if isinstance(reply, bytes) else packb(reply))

    def close(self) -> None:
        self.thread.join(timeout=2)
        self.socket.close(linger=0)
        self.context.term()


def _images() -> dict[str, np.ndarray]:
    return {name: np.zeros((240, 320, 3), np.uint8) for name in ("front_rgb", "left_rgb", "right_rgb")}


@pytest.mark.parametrize("seed", [None, 17, 2**31 - 1])
def test_exact_request_and_action_response(seed: int | None) -> None:
    actions = np.arange(32 * 14, dtype=np.float32).reshape(1, 32, 14)
    server = _Server([{"status": "ok"}, _MODALITY, ({"action": actions}, {"server": "fake"})])
    client = CosmosEdgeClient("127.0.0.1", server.port, _CONTRACT, timeout_sec=1)
    try:
        prediction = client.predict(_images(), np.zeros(14), "pick up the tube", seed=seed)
        assert prediction.actions.shape == (32, 14)
        assert prediction.metadata == {"server": "fake"}
        request = server.requests[-1]
        assert request["endpoint"] == "get_action"
        data = request["data"]
        assert data["options"] == {"executed_action_steps": 32, **({"seed": seed} if seed is not None else {})}
        assert data["prompt"] == ["pick up the tube"]
        assert data["observation"]["observation.state"].shape == (1, 14)
        for name in _images():
            assert data["observation"][f"observation.images.{name}"].shape == (1, 1, 240, 320, 3)
    finally:
        client.close()
        server.close()


def test_rejects_wrong_server_identity() -> None:
    modality = dict(_MODALITY, domain_id=3)
    server = _Server([{"status": "ok"}, modality])
    try:
        with pytest.raises(RuntimeError, match="domain_id mismatch"):
            CosmosEdgeClient("127.0.0.1", server.port, _CONTRACT, timeout_sec=1)
    finally:
        server.close()


def test_model_a_pins_server_normalization_separately_from_raw_actions() -> None:
    contract = load_model_contract(
        Path(__file__).resolve().parents[1] / "robots/arx/manifests/task7_model_a.yaml"
    )
    # The dodo Task7 server reports its internal fallback as minmax while
    # returning raw absolute joint positions (it has no action-stats file).
    modality = {
        "camera_shape_hwc": [240, 320, 3],
        "resolution": "480",
        "action_semantic": _MODALITY["action_semantic"],
        "action_normalization": "minmax",
    }
    server = _Server([{"status": "ok"}, modality])
    try:
        with CosmosEdgeClient("127.0.0.1", server.port, contract, timeout_sec=1):
            pass
    finally:
        server.close()

    server = _Server([{"status": "ok"}, dict(modality, action_normalization="meanstd")])
    try:
        with pytest.raises(RuntimeError, match="action_normalization mismatch"):
            CosmosEdgeClient("127.0.0.1", server.port, contract, timeout_sec=1)
    finally:
        server.close()


@pytest.mark.parametrize(
    "actions",
    [np.zeros((31, 14), np.float32), np.full((32, 14), np.nan, np.float32)],
)
def test_rejects_invalid_action_chunks(actions: np.ndarray) -> None:
    server = _Server([{"status": "ok"}, _MODALITY, ({"action": actions}, {})])
    client = CosmosEdgeClient("127.0.0.1", server.port, _CONTRACT, timeout_sec=1)
    try:
        with pytest.raises(RuntimeError, match="finite action shape"):
            client.predict(_images(), np.zeros(14), "task")
    finally:
        client.close()
        server.close()


def test_malformed_reply_recreates_socket_for_next_request() -> None:
    actions = np.zeros((32, 14), np.float32)
    server = _Server([{"status": "ok"}, _MODALITY, b"not-msgpack", ({"action": actions}, {})])
    client = CosmosEdgeClient("127.0.0.1", server.port, _CONTRACT, timeout_sec=1)
    first_socket = client.socket
    try:
        with pytest.raises(Exception):
            client.predict(_images(), np.zeros(14), "task")
        assert client.socket is not first_socket
        assert client.predict(_images(), np.zeros(14), "task").actions.shape == (32, 14)
    finally:
        client.close()
        server.close()


def test_input_contract_rejects_wrong_camera_before_network() -> None:
    server = _Server([{"status": "ok"}, _MODALITY])
    client = CosmosEdgeClient("127.0.0.1", server.port, _CONTRACT, timeout_sec=1)
    try:
        images = _images()
        images["front_rgb"] = np.zeros((120, 160, 3), np.uint8)
        with pytest.raises(ValueError, match="front_rgb"):
            client.predict(images, np.zeros(14), "task")
        assert len(server.requests) == 2
    finally:
        client.close()
        server.close()
