# Copyright (c) 2026 Zetta Contributors
"""Strict synchronous client for the Zeva Cosmos3-Edge Task7 service."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import threading
import time
import uuid
from typing import Any

import msgpack
import numpy as np

from robots.arx.contracts import ARX_CAMERA_NAMES, ArxModelContract

__all__ = ["CosmosEdgeClient", "Prediction", "packb", "unpackb"]


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": stream.getvalue()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray_class__") is True:
        payload = value.get("as_npy")
        if not isinstance(payload, bytes):
            raise ValueError("invalid ndarray payload")
        return np.load(io.BytesIO(payload), allow_pickle=False)
    return value


def packb(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpackb(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_decode, raw=False)


@dataclasses.dataclass(frozen=True, slots=True)
class Prediction:
    actions: np.ndarray
    metadata: dict[str, Any]
    roundtrip_sec: float


class CosmosEdgeClient:
    """One serialized REQ client; failed exchanges recreate the socket."""

    def __init__(
        self,
        host: str,
        port: int,
        contract: ArxModelContract,
        timeout_sec: float = 300.0,
        *,
        connect_retries: int = 1,
    ) -> None:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyzmq is required for Cosmos3-Edge inference") from exc
        if not host.strip() or not 1 <= int(port) <= 65535:
            raise ValueError("host and port are invalid")
        if timeout_sec <= 0 or connect_retries < 1:
            raise ValueError("timeout_sec and connect_retries must be positive")
        self._zmq, self.contract = zmq, contract
        self.uri = f"tcp://{host}:{int(port)}"
        self.timeout_sec = float(timeout_sec)
        self._lock = threading.Lock()
        self._closed = False
        self.context = zmq.Context()
        self.socket = self._new_socket()
        last_error: Exception | None = None
        for _ in range(connect_retries):
            try:
                pong = self._request({"endpoint": "ping"})
                if not isinstance(pong, dict) or pong.get("status") != "ok":
                    raise RuntimeError(f"unexpected ping response: {pong!r}")
                break
            except Exception as exc:
                last_error = exc
        else:
            self.close()
            raise TimeoutError(f"failed to connect to {self.uri}: {last_error}")
        modality = self._request({"endpoint": "get_modality_config"})
        self.modality_digest = self._validate_modality(modality)
        self.metadata = {
            "endpoint": self.uri,
            "modality": modality,
            "modality_digest": self.modality_digest,
        }

    def _new_socket(self):
        socket = self.context.socket(self._zmq.REQ)
        timeout_ms = max(1, int(self.timeout_sec * 1000))
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.uri)
        return socket

    def _replace_socket(self) -> None:
        self.socket.close(linger=0)
        self.socket = self._new_socket()

    def _request(self, request: dict[str, Any]) -> Any:
        if self._closed:
            raise RuntimeError("Cosmos3-Edge client is closed")
        with self._lock:
            try:
                self.socket.send(packb(request))
                response = unpackb(self.socket.recv())
                if isinstance(response, dict) and "error" in response:
                    raise RuntimeError(f"Cosmos3-Edge server error: {response['error']}")
                return response
            except Exception:
                self._replace_socket()
                raise

    def _validate_modality(self, value: Any) -> str:
        if not isinstance(value, dict):
            raise RuntimeError("server modality config must be an object")
        required = {
            "camera_shape_hwc": [240, 320, 3],
            "resolution": "480",
        }
        for key, expected in required.items():
            if value.get(key) != expected:
                raise RuntimeError(f"server modality {key} mismatch: {value.get(key)!r} != {expected!r}")
        semantic = value.get("action_semantic")
        if not isinstance(semantic, str) or "raw absolute" not in semantic.lower():
            raise RuntimeError(
                "server modality action_semantic must describe raw absolute positions, "
                f"got {semantic!r}"
            )
        optional_identity = {
            "domain_name": self.contract.domain_name,
            "domain_id": self.contract.domain_id,
            "action_dim": self.contract.action_dim,
            "action_horizon": self.contract.action_horizon,
            "view_mode": self.contract.view_mode,
            "action_normalization": (
                self.contract.server_action_normalization
                or self.contract.action_normalization
            ),
        }
        for key, expected in optional_identity.items():
            if key in value and value[key] != expected:
                raise RuntimeError(f"server modality {key} mismatch: {value[key]!r} != {expected!r}")
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def predict(
        self, images: dict[str, np.ndarray], state: np.ndarray, prompt: str,
        *, seed: int | None = None,
    ) -> Prediction:
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
            raise ValueError("seed must be an integer in [0, 2**31)")
        if set(images) != set(ARX_CAMERA_NAMES):
            raise ValueError(f"images must contain exactly {ARX_CAMERA_NAMES}")
        model_images: dict[str, np.ndarray] = {}
        for camera in self.contract.cameras:
            image = np.asarray(images[camera.name])
            expected = (camera.height, camera.width, camera.channels)
            if image.dtype != np.uint8 or image.shape != expected:
                raise ValueError(f"{camera.name} must be uint8 {expected}, got {image.dtype} {image.shape}")
            model_images[camera.name] = np.ascontiguousarray(image)
        model_state = np.ascontiguousarray(state, dtype=np.float32)
        if model_state.shape != (14,) or not np.isfinite(model_state).all():
            raise ValueError("model state must be finite [14]")
        instruction = prompt.strip()
        if not instruction:
            raise ValueError("task instruction must not be empty")
        now_ns = time.time_ns()
        request = {
            "endpoint": "get_action",
            "data": {
                "schema_version": 1,
                "request_id": str(uuid.uuid4()),
                "observation": {
                    **{
                        f"observation.images.{name}": model_images[name][None, None]
                        for name in ARX_CAMERA_NAMES
                    },
                    "observation.state": model_state[None],
                    "observation.timestamps_ns": np.asarray(
                        [[now_ns] * 4], dtype=np.int64
                    ),
                },
                "prompt": [instruction],
                "options": {"executed_action_steps": self.contract.action_horizon},
            },
        }
        started = time.perf_counter()
        if seed is not None:
            request["data"]["options"]["seed"] = seed
        response = self._request(request)
        elapsed = time.perf_counter() - started
        if not isinstance(response, (tuple, list)) or len(response) != 2:
            raise RuntimeError("unexpected get_action response envelope")
        action_dict, metadata = response
        if not isinstance(action_dict, dict) or "action" not in action_dict:
            raise RuntimeError("get_action response is missing action")
        actions = np.ascontiguousarray(action_dict["action"], dtype=np.float32)
        if actions.shape == (1, 32, 14):
            actions = actions[0]
        if actions.shape != (32, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f"expected finite action shape (32,14), got {actions.shape}")
        return Prediction(
            actions,
            metadata if isinstance(metadata, dict) else {"raw": metadata},
            elapsed,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.socket.close(linger=0)
        self.context.term()

    def __enter__(self) -> CosmosEdgeClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
