# Copyright (c) 2026 Zetta Contributors
"""Validated configuration seam for an Agentic ARX Task7 episode."""

from __future__ import annotations

import dataclasses
import asyncio
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

from robots.arx.contracts import load_model_contract, load_task_manifest
from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg, Observation, PolicyRequest, ResetSpec
from rollout_runtime.api.payload_ref import InlineBytes, PayloadCodec
from rollout_runtime.api.result import unwrap
from rollout_runtime.launch.local import build_local_components
from rollout_runtime.serve.client import RemoteRuntimeClient

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ArxAgenticTaskConfig:
    prepared_scene_bundle: Path
    prepared_bundle_digest: str
    mapping_path: Path
    task_manifest: Path
    model_contract: Path
    camera_names: dict[str, str]
    output_dir: Path
    seed: int = 0
    zeva_host: str = "127.0.0.1"
    zeva_port: int = 5581
    timeout_s: float = 600.0
    render_backend: str = "egl"
    runtime_endpoint: str | None = None

    def __post_init__(self) -> None:
        for name in ("prepared_scene_bundle", "mapping_path", "task_manifest", "model_contract", "output_dir"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute Path")
        if not self.prepared_scene_bundle.is_dir():
            raise ValueError("prepared_scene_bundle must exist")
        for filename in ("model.mjb", "reset_state.npz", "metadata.json", "composition_report.json"):
            if not (self.prepared_scene_bundle / filename).is_file():
                raise ValueError(f"prepared scene is missing {filename}")
        for name in ("mapping_path", "task_manifest", "model_contract"):
            if not getattr(self, name).is_file():
                raise ValueError(f"{name} must exist")
        if not _SHA256.fullmatch(self.prepared_bundle_digest):
            raise ValueError("prepared_bundle_digest must be 64 lowercase hex")
        metadata = json.loads((self.prepared_scene_bundle / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("bundle_digest") != self.prepared_bundle_digest:
            raise ValueError("prepared scene bundle digest mismatch")
        task = load_task_manifest(self.task_manifest)
        contract = load_model_contract(self.model_contract)
        if not 1 <= task.execution_steps <= contract.action_horizon:
            raise ValueError("task execution_steps must be within the Runtime policy horizon")
        if tuple(self.camera_names) != tuple(camera.name for camera in contract.cameras):
            raise ValueError("camera_names must follow the model contract order")
        if len(set(self.camera_names.values())) != 3:
            raise ValueError("camera_names values must be unique")
        if not self.zeva_host.strip() or not 1 <= self.zeva_port <= 65535 or self.timeout_s <= 0:
            raise ValueError("Zeva endpoint/timeout is invalid")
        if self.render_backend not in {"egl", "glfw", "osmesa"}:
            raise ValueError("invalid MuJoCo render backend")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

    @property
    def task(self):
        return load_task_manifest(self.task_manifest)

    def runtime_config(self) -> dict[str, Any]:
        task = self.task
        return {
            "env_family": "mujoco",
            "env_config": {
                "provider": "arx_ac_one",
                "env_id": "ZettaArxManipulation-v0",
                "arx_prepared_scene_bundle": str(self.prepared_scene_bundle),
                "arx_mapping_path": str(self.mapping_path),
                "arx_task_manifest": str(self.task_manifest),
                "camera_names": dict(self.camera_names),
                "observation_mode": "rgb_state", "render_mode": "rgb_array",
                "render_backend": self.render_backend,
                "image_width": 320, "image_height": 240,
                "action_dim": 14, "chunk_size": 32, "clip_actions": False,
                "max_episode_steps": task.max_steps, "core_form": "per_slot",
                "process_isolation": True, "rpc_timeout_s": self.timeout_s,
                "instruction": task.instruction, "success_mode": "info_key",
                "success_info_key": "is_success", "return_all_frames": True,
            },
            "transport": {"kind": "inproc", "command_timeout_seconds": self.timeout_s},
            "env_worker": {"num_ranks": 1, "max_sessions_per_rank": 1, "has_accelerator": True},
            "rollout_worker": {
                "num_ranks": 1, "policy_id": "zeva_arx_task7",
                "policy_family": "cosmos3_edge_arx",
                "policy_backend": "cosmos3_edge_arx_remote",
                "device": "remote", "dtype": "float32", "max_concurrent_inferences": 1,
                "policy_config": {
                    "host": self.zeva_host, "port": self.zeva_port,
                    "timeout_sec": self.timeout_s,
                    "model_contract": str(self.model_contract),
                    "task_manifest": str(self.task_manifest),
                    "policy_id": "zeva_arx_task7", "policy_family": "cosmos3_edge_arx",
                    "action_dim": 14, "actions_per_chunk": 32,
                },
                "scheduler": {"max_batch_size": 1, "max_wait_ms": 0.0, "max_queue_depth": 8, "max_inflight_per_application": 1, "max_inflight_per_session": 1},
            },
        }

    def provenance(self) -> dict[str, Any]:
        metadata = json.loads((self.prepared_scene_bundle / "metadata.json").read_text(encoding="utf-8"))
        return {
            "prepared_bundle_digest": self.prepared_bundle_digest,
            "prepared_metadata_sha256": _file_hash(self.prepared_scene_bundle / "metadata.json"),
            "prepared_model_sha256": _file_hash(self.prepared_scene_bundle / "model.mjb"),
            "mapping_sha256": _file_hash(self.mapping_path),
            "task_manifest_sha256": _file_hash(self.task_manifest),
            "model_contract_sha256": _file_hash(self.model_contract),
            "robot_source_archive_sha256": metadata.get("robot_source_archive_sha256"),
        }


class _LoopThread:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="arx-agentic-runtime")
        self.thread.start()
        if not self.ready.wait(10):
            raise RuntimeError("ARX Runtime event loop did not start")

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()
        self.loop.close()

    def call(self, coroutine, timeout: float):
        if self.loop is None:
            coroutine.close()
            raise RuntimeError("ARX Runtime event loop is unavailable")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def close(self) -> None:
        if self.loop is not None and self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(10)


class ArxAgenticTaskAdapter:
    """One planner-visible episode whose motion can only use policy_step."""

    def __init__(self, config: ArxAgenticTaskConfig) -> None:
        self.config = config
        self._loop = _LoopThread()
        self._runtime = None
        self._remote = None
        self._gateway = None
        self._session_id = None
        self._observation: Observation | None = None
        self._observed = False
        self._executed = False
        self._closed = False
        self.runtime_success: bool | None = None
        try:
            self._loop.call(self._initialize(), config.timeout_s)
        except BaseException:
            self._loop.close()
            raise

    async def _initialize(self) -> None:
        if self.config.runtime_endpoint is None:
            self._runtime = build_local_components(self.config.runtime_config())
            await self._runtime.start()
            self._gateway = self._runtime.gateway
        else:
            self._remote = RemoteRuntimeClient(
                self.config.runtime_endpoint,
                operation_timeout_s=self.config.timeout_s,
                session_timeout_s=self.config.timeout_s,
            )
            await self._remote.livez()
            self._gateway = self._remote
        runtime = self.config.runtime_config()
        handle = unwrap((await self._gateway.create_sessions([CreateSessionRequest(
            application_id="zetta-arx-agent",
            client_session_key=f"arx-{self.config.task.name}-{self.config.seed}",
            env_spec=EnvSpecMsg(env_family="mujoco", env_config=runtime["env_config"], pool_size=1),
            default_policy_id="zeva_arx_task7",
            lease_seconds=max(600.0, self.config.timeout_s),
            metadata={"agent_path": "planner->toolkit->gateway->zeva->mujoco"},
        )]))[0])
        self._session_id = handle.session_id
        reset = unwrap((await self._gateway.reset([
            handle.session_id
        ], ResetSpec(seed=self.config.seed, instruction=self.config.task.instruction)))[0])
        if reset.observation is None:
            raise RuntimeError("ARX Runtime reset returned no observation")
        self._observation = reset.observation

    @staticmethod
    def _png(ref: Any, name: str) -> bytes:
        if not isinstance(ref, InlineBytes) or ref.codec is not PayloadCodec.PNG:
            raise RuntimeError(f"ARX observation has no inline PNG {name}")
        return ref.data

    def observe(self) -> dict[str, Any]:
        observation = self._observation
        if observation is None:
            raise RuntimeError("ARX episode has no observation")
        if not observation.extra_view_images:
            raise RuntimeError("ARX observation has no right camera")
        self._observed = True
        return {
            "task": self.config.task.name,
            "instruction": self.config.task.instruction,
            "step_index": observation.step_index,
            "prepared_bundle_digest": self.config.prepared_bundle_digest,
            "_image_bytes": self._png(observation.main_image, "front image"),
            "_image_wrist_bytes": self._png(observation.wrist_image, "left image"),
            "_image_cam_bytes": self._png(observation.extra_view_images[0], "right image"),
        }

    def run(self) -> dict[str, Any]:
        if not self._observed:
            raise RuntimeError("observe_arx_scene must be called before motion")
        if self._executed:
            raise RuntimeError("the ARX episode has already been executed")
        self._executed = True
        return self._loop.call(self._run_episode(), self.config.timeout_s)

    async def _run_episode(self) -> dict[str, Any]:
        if self._gateway is None or self._session_id is None:
            raise RuntimeError("ARX Runtime is not initialized")
        policy_calls = executed = 0
        terminal_reason = None
        final = None
        evaluations: list[dict[str, Any]] = []
        while executed < self.config.task.max_steps:
            final = unwrap((await self._gateway.policy_step([
                self._session_id
            ], PolicyRequest(policy_id="zeva_arx_task7", actions_per_chunk=32)))[0])
            policy_calls += 1
            executed += int(final.executed_horizon)
            for record in final.per_step or []:
                if record.info:
                    evaluations.append(dict(record.info))
                    terminal_reason = record.info.get("terminal_reason") or terminal_reason
            if final.observation is not None:
                self._observation = final.observation
            if final.terminated or final.truncated:
                break
        if final is None or not (final.terminated or final.truncated):
            raise RuntimeError("ARX Runtime reached the adapter limit without a terminal verdict")
        self.runtime_success = bool(final.success)
        summary = {
            "schema_version": "zetta_arx_episode_v1",
            "task": self.config.task.name,
            "instruction": self.config.task.instruction,
            "seed": self.config.seed,
            "success": self.runtime_success,
            "terminated": bool(final.terminated),
            "truncated": bool(final.truncated),
            "terminal_reason": terminal_reason,
            "policy_calls": policy_calls,
            "executed_horizon": executed,
            "final_step_index": final.observation.step_index if final.observation else None,
            "policy_backend": "cosmos3_edge_arx_remote",
            "provenance": self.config.provenance(),
            "last_evaluation": evaluations[-1] if evaluations else {},
            "runtime_mode": "remote" if self.config.runtime_endpoint else "local",
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.config.output_dir / "agentic-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        return {**summary, "summary_path": str(summary_path)}

    async def _async_close(self) -> None:
        if self._gateway is not None and self._session_id is not None:
            await self._gateway.close_sessions([self._session_id])
        if self._runtime is not None:
            await self._runtime.gateway.stop()
            await self._runtime.aclose()
        if self._remote is not None:
            await self._remote.aclose()
        self._session_id = None
        self._gateway = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call(self._async_close(), 30.0)
        finally:
            self._loop.close()
