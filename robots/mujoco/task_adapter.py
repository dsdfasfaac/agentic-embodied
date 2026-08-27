# Copyright (c) 2026 Zetta Contributors
"""Agent-facing adapter for a complete reBot task through Rollout Runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine, TypeVar

import imageio.v2 as imageio

from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    Observation,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.payload_ref import InlineBytes, PayloadCodec
from rollout_runtime.api.result import unwrap
from rollout_runtime.backends.rebot_g1d_policy import (
    REBOT_G1D_SKILL_POLICY_FAMILY,
)
from rollout_runtime.backends.rebot_g1d_session import REBOT_G1D_TASK_SPECS
from rollout_runtime.core import payload as payload_module
from rollout_runtime.launch.local import LocalRuntime, build_local_components
from rollout_runtime.serve.client import RemoteRuntimeClient

__all__ = ["RebotAgenticTaskAdapter", "RebotTaskConfig"]

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RebotTaskConfig:
    """One fixed Agentic reBot experiment."""

    task: str
    asset_root: Path
    asset_manifest_sha256: str
    seed: int
    output_dir: Path
    video_path: Path
    camera_name: str
    image_width: int = 640
    image_height: int = 480
    chunk_size: int = 32
    max_episode_steps: int = 2500
    video_fps: float = 25.0
    video_stride: int = 4
    timeout_s: float = 900.0
    runtime_endpoint: str | None = None

    def __post_init__(self) -> None:
        if self.task not in REBOT_G1D_TASK_SPECS:
            raise ValueError(f"task must be one of {sorted(REBOT_G1D_TASK_SPECS)}")
        if not self.asset_root.is_absolute():
            raise ValueError("asset_root must be absolute")
        if len(self.asset_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.asset_manifest_sha256
        ):
            raise ValueError("asset_manifest_sha256 must be 64 lowercase hex")
        for name, value in (
            ("image_width", self.image_width),
            ("image_height", self.image_height),
            ("chunk_size", self.chunk_size),
            ("max_episode_steps", self.max_episode_steps),
            ("video_stride", self.video_stride),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.video_fps <= 0.0 or self.timeout_s <= 0.0:
            raise ValueError("video_fps and timeout_s must be positive")
        if self.runtime_endpoint is not None and not self.runtime_endpoint.strip():
            raise ValueError("runtime_endpoint must not be blank")


class _EventLoopThread:
    """Own the Runtime's asyncio objects for synchronous planner tool calls."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="rebot-agentic-runtime",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("reBot Runtime event loop did not start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def call(self, coroutine: Coroutine[Any, Any, _T], *, timeout: float) -> _T:
        loop = self._loop
        if loop is None or not self._thread.is_alive():
            coroutine.close()
            raise RuntimeError("reBot Runtime event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result(timeout=timeout)

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and self._thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=10.0)


class RebotAgenticTaskAdapter:
    """Run one planner-selected skill through Gateway, Policy, and MuJoCo."""

    def __init__(self, config: RebotTaskConfig) -> None:
        self.config = config
        self._loop = _EventLoopThread()
        self._runtime: LocalRuntime | None = None
        self._remote_client: RemoteRuntimeClient | None = None
        self._gateway: Any = None
        self._session_id: Any = None
        self._observation: Observation | None = None
        self._executed = False
        self._closed = False
        try:
            self._loop.call(self._initialize(), timeout=config.timeout_s)
        except BaseException:
            self._loop.stop()
            raise

    @property
    def policy_id(self) -> str:
        return f"rebot_{self.config.task}_skill"

    def _runtime_config(self) -> dict[str, Any]:
        env_config = self._env_config()
        task_spec = REBOT_G1D_TASK_SPECS[self.config.task]
        return {
            "env_family": "mujoco",
            "env_config": env_config,
            "transport": {"kind": "inproc"},
            "env_worker": {
                "num_ranks": 1,
                "max_sessions_per_rank": 1,
                "has_accelerator": False,
            },
            "rollout_worker": {
                "num_ranks": 1,
                "policy_id": self.policy_id,
                "policy_family": REBOT_G1D_SKILL_POLICY_FAMILY,
                "policy_backend": "rebot_g1d_skill",
                "policy_config": {
                    "asset_root": str(self.config.asset_root),
                    "asset_manifest_sha256": self.config.asset_manifest_sha256,
                    "task": self.config.task,
                    "scene_config": task_spec["scene_config"],
                    "seed": self.config.seed,
                    "policy_id": self.policy_id,
                },
                "device": "cpu",
                "dtype": "float32",
                "max_concurrent_inferences": 1,
                "scheduler": {
                    "max_batch_size": 1,
                    "max_wait_ms": 0.0,
                    "max_queue_depth": 8,
                    "max_inflight_per_application": 1,
                    "max_inflight_per_session": 1,
                },
            },
        }

    def _env_config(self) -> dict[str, Any]:
        task_spec = REBOT_G1D_TASK_SPECS[self.config.task]
        return {
            "provider": "rebot_g1d",
            "env_id": "reBot-DevArm-Grasp-v0",
            "asset_root": str(self.config.asset_root),
            "asset_manifest_sha256": self.config.asset_manifest_sha256,
            "rebot_scene_config": task_spec["scene_config"],
            "rebot_randomize_bottle": True,
            "rebot_task": self.config.task,
            "observation_mode": "rgb_state",
            "render_mode": "rgb_array",
            "render_backend": "egl",
            "camera_name": self.config.camera_name,
            "image_width": self.config.image_width,
            "image_height": self.config.image_height,
            "action_dim": 22,
            "chunk_size": self.config.chunk_size,
            "clip_actions": True,
            "max_episode_steps": self.config.max_episode_steps,
            "core_form": "per_slot",
            "process_isolation": True,
            "rpc_timeout_s": self.config.timeout_s,
            "instruction": self.instruction,
            "success_mode": "info_key",
            "success_info_key": "is_success",
            "return_all_frames": False,
        }

    @property
    def instruction(self) -> str:
        if self.config.task == "grasp":
            return "Grasp the upright water bottle and return it safely to ready."
        return "Pick up the fallen water bottle and deposit it inside the bin."

    async def _initialize(self) -> None:
        if self.config.runtime_endpoint is None:
            self._runtime = build_local_components(self._runtime_config())
            await self._runtime.start()
            self._gateway = self._runtime.gateway
        else:
            self._remote_client = RemoteRuntimeClient(
                self.config.runtime_endpoint,
                operation_timeout_s=self.config.timeout_s,
                session_timeout_s=self.config.timeout_s,
            )
            await self._remote_client.livez()
            self._gateway = self._remote_client
        env_spec = EnvSpecMsg(
            env_family="mujoco",
            env_config=self._env_config(),
            pool_size=1,
        )
        handle = unwrap(
            (
                await self._gateway.create_sessions(
                    [
                        CreateSessionRequest(
                            application_id="zetta-rebot-agent",
                            client_session_key=(
                                f"rebot-{self.config.task}-{self.config.seed}"
                            ),
                            env_spec=env_spec,
                            default_policy_id=self.policy_id,
                            lease_seconds=max(600.0, self.config.timeout_s),
                            metadata={
                                "agent_path": "planner->toolkit->gateway->policy->runtime"
                            },
                        )
                    ]
                )
            )[0]
        )
        self._session_id = handle.session_id
        reset = unwrap(
            (
                await self._gateway.reset(
                    [handle.session_id],
                    ResetSpec(seed=self.config.seed, instruction=self.instruction),
                )
            )[0]
        )
        if reset.observation is None:
            raise RuntimeError("reBot Runtime reset returned no observation")
        self._observation = reset.observation

    @staticmethod
    def _png_bytes(observation: Observation) -> bytes:
        payload = observation.main_image
        if (
            not isinstance(payload, InlineBytes)
            or payload.codec is not PayloadCodec.PNG
        ):
            raise RuntimeError("reBot observation has no inline PNG main image")
        return payload.data

    def observe(self) -> dict[str, Any]:
        observation = self._observation
        if observation is None:
            raise RuntimeError("reBot task has no initialized observation")
        return {
            "task": self.config.task,
            "instruction": self.instruction,
            "step_index": observation.step_index,
            "asset_manifest_sha256": self.config.asset_manifest_sha256,
            "policy_id": self.policy_id,
            "_image_bytes": self._png_bytes(observation),
        }

    def run(self) -> dict[str, Any]:
        if self._executed:
            raise RuntimeError("the configured reBot episode has already been executed")
        self._executed = True
        return self._loop.call(self._run_episode(), timeout=self.config.timeout_s)

    async def _run_episode(self) -> dict[str, Any]:
        runtime = self._runtime
        gateway = self._gateway
        observation = self._observation
        if gateway is None or self._session_id is None or observation is None:
            raise RuntimeError("reBot Runtime is not initialized")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            self.config.video_path,
            fps=self.config.video_fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        frame_count = 0
        last_recorded_step = -1

        def append_frame(item: Observation) -> None:
            nonlocal frame_count, last_recorded_step
            if item.main_image is None:
                raise RuntimeError("reBot task frame is missing its RGB image")
            writer.append_data(payload_module.decode_image(item.main_image))
            frame_count += 1
            last_recorded_step = item.step_index

        append_frame(observation)
        policy_steps = 0
        executed_horizon = 0
        final_step = None
        last_evaluation: dict[str, Any] = {}
        try:
            while executed_horizon < self.config.max_episode_steps:
                final_step = unwrap(
                    (
                        await gateway.policy_step(
                            [self._session_id],
                            PolicyRequest(
                                policy_id=self.policy_id,
                                actions_per_chunk=self.config.chunk_size,
                            ),
                        )
                    )[0]
                )
                policy_steps += 1
                executed_horizon += final_step.executed_horizon
                if final_step.per_step is None:
                    raise RuntimeError("reBot task did not return per-step audit data")
                recorded_per_step_frame = False
                for record in final_step.per_step:
                    if record.info:
                        last_evaluation = dict(record.info)
                    item = record.observation
                    if (
                        item is not None
                        and item.step_index % self.config.video_stride == 0
                    ):
                        append_frame(item)
                        recorded_per_step_frame = True
                self._observation = final_step.observation
                if not recorded_per_step_frame and final_step.observation is not None:
                    append_frame(final_step.observation)
                if final_step.terminated or final_step.truncated:
                    break
            if final_step is None or final_step.observation is None:
                raise RuntimeError("reBot task executed no policy steps")
            if final_step.observation.step_index != last_recorded_step:
                append_frame(final_step.observation)
        finally:
            writer.close()

        success = bool(final_step.success)
        if runtime is None:
            compiled_skill_audit: dict[str, Any] = {
                "availability": "server_side_not_exposed_by_remote_api"
            }
        else:
            compiled_skill_audit = dict(getattr(runtime.policies[0], "audit", {}))
        summary = {
            "schema_version": 1,
            "experiment": "agentic_rebot_mujoco",
            "task": self.config.task,
            "instruction": self.instruction,
            "seed": self.config.seed,
            "success": success,
            "terminated": bool(final_step.terminated),
            "truncated": bool(final_step.truncated),
            "policy_steps": policy_steps,
            "executed_horizon": executed_horizon,
            "final_step_index": final_step.observation.step_index,
            "asset_manifest_sha256": self.config.asset_manifest_sha256,
            "policy_backend": "rebot_g1d_skill",
            "policy_id": self.policy_id,
            "model_version": final_step.info.get("model_version"),
            "runtime_mode": (
                "remote" if self.config.runtime_endpoint is not None else "local"
            ),
            "execution_path": [
                "zetta.planner",
                "MujocoToolkit.run_rebot_skill",
                "RuntimeGateway.policy_step",
                "RuntimeRolloutWorker",
                "RebotG1DSkillPolicyCore",
                "MujocoEnvCore",
                "RebotG1DSession",
            ],
            "runtime_success_evaluation": last_evaluation,
            "compiled_skill_audit": compiled_skill_audit,
            "video": {
                "path": str(self.config.video_path),
                "fps": self.config.video_fps,
                "stride": self.config.video_stride,
                "frame_count": frame_count,
                "sha256": hashlib.sha256(
                    self.config.video_path.read_bytes()
                ).hexdigest(),
            },
        }
        summary_path = self.config.output_dir / "agentic-summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = {
            **summary,
            "summary_path": str(summary_path),
            "_image_bytes": self._png_bytes(final_step.observation),
        }
        if not success:
            result["error"] = "Runtime did not validate task success"
        return result

    async def _async_close(self) -> None:
        runtime = self._runtime
        remote_client = self._remote_client
        gateway = self._gateway
        if gateway is not None and self._session_id is not None:
            await gateway.close_sessions([self._session_id])
            self._session_id = None
        if runtime is not None:
            await runtime.gateway.stop()
            await runtime.aclose()
        if remote_client is not None:
            await remote_client.aclose()
        self._gateway = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.call(self._async_close(), timeout=30.0)
        finally:
            self._loop.stop()
