# Copyright (c) 2026 Zetta Contributors
"""First-class Gymnasium MuJoCo environment backend.

The backend implements the Runtime's synchronous ``EnvExecutionCore`` contract
without importing Gymnasium or MuJoCo until a concrete environment is built.
It deliberately starts with independent ``per_slot`` environments: Gymnasium
MuJoCo has no universal vector-state, task-success, or action-batching contract.
"""

from __future__ import annotations

import dataclasses
import math
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import EpisodeId, SessionId
from rollout_runtime.api.messages import (
    EnvFamilyCapability,
    EnvSpecMsg,
    Observation,
    ResetSpec,
)
from rollout_runtime.backends.mujoco_session import (
    MujocoSessionError,
    create_mujoco_session,
    spawn_mujoco_session,
)
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import (
    PER_SLOT_FORM,
    ChunkOutcome,
    EnvFamilyBehavior,
    normalize_chunk_outcome,
)
from rollout_runtime.core.env_registry import (
    MUJOCO_ENV_FAMILY,
    behavior_for,
    capability_from_behavior,
    register_env_family,
    requested_core_form,
)

__all__ = [
    "MujocoEnvConfig",
    "MujocoEnvCore",
    "MujocoEnvFamily",
    "mujoco_env_capability",
    "register_mujoco_env_family",
]

_OBSERVATION_MODES = frozenset({"state", "rgb", "rgb_state"})
_SUCCESS_MODES = frozenset({"none", "info_key", "return_threshold"})
_RENDER_BACKENDS = frozenset({"egl", "glfw", "osmesa"})
_PROVIDERS = frozenset({"gymnasium", "rebot_g1d", "arx_ac_one"})
_REBOT_TASKS = frozenset({"grasp", "fallen"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REBOT_ACTION_DIM = 22


@dataclasses.dataclass(kw_only=True)
class MujocoEnvConfig:
    """Validated family-private configuration for supported MuJoCo providers."""

    provider: str = "gymnasium"
    env_id: str = "InvertedPendulum-v5"
    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    asset_root: str | None = None
    asset_manifest_sha256: str | None = None
    rebot_scene_config: str | None = None
    rebot_randomize_bottle: bool = True
    rebot_task: str | None = None
    arx_prepared_scene_bundle: str | None = None
    arx_mapping_path: str | None = None
    arx_task_manifest: str | None = None
    camera_names: dict[str, str] | None = None
    observation_mode: str = "state"
    render_mode: str | None = None
    render_backend: str = "egl"
    camera_name: str | None = None
    image_width: int = 256
    image_height: int = 256
    action_dim: int = 1
    chunk_size: int = 4
    clip_actions: bool = True
    max_episode_steps: int | None = None
    core_form: str = PER_SLOT_FORM
    process_isolation: bool | None = None
    rpc_timeout_s: float = 120.0
    instruction: str = ""
    success_mode: str = "none"
    success_info_key: str = "is_success"
    success_return_threshold: float | None = None
    return_all_frames: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> MujocoEnvConfig:
        """Construct and validate a config, rejecting unknown keys."""
        values = dict(config or {})
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown mujoco env config keys: {unknown}",
                    unknown_keys=unknown,
                    known_keys=sorted(known),
                )
            )
        result = cls(**values)
        result._validate()
        return result

    def _validate(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in _PROVIDERS:
            raise self._invalid(f"provider must be one of {sorted(_PROVIDERS)}")
        if not isinstance(self.env_id, str) or not self.env_id.strip():
            raise self._invalid("env_id must be a non-empty string")
        if not isinstance(self.env_kwargs, dict):
            raise self._invalid("env_kwargs must be an object")
        if self.provider == "gymnasium":
            if self.asset_root is not None or self.asset_manifest_sha256 is not None:
                raise self._invalid(
                    "asset_root and asset_manifest_sha256 require provider='rebot_g1d'"
                )
            if self.rebot_scene_config is not None:
                raise self._invalid("rebot_scene_config requires provider='rebot_g1d'")
            if self.rebot_task is not None:
                raise self._invalid("rebot_task requires provider='rebot_g1d'")
            if any((self.arx_prepared_scene_bundle, self.arx_mapping_path, self.arx_task_manifest, self.camera_names)):
                raise self._invalid("ARX fields require provider='arx_ac_one'")
        elif self.provider == "rebot_g1d":
            if self.env_id != "reBot-DevArm-Grasp-v0":
                raise self._invalid(
                    "provider='rebot_g1d' requires env_id='reBot-DevArm-Grasp-v0'"
                )
            if not isinstance(self.asset_root, str) or not self.asset_root:
                raise self._invalid(
                    "provider='rebot_g1d' requires a non-empty asset_root"
                )
            if not Path(self.asset_root).expanduser().is_absolute():
                raise self._invalid("reBot asset_root must be an absolute path")
            if not isinstance(self.asset_manifest_sha256, str) or not (
                _SHA256_PATTERN.fullmatch(self.asset_manifest_sha256)
            ):
                raise self._invalid(
                    "reBot asset_manifest_sha256 must be 64 lowercase hex characters"
                )
            if self.rebot_scene_config is not None and (
                not isinstance(self.rebot_scene_config, str)
                or not self.rebot_scene_config
            ):
                raise self._invalid(
                    "rebot_scene_config must be a non-empty string or null"
                )
            scene_config = self.rebot_scene_config or "config/g1d_mujoco.yaml"
            scene_path = PurePosixPath(scene_config)
            if (
                scene_path.is_absolute()
                or ".." in scene_path.parts
                or "\\" in scene_config
                or scene_path.suffix not in {".yaml", ".yml"}
            ):
                raise self._invalid(
                    "rebot_scene_config must be a relative YAML path under asset_root"
                )
            self.rebot_scene_config = scene_config
            if self.rebot_task is not None:
                if (
                    not isinstance(self.rebot_task, str)
                    or self.rebot_task not in _REBOT_TASKS
                ):
                    raise self._invalid(
                        f"rebot_task must be one of {sorted(_REBOT_TASKS)} or null"
                    )
                expected_scene = {
                    "grasp": "config/g1d_mujoco.yaml",
                    "fallen": "config/g1d_fallen_bottle_to_bin.yaml",
                }[self.rebot_task]
                if scene_config != expected_scene:
                    raise self._invalid(
                        f"rebot_task={self.rebot_task!r} requires "
                        f"rebot_scene_config={expected_scene!r}"
                    )
            if self.env_kwargs:
                raise self._invalid(
                    "env_kwargs is only supported by provider='gymnasium'"
                )
            if self.action_dim != _REBOT_ACTION_DIM:
                raise self._invalid(
                    f"provider='rebot_g1d' requires action_dim={_REBOT_ACTION_DIM}"
                )
            if self.process_isolation is False:
                raise self._invalid(
                    "provider='rebot_g1d' requires process_isolation=true"
                )
            if self.rebot_task is not None and (
                self.success_mode != "info_key" or self.success_info_key != "is_success"
            ):
                raise self._invalid(
                    "reBot tasks require success_mode='info_key' and "
                    "success_info_key='is_success'"
                )
            if any((self.arx_prepared_scene_bundle, self.arx_mapping_path, self.arx_task_manifest, self.camera_names)):
                raise self._invalid("ARX fields require provider='arx_ac_one'")
        else:
            for name in ("arx_prepared_scene_bundle", "arx_mapping_path", "arx_task_manifest"):
                value = getattr(self, name)
                if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
                    raise self._invalid(f"provider='arx_ac_one' requires absolute {name}")
            if not isinstance(self.camera_names, dict) or tuple(self.camera_names) != (
                "front_rgb", "left_rgb", "right_rgb"
            ):
                raise self._invalid("ARX camera_names must contain ordered front_rgb,left_rgb,right_rgb")
            if len(set(self.camera_names.values())) != 3 or any(not str(value).strip() for value in self.camera_names.values()):
                raise self._invalid("ARX MuJoCo camera names must be unique and non-empty")
            if self.env_kwargs or self.asset_root is not None or self.rebot_task is not None:
                raise self._invalid("ARX provider rejects Gymnasium/reBot-specific fields")
            if self.action_dim != 14 or self.chunk_size != 32:
                raise self._invalid("ARX provider requires action_dim=14 and chunk_size=32")
            if (self.image_width, self.image_height) != (320, 240):
                raise self._invalid("ARX provider requires 320x240 camera images")
            if self.observation_mode != "rgb_state" or self.render_mode != "rgb_array":
                raise self._invalid("ARX provider requires observation_mode='rgb_state' and render_mode='rgb_array'")
            if self.success_mode != "info_key" or self.success_info_key != "is_success":
                raise self._invalid("ARX provider requires environment-owned is_success")
        if not isinstance(self.rebot_randomize_bottle, bool):
            raise self._invalid("rebot_randomize_bottle must be a boolean")
        if (
            not isinstance(self.observation_mode, str)
            or self.observation_mode not in _OBSERVATION_MODES
        ):
            raise self._invalid(
                f"observation_mode must be one of {sorted(_OBSERVATION_MODES)}"
            )
        if self.observation_mode in {"rgb", "rgb_state"}:
            if self.render_mode is None:
                self.render_mode = "rgb_array"
            if self.render_mode != "rgb_array":
                raise self._invalid(
                    "RGB observation modes require render_mode='rgb_array'"
                )
        elif self.render_mode is not None:
            raise self._invalid("state observation mode requires render_mode=null")
        if self.render_mode is not None and not isinstance(self.render_mode, str):
            raise self._invalid("render_mode must be a string or null")
        if (
            not isinstance(self.render_backend, str)
            or self.render_backend not in _RENDER_BACKENDS
        ):
            raise self._invalid(
                f"render_backend must be one of {sorted(_RENDER_BACKENDS)}"
            )
        if self.camera_name is not None and (
            not isinstance(self.camera_name, str) or not self.camera_name.strip()
        ):
            raise self._invalid("camera_name must be a non-empty string or null")
        if not self.uses_rgb and self.camera_name is not None:
            raise self._invalid("camera_name is only valid for RGB observation modes")
        for name, value in (
            ("image_width", self.image_width),
            ("image_height", self.image_height),
            ("action_dim", self.action_dim),
            ("chunk_size", self.chunk_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise self._invalid(f"{name} must be a positive integer")
        if self.max_episode_steps is not None and (
            not isinstance(self.max_episode_steps, int)
            or isinstance(self.max_episode_steps, bool)
            or self.max_episode_steps < 1
        ):
            raise self._invalid("max_episode_steps must be a positive integer or null")
        if not isinstance(self.clip_actions, bool):
            raise self._invalid("clip_actions must be a boolean")
        if not isinstance(self.core_form, str) or self.core_form != PER_SLOT_FORM:
            raise self._invalid("mujoco v1 only supports core_form='per_slot'")
        if self.process_isolation is not None and not isinstance(
            self.process_isolation, bool
        ):
            raise self._invalid("process_isolation must be a boolean or null")
        if (
            isinstance(self.rpc_timeout_s, bool)
            or not isinstance(self.rpc_timeout_s, (int, float))
            or not math.isfinite(float(self.rpc_timeout_s))
            or self.rpc_timeout_s <= 0
        ):
            raise self._invalid("rpc_timeout_s must be finite and positive")
        if not isinstance(self.instruction, str):
            raise self._invalid("instruction must be a string")
        if (
            not isinstance(self.success_mode, str)
            or self.success_mode not in _SUCCESS_MODES
        ):
            raise self._invalid(f"success_mode must be one of {sorted(_SUCCESS_MODES)}")
        if not isinstance(self.success_info_key, str) or not self.success_info_key:
            raise self._invalid("success_info_key must be a non-empty string")
        if self.success_mode == "return_threshold":
            threshold = self.success_return_threshold
            if (
                threshold is None
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
            ):
                raise self._invalid(
                    "return_threshold mode requires a finite success_return_threshold"
                )
        elif self.success_return_threshold is not None:
            raise self._invalid(
                "success_return_threshold is only valid with "
                "success_mode='return_threshold'"
            )
        if not isinstance(self.return_all_frames, bool):
            raise self._invalid("return_all_frames must be a boolean")

    @staticmethod
    def _invalid(message: str) -> RuntimeApiError:
        return RuntimeApiError(make_error(ErrorCode.INVALID_ARGUMENT, message))

    @property
    def uses_rgb(self) -> bool:
        return self.observation_mode in {"rgb", "rgb_state"}

    @property
    def resolved_process_isolation(self) -> bool:
        if self.provider in {"rebot_g1d", "arx_ac_one"}:
            return True
        if self.process_isolation is not None:
            return self.process_isolation
        return self.uses_rgb

    def session_config(self) -> dict[str, Any]:
        """Project fields used by the in-process or child session."""
        return {
            "provider": self.provider,
            "env_id": self.env_id,
            "env_kwargs": dict(self.env_kwargs),
            "asset_root": self.asset_root,
            "asset_manifest_sha256": self.asset_manifest_sha256,
            "rebot_scene_config": self.rebot_scene_config,
            "rebot_randomize_bottle": self.rebot_randomize_bottle,
            "rebot_task": self.rebot_task,
            "arx_prepared_scene_bundle": self.arx_prepared_scene_bundle,
            "arx_mapping_path": self.arx_mapping_path,
            "arx_task_manifest": self.arx_task_manifest,
            "camera_names": dict(self.camera_names or {}),
            "render_mode": self.render_mode,
            "render_backend": self.render_backend,
            "camera_name": self.camera_name,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "max_episode_steps": self.max_episode_steps,
            "rpc_timeout_s": float(self.rpc_timeout_s),
        }


@dataclasses.dataclass
class _MujocoSlot:
    session: Any
    default_seed: int
    instruction: str = ""
    seed: int | None = None
    step_index: int = 0
    episode_return: float = 0.0
    terminated: bool = False
    truncated: bool = False
    success: bool | None = None
    started: bool = False
    raw_observation: Any = None
    last_observation: Observation | None = None
    observation_layout: tuple[tuple[str, tuple[int, ...]], ...] = ()


def _layout_and_state(
    value: Any,
) -> tuple[list[float], tuple[tuple[str, tuple[int, ...]], ...]]:
    values: list[float] = []
    layout: list[tuple[str, tuple[int, ...]]] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda entry: str(entry)):
                child = f"{path}.{key}" if path else str(key)
                visit(item[key], child)
            return
        array = np.asarray(item)
        if array.dtype.kind not in "biuf":
            raise ValueError(
                f"observation field {path or '<root>'!r} is not numeric "
                f"(dtype={array.dtype})"
            )
        flat = np.asarray(array, dtype=np.float32).reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError(
                f"observation field {path or '<root>'!r} contains NaN or Inf"
            )
        layout.append((path or "state", tuple(int(dim) for dim in array.shape)))
        values.extend(float(item) for item in flat)

    visit(value, "")
    return values, tuple(layout)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


class MujocoEnvCore:
    """Independent-slot execution core for supported MuJoCo providers."""

    def __init__(self) -> None:
        self.config = MujocoEnvConfig()
        self.env_spec: EnvSpecMsg | None = None
        self.seed_offset = 0
        self.closed = False
        self._core_form = PER_SLOT_FORM
        self._slots: list[_MujocoSlot] = []
        self._slot_mutation_lock = threading.Lock()
        self._expected_layout: tuple[tuple[str, tuple[int, ...]], ...] | None = None

    @property
    def behavior(self) -> EnvFamilyBehavior:
        return behavior_for(MUJOCO_ENV_FAMILY)

    @property
    def core_form(self) -> str:
        return self._core_form

    def build(
        self,
        env_spec: EnvSpecMsg,
        *,
        num_envs: int,
        seed_offset: int = 0,
        total_num_processes: int = 1,
    ) -> None:
        if num_envs < 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"num_envs must be >= 1, got {num_envs}",
                )
            )
        if self._slots:
            self.close()
        self._expected_layout = None
        self.config = MujocoEnvConfig.from_mapping(env_spec.env_config)
        self._core_form = requested_core_form(env_spec, self.behavior)
        self.env_spec = env_spec
        self.seed_offset = int(seed_offset)
        self.closed = False
        del total_num_processes
        try:
            for index in range(num_envs):
                self._append_slot(self.seed_offset + index)
        except BaseException:
            self.close()
            raise

    def _new_session(self) -> Any:
        config = self.config.session_config()
        try:
            if self.config.resolved_process_isolation:
                return spawn_mujoco_session(config)
            return create_mujoco_session(config)
        except MujocoSessionError as exc:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    str(exc),
                    env_family=MUJOCO_ENV_FAMILY,
                    env_id=self.config.env_id,
                )
            ) from exc

    def _append_slot(self, default_seed: int) -> int:
        session = self._new_session()
        shape = tuple(int(item) for item in session.descriptor["action_shape"])
        if shape != (int(self.config.action_dim),):
            session.close()
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"configured action_dim={self.config.action_dim} does not match "
                    f"{self.config.env_id!r} action shape {shape}",
                    configured_action_dim=int(self.config.action_dim),
                    action_shape=shape,
                )
            )
        index = len(self._slots)
        slot = _MujocoSlot(
            session=session,
            default_seed=int(default_seed),
            instruction=self.config.instruction,
        )
        self._slots.append(slot)
        try:
            # A cold validation reset catches missing MuJoCo assets, an invalid
            # camera, and EGL initialization before the pool is advertised.
            raw, _info = session.reset(seed=slot.default_seed, options={})
            frame = session.render() if self.config.uses_rgb else None
            preview = self._make_observation(index, raw, frame)
            slot.observation_layout = self._layout_of(preview)
            self._check_layout(slot.observation_layout)
            slot.raw_observation = None
            slot.last_observation = None
            slot.started = False
        except MujocoSessionError as exc:
            self._slots.pop()
            session.close()
            raise self._session_failure(exc) from exc
        except BaseException:
            self._slots.pop()
            session.close()
            raise
        return index

    def slot_count(self) -> int:
        return len(self._slots)

    def add_slot(self, seed_offset: int) -> int:
        with self._slot_mutation_lock:
            if self.closed:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY,
                        "cannot add a slot to a closed mujoco core",
                    )
                )
            return self._append_slot(int(seed_offset))

    def remove_slot(self, slot_index: int) -> None:
        with self._slot_mutation_lock:
            last = len(self._slots) - 1
            if slot_index < 0 or slot_index != last:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        f"can only remove trailing mujoco slot {last}, got {slot_index}",
                    )
                )
            slot = self._slots.pop()
            slot.session.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        slots, self._slots = self._slots, []
        for slot in slots:
            try:
                slot.session.close()
            except BaseException:
                pass

    def reset(self, slots: Sequence[int], reset_spec: ResetSpec) -> list[Observation]:
        if reset_spec.task_id is not None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "generic mujoco pools freeze env_id in env_config; task_id is unsupported",
                )
            )
        if reset_spec.reset_state_id is not None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "generic mujoco does not support reset_state_id",
                )
            )
        observations: list[Observation] = []
        for slot_index in slots:
            slot = self._require_slot(slot_index)
            seed = (
                int(reset_spec.seed)
                if reset_spec.seed is not None
                else slot.default_seed
            )
            try:
                raw, info = slot.session.reset(
                    seed=seed, options=dict(reset_spec.options or {})
                )
            except MujocoSessionError as exc:
                raise self._session_failure(exc, side_effect_applied=True) from exc
            slot.seed = seed
            slot.step_index = 0
            slot.episode_return = 0.0
            slot.terminated = False
            slot.truncated = False
            slot.success = self._success_from(info, slot.episode_return)
            slot.started = True
            slot.instruction = reset_spec.instruction or self.config.instruction
            slot.raw_observation = raw
            try:
                frame = slot.session.render() if self.config.uses_rgb else None
            except MujocoSessionError as exc:
                raise self._session_failure(exc, side_effect_applied=True) from exc
            observation = self._make_observation(slot_index, raw, frame)
            slot.observation_layout = self._layout_of(observation)
            self._check_layout(slot.observation_layout)
            slot.last_observation = observation
            observations.append(observation)
        return observations

    def observe(self, slots: Sequence[int]) -> list[Observation]:
        observations: list[Observation] = []
        for index in slots:
            slot = self._require_started(index)
            if slot.last_observation is None:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.SESSION_NOT_READY, "mujoco slot has no observation"
                    )
                )
            observations.append(slot.last_observation)
        return observations

    def chunk_step(
        self, slots: Sequence[int], chunk_actions: Sequence[np.ndarray]
    ) -> list[ChunkOutcome]:
        if len(slots) != len(chunk_actions):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"slots/actions length mismatch: {len(slots)} != {len(chunk_actions)}",
                )
            )
        # Validate every lane before the first Gym call. Otherwise a malformed
        # action for a later lane could partially advance earlier lanes in the
        # same batched Runtime operation.
        prepared: list[np.ndarray] = []
        for slot_index, actions in zip(slots, chunk_actions, strict=True):
            slot = self._require_active_slot(slot_index)
            prepared.append(self._prepare_actions(slot, actions))
        return [
            self._chunk_step_one(slot_index, actions, actions_validated=True)
            for slot_index, actions in zip(slots, prepared, strict=True)
        ]

    def _chunk_step_one(
        self,
        slot_index: int,
        actions: np.ndarray,
        *,
        actions_validated: bool = False,
    ) -> ChunkOutcome:
        slot = self._require_active_slot(slot_index)
        block = actions if actions_validated else self._prepare_actions(slot, actions)
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        infos: list[dict[str, Any]] = []
        frames: list[Observation] = []
        step_indices: list[int] = []
        defer_rgb_render = self.config.uses_rgb and not self.config.return_all_frames
        for action in block:
            if slot.terminated or slot.truncated:
                break
            try:
                raw, reward, terminated, truncated, info = slot.session.step(action)
            except MujocoSessionError as exc:
                raise self._session_failure(exc, side_effect_applied=True) from exc
            slot.step_index += 1
            slot.episode_return += float(reward)
            slot.terminated = slot.terminated or bool(terminated)
            slot.truncated = slot.truncated or bool(truncated)
            step_success = self._success_from(info, slot.episode_return)
            if step_success is not None:
                slot.success = bool(slot.success) or bool(step_success)
            slot.raw_observation = raw
            step_indices.append(slot.step_index)
            if defer_rgb_render:
                layout: tuple[tuple[str, tuple[int, ...]], ...] = ()
                if self.config.observation_mode == "rgb_state":
                    try:
                        _state, layout = _layout_and_state(raw)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeApiError(
                            make_error(
                                ErrorCode.INVALID_ARGUMENT,
                                f"unsupported MuJoCo observation: {exc}",
                            )
                        ) from exc
                self._check_layout(layout)
            else:
                try:
                    frame = slot.session.render() if self.config.uses_rgb else None
                except MujocoSessionError as exc:
                    raise self._session_failure(exc, side_effect_applied=True) from exc
                observation = self._make_observation(slot_index, raw, frame)
                self._check_layout(self._layout_of(observation))
                slot.last_observation = observation
                frames.append(observation)
            rewards.append(float(reward))
            terminations.append(slot.terminated)
            truncations.append(slot.truncated)
            infos.append(
                {
                    **_json_safe(info),
                    "episode_return": float(slot.episode_return),
                    "success": slot.success,
                }
            )
        if defer_rgb_render:
            try:
                frame = slot.session.render()
            except MujocoSessionError as exc:
                raise self._session_failure(exc, side_effect_applied=True) from exc
            final_observation = self._make_observation(
                slot_index, slot.raw_observation, frame
            )
            self._check_layout(self._layout_of(final_observation))
            slot.last_observation = final_observation
            frames = [
                Observation(
                    session_id=SessionId(""),
                    episode_id=EpisodeId(0),
                    step_index=step_index,
                )
                for step_index in step_indices
            ]
        final = slot.last_observation
        if final is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY, "mujoco slot has no observation"
                )
            )
        return normalize_chunk_outcome(
            behavior=self.behavior,
            final_observation=final,
            step_observations=frames,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            requested_horizon=int(block.shape[0]),
            success=slot.success,
            per_step_info=infos,
            include_step_observations=self.config.return_all_frames,
            info={
                "env_id": self.config.env_id,
                "episode_return": float(slot.episode_return),
                "process_isolation": self.config.resolved_process_isolation,
            },
        )

    def _prepare_actions(self, slot: _MujocoSlot, actions: np.ndarray) -> np.ndarray:
        try:
            block = np.asarray(actions, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"actions must be numeric and convertible to float32: {exc}",
                )
            ) from exc
        expected = int(self.config.action_dim)
        if block.ndim != 2 or block.shape[1] != expected:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"mujoco expects [chunk, {expected}] actions, got {block.shape}",
                    action_dim=expected,
                )
            )
        if block.shape[0] < 1:
            raise RuntimeApiError(
                make_error(ErrorCode.INVALID_ARGUMENT, "action chunk must not be empty")
            )
        if not np.isfinite(block).all():
            raise RuntimeApiError(
                make_error(ErrorCode.INVALID_ARGUMENT, "actions contain NaN or Inf")
            )
        low = np.asarray(slot.session.descriptor["action_low"], dtype=np.float32)
        high = np.asarray(slot.session.descriptor["action_high"], dtype=np.float32)
        if self.config.clip_actions:
            return np.clip(block, low, high).astype(np.float32, copy=False)
        if np.any(block < low) or np.any(block > high):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    "actions exceed the MuJoCo action-space bounds",
                )
            )
        return block

    def _make_observation(
        self, slot_index: int, raw: Any, frame: np.ndarray | None
    ) -> Observation:
        slot = self._require_slot(slot_index)
        if self.config.provider == "arx_ac_one":
            if not isinstance(raw, Mapping) or "state" not in raw:
                raise RuntimeApiError(make_error(ErrorCode.INVALID_ARGUMENT, "ARX observation must contain state"))
            state_array = np.asarray(raw["state"], dtype=np.float32)
            if state_array.shape != (14,) or not np.isfinite(state_array).all():
                raise RuntimeApiError(make_error(ErrorCode.INVALID_ARGUMENT, "ARX state must be finite [14]"))
            refs = {}
            for name in ("front_rgb", "left_rgb", "right_rgb"):
                array = np.asarray(raw.get(name))
                expected = (self.config.image_height, self.config.image_width, 3)
                if array.dtype != np.uint8 or array.shape != expected:
                    raise RuntimeApiError(make_error(ErrorCode.INVALID_ARGUMENT, f"ARX {name} must be uint8 {expected}"))
                refs[name] = payload_module.encode_image(np.ascontiguousarray(array))
            return Observation(
                session_id=SessionId(""), episode_id=EpisodeId(0), step_index=slot.step_index,
                main_image=refs["front_rgb"], wrist_image=refs["left_rgb"],
                extra_view_images=[refs["right_rgb"]], state=state_array.tolist(),
                instruction=slot.instruction or self.config.instruction,
                extras={
                    "env_family": MUJOCO_ENV_FAMILY, "env_id": self.config.env_id,
                    "provider": self.config.provider, "raw_state": state_array.tolist(),
                    "camera_names": {"front_rgb": "main_image", "left_rgb": "wrist_image", "right_rgb": "extra_view_images.0"},
                    "seed": slot.seed, "slot_index": slot_index,
                },
            )
        state: list[float] = []
        layout: tuple[tuple[str, tuple[int, ...]], ...] = ()
        if self.config.observation_mode in {"state", "rgb_state"}:
            try:
                state, layout = _layout_and_state(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        f"unsupported MuJoCo observation: {exc}",
                    )
                ) from exc
            if not layout:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        "MuJoCo state observation must contain at least one numeric value",
                    )
                )
        main_image = None
        if self.config.uses_rgb:
            if frame is None:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.ENV_FAILURE,
                        "RGB observation mode produced no rendered frame",
                    )
                )
            array = np.asarray(frame)
            expected = (int(self.config.image_height), int(self.config.image_width), 3)
            if array.shape != expected or array.dtype != np.uint8:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        f"rendered frame must be uint8 {expected}, got "
                        f"shape={array.shape}, dtype={array.dtype}",
                    )
                )
            main_image = payload_module.encode_image(array)
        return Observation(
            session_id=SessionId(""),
            episode_id=EpisodeId(0),
            step_index=slot.step_index,
            main_image=main_image,
            state=state,
            instruction=slot.instruction or self.config.instruction,
            extras={
                "env_family": MUJOCO_ENV_FAMILY,
                "env_id": self.config.env_id,
                "provider": self.config.provider,
                "action_names": list(slot.session.descriptor.get("action_names", ())),
                "observation_layout": [
                    {
                        "path": path,
                        "shape": list(shape),
                        "start": sum(
                            math.prod(previous_shape)
                            for _previous_path, previous_shape in layout[:index]
                        ),
                        "stop": sum(
                            math.prod(previous_shape)
                            for _previous_path, previous_shape in layout[: index + 1]
                        ),
                        "dtype": "float32",
                    }
                    for index, (path, shape) in enumerate(layout)
                ],
                "seed": slot.seed,
                "slot_index": slot_index,
            },
        )

    @staticmethod
    def _layout_of(observation: Observation) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (str(item["path"]), tuple(int(dim) for dim in item["shape"]))
            for item in observation.extras.get("observation_layout", ())
        )

    def _check_layout(self, layout: tuple[tuple[str, tuple[int, ...]], ...]) -> None:
        if self._expected_layout is None:
            self._expected_layout = layout
            return
        if layout != self._expected_layout:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.ENV_FAILURE,
                    "MuJoCo observation schema changed within one pool",
                    expected_layout=self._expected_layout,
                    actual_layout=layout,
                )
            )

    def _success_from(
        self, info: Mapping[str, Any], episode_return: float
    ) -> bool | None:
        if self.config.success_mode == "none":
            return None
        if self.config.success_mode == "return_threshold":
            assert self.config.success_return_threshold is not None
            return episode_return >= float(self.config.success_return_threshold)
        if self.config.success_info_key not in info:
            return None
        value = np.asarray(info[self.config.success_info_key])
        if value.size != 1:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"success info key {self.config.success_info_key!r} must be scalar",
                )
            )
        return bool(value.reshape(-1)[0])

    def _require_slot(self, slot_index: int) -> _MujocoSlot:
        if not 0 <= slot_index < len(self._slots):
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"mujoco slot {slot_index} is outside pool size {len(self._slots)}",
                )
            )
        return self._slots[slot_index]

    def _session_failure(
        self, exc: MujocoSessionError, *, side_effect_applied: bool = False
    ) -> RuntimeApiError:
        return RuntimeApiError(
            make_error(
                ErrorCode.ENV_FAILURE,
                str(exc),
                side_effect_applied=side_effect_applied,
                env_family=MUJOCO_ENV_FAMILY,
                env_id=self.config.env_id,
            )
        )

    def _require_started(self, slot_index: int) -> _MujocoSlot:
        slot = self._require_slot(slot_index)
        if not slot.started:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"mujoco slot {slot_index} has not been reset",
                )
            )
        return slot

    def _require_active_slot(self, slot_index: int) -> _MujocoSlot:
        slot = self._require_started(slot_index)
        if slot.terminated or slot.truncated:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.EPISODE_TERMINATED,
                    f"mujoco slot {slot_index} has already ended; reset it first",
                )
            )
        return slot

    def extension(
        self, slot: int, namespace: str, method: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        del slot, args
        raise RuntimeApiError(
            make_error(
                ErrorCode.UNSUPPORTED_EXTENSION,
                f"mujoco does not implement extension {namespace}.{method}",
            )
        )


def mujoco_env_capability() -> EnvFamilyCapability:
    """Return the MuJoCo family's Gateway capability."""
    return capability_from_behavior(
        behavior_for(MUJOCO_ENV_FAMILY),
        supports_auto_reset=False,
        supports_reset_state_id=False,
    )


class MujocoEnvFamily:
    """MuJoCo ``EnvFamilyAdapter``."""

    @property
    def env_family(self) -> str:
        return MUJOCO_ENV_FAMILY

    @property
    def capability(self) -> EnvFamilyCapability:
        return mujoco_env_capability()

    def create_core(self) -> MujocoEnvCore:
        return MujocoEnvCore()


def register_mujoco_env_family(*, replace: bool = True) -> MujocoEnvFamily:
    """Register and return the MuJoCo family adapter."""
    adapter = MujocoEnvFamily()
    register_env_family(adapter, replace=replace)
    return adapter
