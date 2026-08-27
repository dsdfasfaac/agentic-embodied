# Copyright (c) 2026 Zetta Contributors
"""Simulator-free tests for the Gymnasium MuJoCo Runtime backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rollout_runtime.adapters.eval_adapter import EvaluationAdapter, EvaluationTask
from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends import mujoco_session as session_module
from rollout_runtime.backends.mujoco_env import (
    MujocoEnvConfig,
    MujocoEnvCore,
    mujoco_env_capability,
)
from rollout_runtime.config.schema import load_config
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.env_execution import PER_SLOT_FORM
from rollout_runtime.core.env_registry import validate_env_spec
from rollout_runtime.launch.local import build_local_components
from tests.runtime.conftest import local_runtime_config


class _FakeActionSpace:
    """Small continuous Box-like action space used by the stub."""

    def __init__(self, action_dim: int) -> None:
        self.shape = (action_dim,)
        self.low = np.full((action_dim,), -1.0, dtype=np.float32)
        self.high = np.full((action_dim,), 1.0, dtype=np.float32)


class _FakeMujocoEnv:
    """Gymnasium-shaped environment with configurable lifecycle behavior."""

    def __init__(self, owner: _FakeGymnasium, env_id: str, kwargs: dict[str, Any]):
        self.owner = owner
        self.env_id = env_id
        self.kwargs = dict(kwargs)
        self.action_space = _FakeActionSpace(owner.action_dim)
        self.actions: list[np.ndarray] = []
        self.resets: list[tuple[int | None, dict[str, Any]]] = []
        self.step_index = 0
        self.seed: int | None = None
        self.closed = False

    def reset(
        self, *, seed: int | None, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        self.seed = seed
        self.step_index = 0
        self.resets.append((seed, dict(options)))
        info: dict[str, Any] = {"reset_seed": seed}
        if self.owner.reset_success is not None:
            info["is_success"] = self.owner.reset_success
        return self._observation(), info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.owner.fail_step:
            raise RuntimeError("native step failed")
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.step_index += 1
        terminated = (
            self.owner.terminate_at is not None
            and self.step_index >= self.owner.terminate_at
        )
        truncated = (
            self.owner.truncate_at is not None
            and self.step_index >= self.owner.truncate_at
        )
        info: dict[str, Any] = {"step": self.step_index}
        if self.owner.success_at is not None:
            info["is_success"] = self.step_index >= self.owner.success_at
        return (
            self._observation(),
            float(self.owner.reward_scale * self.step_index),
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray:
        self.owner.render_calls += 1
        height = self.owner.render_height or int(self.kwargs["height"])
        width = self.owner.render_width or int(self.kwargs["width"])
        channels = self.owner.render_channels
        return np.full(
            (height, width, channels),
            self.step_index % 255,
            dtype=self.owner.render_dtype,
        )

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> Any:
        width = 3 if self.owner.change_schema and self.step_index else 2
        seed_value = float(self.seed or 0)
        if self.owner.observation_kind == "dict":
            return {
                "z": np.arange(width, dtype=np.float64) + self.step_index,
                "a": np.float64(seed_value),
            }
        if self.owner.observation_kind == "empty_dict":
            return {}
        return np.asarray([seed_value, float(self.step_index)], dtype=np.float64)


class _FakeGymnasium:
    """Object exposing only the ``gymnasium.make`` surface used by the backend."""

    def __init__(self) -> None:
        self.action_dim = 2
        self.observation_kind = "array"
        self.change_schema = False
        self.terminate_at: int | None = None
        self.truncate_at: int | None = None
        self.success_at: int | None = None
        self.reset_success: bool | None = None
        self.reward_scale = 1.0
        self.fail_step = False
        self.render_height: int | None = None
        self.render_width: int | None = None
        self.render_channels = 3
        self.render_dtype: Any = np.uint8
        self.render_calls = 0
        self.instances: list[_FakeMujocoEnv] = []

    def make(self, env_id: str, **kwargs: Any) -> _FakeMujocoEnv:
        env = _FakeMujocoEnv(self, env_id, kwargs)
        self.instances.append(env)
        return env


@pytest.fixture
def fake_gymnasium(monkeypatch: pytest.MonkeyPatch) -> _FakeGymnasium:
    """Replace the lazy Gymnasium import without installing MuJoCo."""
    fake = _FakeGymnasium()
    monkeypatch.setattr(session_module, "_load_gymnasium", lambda: fake)
    return fake


def _spec(**overrides: Any) -> EnvSpecMsg:
    config: dict[str, Any] = {
        "env_id": "StubMujoco-v0",
        "action_dim": 2,
        "chunk_size": 3,
        "process_isolation": False,
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="mujoco", env_config=config, pool_size=1)


def _build_core(fake_gymnasium: _FakeGymnasium, **overrides: Any) -> MujocoEnvCore:
    del fake_gymnasium
    core = MujocoEnvCore()
    core.build(_spec(**overrides), num_envs=1, seed_offset=10)
    return core


def test_config_defaults_and_capability_are_cpu_capable() -> None:
    """State-only defaults stay in-process and the family itself needs no GPU."""
    config = MujocoEnvConfig.from_mapping({})
    assert config.env_id == "InvertedPendulum-v5"
    assert config.resolved_process_isolation is False
    assert config.core_form == PER_SLOT_FORM
    capability = mujoco_env_capability()
    assert capability.env_family == "mujoco"
    assert capability.needs_accelerator is False
    assert capability.core_forms == frozenset({PER_SLOT_FORM})

    rgb = MujocoEnvConfig.from_mapping({"observation_mode": "rgb"})
    assert rgb.render_mode == "rgb_array"
    assert rgb.resolved_process_isolation is True


def test_rtx4090_preset_requests_accelerator_per_spec() -> None:
    """The RGB preset requests GPU placement without making the family GPU-only."""
    config = load_config("rtx4090_mujoco")
    assert config.env_family == "mujoco"
    assert config.env_resource_hints == {"accelerator": True}
    assert config.env_worker.accelerator_present() is True
    assert config.env_config["process_isolation"] is True

    with pytest.raises(ValueError, match="accelerator"):
        load_config({"env_resource_hints": {"accelerator": "yes"}})

    with pytest.raises(RuntimeApiError) as excinfo:
        validate_env_spec(
            EnvSpecMsg(
                env_family="mujoco",
                resource_hints={"accelerator": "yes"},
            ),
            capabilities={"mujoco": mujoco_env_capability()},
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    "config",
    [
        {"unknown": True},
        {"provider": "other"},
        {"observation_mode": "pixels"},
        {"observation_mode": "state", "render_mode": "rgb_array"},
        {"camera_name": "track"},
        {"image_width": "256"},
        {"action_dim": True},
        {"clip_actions": 1},
        {"process_isolation": "yes"},
        {"rpc_timeout_s": float("inf")},
        {"success_mode": "return_threshold"},
        {"success_return_threshold": 1.0},
        {"return_all_frames": 1},
        {"core_form": "lockstep_vector"},
    ],
)
def test_config_rejects_unknown_or_ambiguous_values(config: dict[str, Any]) -> None:
    """Config errors fail before a simulator is constructed."""
    with pytest.raises(RuntimeApiError) as excinfo:
        MujocoEnvConfig.from_mapping(config)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_reset_flattens_dict_observation_and_propagates_seed_options(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """Dict keys are stable, sliced, finite, and reset inputs pass through."""
    fake_gymnasium.observation_kind = "dict"
    core = MujocoEnvCore()
    core.build(_spec(instruction="default"), num_envs=2, seed_offset=10)
    observation = core.reset(
        [1],
        ResetSpec(seed=42, instruction="override", options={"domain": "test"}),
    )[0]

    assert observation.state == pytest.approx([42.0, 0.0, 1.0])
    assert observation.instruction == "override"
    assert observation.extras["seed"] == 42
    assert observation.extras["observation_layout"] == [
        {"path": "a", "shape": [], "start": 0, "stop": 1, "dtype": "float32"},
        {
            "path": "z",
            "shape": [2],
            "start": 1,
            "stop": 3,
            "dtype": "float32",
        },
    ]
    assert fake_gymnasium.instances[1].resets[-1] == (42, {"domain": "test"})

    assert core.add_slot(99) == 2
    assert core.slot_count() == 3
    with pytest.raises(RuntimeApiError):
        core.remove_slot(0)
    core.remove_slot(2)
    assert core.slot_count() == 2
    core.close()
    core.close()
    assert all(env.closed for env in fake_gymnasium.instances)


def test_build_rejects_action_shape_mismatch_and_closes_session(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """The configured policy shape must agree with Gymnasium before reset."""
    core = MujocoEnvCore()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.build(_spec(action_dim=1), num_envs=1, seed_offset=0)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert fake_gymnasium.instances[0].closed is True
    assert core.slot_count() == 0


def test_chunk_clips_actions_accumulates_reward_and_stops_early(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """One Gym step equals one Runtime horizon unit and termination stops a chunk."""
    fake_gymnasium.terminate_at = 2
    fake_gymnasium.success_at = 2
    core = _build_core(fake_gymnasium, success_mode="info_key", return_all_frames=True)
    core.reset([0], ResetSpec(seed=1))
    outcome = core.chunk_step(
        [0],
        [
            np.asarray(
                [[3.0, -3.0], [0.25, 0.5], [0.75, 0.75]],
                dtype=np.float32,
            )
        ],
    )[0]

    assert outcome.executed_horizon == 2
    assert outcome.reward == pytest.approx(3.0)
    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.success is True
    assert outcome.per_step is not None and len(outcome.per_step) == 2
    assert all(record.observation is not None for record in outcome.per_step)
    assert fake_gymnasium.instances[0].actions[0] == pytest.approx([1.0, -1.0])
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((1, 2), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.EPISODE_TERMINATED
    core.close()


def test_rgb_chunk_without_all_frames_renders_only_final_observation(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """Bandwidth-saving RGB chunks render once while preserving step audit rows."""
    core = _build_core(
        fake_gymnasium,
        observation_mode="rgb_state",
        render_mode="rgb_array",
        return_all_frames=False,
    )
    core.reset([0], ResetSpec(seed=1))
    render_calls_before = fake_gymnasium.render_calls
    outcome = core.chunk_step([0], [np.zeros((3, 2), dtype=np.float32)])[0]

    assert fake_gymnasium.render_calls - render_calls_before == 1
    assert outcome.observation is not None
    assert outcome.observation.step_index == 3
    assert outcome.per_step is not None
    assert [record.step_index for record in outcome.per_step] == [1, 2, 3]
    assert all(record.observation is None for record in outcome.per_step)
    core.close()


def test_action_validation_happens_before_environment_side_effects(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """Rank, dimension, finite-value, and bounds errors execute no Gym step."""
    core = _build_core(fake_gymnasium, clip_actions=False)
    core.reset([0], ResetSpec(seed=1))
    invalid = (
        np.zeros((2,), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([[np.nan, 0.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        [["not-a-number", 0.0]],
    )
    for actions in invalid:
        with pytest.raises(RuntimeApiError) as excinfo:
            core.chunk_step([0], [actions])
        assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert fake_gymnasium.instances[0].actions == []
    core.close()


def test_multi_slot_action_validation_is_atomic(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """An invalid later lane cannot partially advance an earlier lane."""
    core = MujocoEnvCore()
    core.build(_spec(clip_actions=False), num_envs=2, seed_offset=0)
    core.reset([0, 1], ResetSpec(seed=1))

    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step(
            [0, 1],
            [
                np.zeros((1, 2), dtype=np.float32),
                np.asarray([[2.0, 0.0]], dtype=np.float32),
            ],
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT
    assert fake_gymnasium.instances[0].actions == []
    assert fake_gymnasium.instances[1].actions == []

    core.close()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.add_slot(2)
    assert excinfo.value.info.code is ErrorCode.SESSION_NOT_READY


def test_native_session_failure_is_normalized_with_side_effect_status(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """Simulator exceptions become ENV_FAILURE instead of leaking raw errors."""
    core = _build_core(fake_gymnasium)
    core.reset([0], ResetSpec(seed=1))
    fake_gymnasium.fail_step = True
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((1, 2), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.ENV_FAILURE
    assert excinfo.value.info.side_effect_applied is True
    core.close()


def test_success_modes_keep_return_only_separate_from_binary_success(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """None, False, and True remain distinct and True is latched."""
    return_only = _build_core(fake_gymnasium)
    return_only.reset([0], ResetSpec(seed=1))
    first = return_only.chunk_step([0], [np.zeros((1, 2), dtype=np.float32)])[0]
    assert first.success is None
    return_only.close()

    threshold = _build_core(
        fake_gymnasium,
        success_mode="return_threshold",
        success_return_threshold=2.5,
    )
    threshold.reset([0], ResetSpec(seed=1))
    below = threshold.chunk_step([0], [np.zeros((1, 2), dtype=np.float32)])[0]
    above = threshold.chunk_step([0], [np.zeros((1, 2), dtype=np.float32)])[0]
    assert below.success is False
    assert above.success is True
    threshold.close()


def test_rgb_state_maps_rendered_frame_and_camera_config(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """RGB is fixed-shape uint8 HWC and explicit camera settings reach Gym."""
    fake_gymnasium.render_channels = 4
    core = _build_core(
        fake_gymnasium,
        observation_mode="rgb_state",
        render_mode="rgb_array",
        image_height=8,
        image_width=10,
        camera_name="track",
    )
    observation = core.reset([0], ResetSpec(seed=4))[0]
    assert observation.main_image is not None
    image = payload_module.decode_image(observation.main_image)
    assert image.shape == (8, 10, 3)
    assert image.dtype == np.uint8
    assert observation.state == pytest.approx([4.0, 0.0])
    kwargs = fake_gymnasium.instances[0].kwargs
    assert kwargs["render_mode"] == "rgb_array"
    assert kwargs["camera_name"] == "track"
    assert kwargs["height"] == 8
    assert kwargs["width"] == 10
    core.close()


def test_render_and_observation_schema_mismatches_fail_explicitly(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """A pool never silently pads a changed state or resized frame."""
    fake_gymnasium.observation_kind = "dict"
    fake_gymnasium.change_schema = True
    core = _build_core(fake_gymnasium)
    core.reset([0], ResetSpec(seed=1))
    with pytest.raises(RuntimeApiError) as excinfo:
        core.chunk_step([0], [np.zeros((1, 2), dtype=np.float32)])
    assert excinfo.value.info.code is ErrorCode.ENV_FAILURE
    core.close()

    fake_gymnasium.observation_kind = "array"
    fake_gymnasium.change_schema = False
    fake_gymnasium.render_height = 9
    with pytest.raises(RuntimeApiError) as excinfo:
        _build_core(
            fake_gymnasium,
            observation_mode="rgb",
            render_mode="rgb_array",
            image_height=8,
            image_width=10,
        )
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_empty_state_and_undeclared_extension_are_rejected(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """Unsupported observation and privileged APIs fail with stable codes."""
    fake_gymnasium.observation_kind = "empty_dict"
    with pytest.raises(RuntimeApiError) as excinfo:
        _build_core(fake_gymnasium)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT

    fake_gymnasium.observation_kind = "array"
    core = _build_core(fake_gymnasium)
    with pytest.raises(RuntimeApiError) as excinfo:
        core.extension(0, "mujoco", "model", {})
    assert excinfo.value.info.code is ErrorCode.UNSUPPORTED_EXTENSION
    core.close()


async def test_local_runtime_supports_explicit_and_policy_actions(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """The MuJoCo core runs end-to-end only through the public Gateway facade."""
    fake_gymnasium.terminate_at = 3
    fake_gymnasium.success_at = 3
    config = local_runtime_config()
    config.env_family = "mujoco"
    config.env_config = dict(_spec(success_mode="info_key").env_config)
    runtime = build_local_components(config)
    await runtime.start()
    facade = RuntimeGymEnv(runtime.gateway, _spec(success_mode="info_key"))
    try:
        observation, info = await facade.reset(seed=12)
        assert observation.state == pytest.approx([12.0, 0.0])
        assert info["session_id"] == str(facade.session_id)

        _observation, reward, terminated, truncated, step_info = await facade.step(
            np.zeros((1, 2), dtype=np.float32)
        )
        assert reward == pytest.approx(1.0)
        assert terminated is False and truncated is False
        assert step_info["success"] is False

        _observation, reward, terminated, truncated, step_info = await facade.step()
        assert reward == pytest.approx(5.0)
        assert terminated is True and truncated is False
        assert step_info["executed_horizon"] == 2
        assert step_info["success"] is True
        assert len(runtime.env_workers[0].sessions) == 1

        await facade.reset(seed=13)
        assert len(runtime.env_workers[0].sessions) == 1
    finally:
        await facade.close()
        assert runtime.env_workers[0].sessions == {}
        await runtime.gateway.stop()
        await runtime.aclose()


async def test_return_only_mujoco_episode_is_valid_but_not_binary_scored(
    fake_gymnasium: _FakeGymnasium,
) -> None:
    """Evaluation reports return without inventing success from termination."""
    fake_gymnasium.terminate_at = 2
    config = local_runtime_config()
    config.env_family = "mujoco"
    config.env_config = dict(_spec(success_mode="none").env_config)
    runtime = build_local_components(config)
    await runtime.start()
    try:
        adapter = EvaluationAdapter(
            runtime.gateway,
            _spec(success_mode="none"),
            concurrency=1,
            max_steps=4,
        )
        report = await adapter.run_episodes([EvaluationTask(seed=3)])
        assert report.valid == 1
        assert report.invalid == 0
        assert report.binary_scored == 0
        assert report.successes == 0
        assert report.success_rate is None
        assert report.outcomes[0].success is None
        assert report.outcomes[0].total_reward == pytest.approx(3.0)
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()
