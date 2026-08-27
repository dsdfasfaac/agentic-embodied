# Copyright (c) 2026 Zetta Contributors
"""Real MuJoCo acceptance tests for an explicitly provisioned host."""

from __future__ import annotations

import numpy as np
import pytest

from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends.mujoco_env import MujocoEnvCore
from rollout_runtime.core import payload as payload_module
from rollout_runtime.launch.local import build_local_components
from tests.runtime.conftest import local_runtime_config

pytestmark = [pytest.mark.mujoco, pytest.mark.remote]


def _state_spec(**overrides: object) -> EnvSpecMsg:
    config: dict[str, object] = {
        "env_id": "InvertedPendulum-v5",
        "observation_mode": "state",
        "render_mode": None,
        "action_dim": 1,
        "chunk_size": 4,
        "max_episode_steps": 1000,
        "process_isolation": False,
        "success_mode": "none",
    }
    config.update(overrides)
    return EnvSpecMsg(env_family="mujoco", env_config=config)


def _rgb_spec(**overrides: object) -> EnvSpecMsg:
    config = dict(_state_spec().env_config)
    config.update(
        {
            "observation_mode": "rgb_state",
            "render_mode": "rgb_array",
            "render_backend": "egl",
            "image_width": 160,
            "image_height": 120,
            "process_isolation": True,
        }
    )
    config.update(overrides)
    return EnvSpecMsg(
        env_family="mujoco",
        env_config=config,
        resource_hints={"accelerator": True},
    )


def _rollout_state(core: MujocoEnvCore, seed: int) -> tuple[np.ndarray, np.ndarray]:
    core.reset([0], ResetSpec(seed=seed))
    states: list[np.ndarray] = []
    rewards: list[float] = []
    actions = np.asarray([[0.1], [-0.2], [0.05], [0.0]], dtype=np.float32)
    for action in actions:
        outcome = core.chunk_step([0], [action[None, :]])[0]
        assert outcome.observation is not None
        states.append(np.asarray(outcome.observation.state, dtype=np.float64))
        rewards.append(outcome.reward)
    return np.stack(states), np.asarray(rewards)


def test_real_state_rollout_is_deterministic() -> None:
    """Same seed and action sequence reproduce state and reward."""
    core = MujocoEnvCore()
    core.build(_state_spec(), num_envs=1, seed_offset=0)
    try:
        first_states, first_rewards = _rollout_state(core, 17)
        second_states, second_rewards = _rollout_state(core, 17)
        assert np.isfinite(first_states).all()
        assert np.isfinite(first_rewards).all()
        assert np.allclose(first_states, second_states, rtol=0.0, atol=1e-12)
        assert np.allclose(first_rewards, second_rewards, rtol=0.0, atol=1e-12)
    finally:
        core.close()


def test_real_egl_renderer_isolated_process_closes() -> None:
    """EGL produces fixed RGB and its per-slot subprocess is reclaimed."""
    core = MujocoEnvCore()
    core.build(_rgb_spec(), num_envs=1, seed_offset=0)
    process = core._slots[0].session._process
    try:
        for seed in range(10):
            observation = core.reset([0], ResetSpec(seed=seed))[0]
            assert observation.main_image is not None
            image = payload_module.decode_image(observation.main_image)
            assert image.shape == (120, 160, 3)
            assert image.dtype == np.uint8
        assert process.is_alive()
    finally:
        core.close()
    assert not process.is_alive()


def test_real_invalid_camera_fails_without_leaving_a_slot() -> None:
    """A bad camera is rejected during cold validation and its child closes."""
    core = MujocoEnvCore()
    with pytest.raises(RuntimeApiError) as excinfo:
        core.build(
            _rgb_spec(camera_name="camera-that-does-not-exist"),
            num_envs=1,
            seed_offset=0,
        )
    assert excinfo.value.info.code is ErrorCode.ENV_FAILURE
    assert core.slot_count() == 0


def test_real_dead_renderer_process_returns_normalized_error() -> None:
    """A lost native child fails promptly as ENV_FAILURE instead of hanging."""
    core = MujocoEnvCore()
    core.build(_rgb_spec(rpc_timeout_s=5.0), num_envs=1, seed_offset=0)
    process = core._slots[0].session._process
    process.terminate()
    process.join(timeout=5.0)
    assert not process.is_alive()
    try:
        with pytest.raises(RuntimeApiError) as excinfo:
            core.reset([0], ResetSpec(seed=9))
        assert excinfo.value.info.code is ErrorCode.ENV_FAILURE
    finally:
        core.close()


def test_real_post_termination_action_is_rejected_without_side_effect() -> None:
    """The first time-limit step ends the episode and later actions are rejected."""
    core = MujocoEnvCore()
    core.build(_state_spec(max_episode_steps=1), num_envs=1, seed_offset=0)
    try:
        core.reset([0], ResetSpec(seed=3))
        outcome = core.chunk_step([0], [np.zeros((2, 1), dtype=np.float32)])[0]
        assert outcome.executed_horizon == 1
        assert outcome.truncated is True
        step_index = core._slots[0].step_index
        with pytest.raises(RuntimeApiError) as excinfo:
            core.chunk_step([0], [np.zeros((1, 1), dtype=np.float32)])
        assert excinfo.value.info.code is ErrorCode.EPISODE_TERMINATED
        assert core._slots[0].step_index == step_index
    finally:
        core.close()


async def test_real_local_runtime_reuses_one_session_for_100_steps() -> None:
    """Repeated episodes cross the Gateway without leaking sessions."""
    config = local_runtime_config()
    config.env_family = "mujoco"
    config.env_config = dict(_state_spec(max_episode_steps=20, chunk_size=1).env_config)
    runtime = build_local_components(config)
    await runtime.start()
    facade = RuntimeGymEnv(
        runtime.gateway,
        _state_spec(max_episode_steps=20, chunk_size=1),
    )
    try:
        await facade.reset(seed=5)
        executed = 0
        episode = 0
        while executed < 100:
            _observation, _reward, terminated, truncated, info = await facade.step(
                np.zeros((1,), dtype=np.float32)
            )
            executed += int(info["executed_horizon"])
            if terminated or truncated:
                episode += 1
                await facade.reset(seed=5 + episode)
        assert executed == 100
        assert len(runtime.env_workers[0].sessions) == 1
    finally:
        await facade.close()
        assert runtime.env_workers[0].sessions == {}
        await runtime.gateway.stop()
        await runtime.aclose()
