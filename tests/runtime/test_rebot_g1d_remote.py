# Copyright (c) 2026 Zetta Contributors
"""Real reBot G1-D/Dex1-1 acceptance test for a provisioned MuJoCo host."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends.mujoco_env import MujocoEnvCore
from rollout_runtime.backends.rebot_g1d_session import (
    REBOT_G1D_ACTION_NAMES,
    compute_rebot_asset_manifest,
)
from rollout_runtime.core import payload as payload_module

pytestmark = [pytest.mark.mujoco, pytest.mark.remote]


def _assets() -> tuple[Path, str]:
    root = os.environ.get("REBOT_G1D_ASSET_ROOT")
    digest = os.environ.get("REBOT_G1D_ASSET_SHA256")
    if not root or not digest:
        pytest.skip(
            "set REBOT_G1D_ASSET_ROOT and REBOT_G1D_ASSET_SHA256 on the "
            "configured MuJoCo host"
        )
    return Path(root), digest


def _slice(observation: object, path: str) -> np.ndarray:
    extras = getattr(observation, "extras")
    state = getattr(observation, "state")
    for item in extras["observation_layout"]:
        if item["path"] == path:
            return np.asarray(
                state[int(item["start"]) : int(item["stop"])], dtype=np.float32
            )
    raise AssertionError(f"observation has no {path!r} state slice")


def test_real_rebot_scene_reset_step_render_and_cleanup() -> None:
    """Pinned assets cross the core with deterministic reset and real EGL RGB."""
    asset_root, expected_digest = _assets()
    manifest = compute_rebot_asset_manifest(asset_root)
    assert manifest["sha256"] == expected_digest
    spec = EnvSpecMsg(
        env_family="mujoco",
        env_config={
            "provider": "rebot_g1d",
            "env_id": "reBot-DevArm-Grasp-v0",
            "asset_root": str(asset_root),
            "asset_manifest_sha256": expected_digest,
            "rebot_scene_config": "config/g1d_mujoco.yaml",
            "rebot_randomize_bottle": True,
            "observation_mode": "rgb_state",
            "render_mode": "rgb_array",
            "render_backend": "egl",
            "camera_name": "overview",
            "image_width": 320,
            "image_height": 240,
            "action_dim": 22,
            "chunk_size": 10,
            "max_episode_steps": 100,
            "process_isolation": True,
            "rpc_timeout_s": 180.0,
            "success_mode": "none",
        },
        resource_hints={"accelerator": True},
    )
    core = MujocoEnvCore()
    core.build(spec, num_envs=1, seed_offset=0)
    child = core._slots[0].session._process
    try:
        first = core.reset([0], ResetSpec(seed=17))[0]
        second = core.reset([0], ResetSpec(seed=17))[0]
        np.testing.assert_allclose(
            _slice(first, "water_bottle_position"),
            _slice(second, "water_bottle_position"),
            rtol=0.0,
            atol=1e-7,
        )
        action = _slice(second, "joint_targets")
        outcome = core.chunk_step([0], [np.repeat(action[None, :], 10, axis=0)])[0]
        assert outcome.executed_horizon == 10
        assert outcome.terminated is False
        assert outcome.truncated is False
        assert outcome.success is None
        assert outcome.observation is not None
        assert np.isfinite(outcome.observation.state).all()
        assert outcome.observation.extras["provider"] == "rebot_g1d"
        assert outcome.observation.extras["action_names"] == list(
            REBOT_G1D_ACTION_NAMES
        )
        assert outcome.observation.main_image is not None
        image = payload_module.decode_image(outcome.observation.main_image)
        assert image.shape == (240, 320, 3)
        assert image.dtype == np.uint8
    finally:
        core.close()
    assert not child.is_alive()
