# Copyright (c) 2026 Zetta Contributors
"""Simulator-free Runtime observation wiring for the ARX provider."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rollout_runtime.backends.mujoco_env import MujocoEnvConfig, MujocoEnvCore, _MujocoSlot
from rollout_runtime.core import payload as payload_module


class _Session:
    descriptor = {"action_names": [f"channel_{index}" for index in range(14)]}


def _config(tmp_path: Path, **overrides):
    for name in ("prepared", "mapping.json", "task.yaml"):
        path = tmp_path / name
        path.mkdir() if name == "prepared" else path.write_text("{}")
    value = {
        "provider": "arx_ac_one",
        "env_id": "ZettaArxManipulation-v0",
        "arx_prepared_scene_bundle": str(tmp_path / "prepared"),
        "arx_mapping_path": str(tmp_path / "mapping.json"),
        "arx_task_manifest": str(tmp_path / "task.yaml"),
        "camera_names": {"front_rgb": "front", "left_rgb": "left", "right_rgb": "right"},
        "observation_mode": "rgb_state",
        "render_mode": "rgb_array",
        "action_dim": 14,
        "chunk_size": 32,
        "image_width": 320,
        "image_height": 240,
        "instruction": "Pick up test tube with the pink label.",
        "success_mode": "info_key",
        "success_info_key": "is_success",
    }
    value.update(overrides)
    return MujocoEnvConfig.from_mapping(value)


def test_arx_runtime_encodes_three_semantic_views(tmp_path: Path) -> None:
    core = MujocoEnvCore()
    core.config = _config(tmp_path)
    core._slots = [_MujocoSlot(session=_Session(), default_seed=0, instruction=core.config.instruction)]
    raw = {
        "state": np.arange(14, dtype=np.float32),
        "front_rgb": np.full((240, 320, 3), 1, np.uint8),
        "left_rgb": np.full((240, 320, 3), 2, np.uint8),
        "right_rgb": np.full((240, 320, 3), 3, np.uint8),
    }
    observation = core._make_observation(0, raw, raw["front_rgb"])
    assert observation.state == list(range(14))
    assert int(payload_module.decode_image(observation.main_image)[0, 0, 0]) == 1
    assert int(payload_module.decode_image(observation.wrist_image)[0, 0, 0]) == 2
    assert int(payload_module.decode_image(observation.extra_view_images[0])[0, 0, 0]) == 3
    assert observation.extras["camera_names"]["right_rgb"] == "extra_view_images.0"


@pytest.mark.parametrize(
    "override",
    [
        {"action_dim": 13},
        {"camera_names": {"front_rgb": "same", "left_rgb": "same", "right_rgb": "right"}},
        {"observation_mode": "state", "render_mode": None},
    ],
)
def test_arx_runtime_config_fails_closed(tmp_path: Path, override: dict) -> None:
    with pytest.raises(Exception):
        _config(tmp_path, **override)
