# Copyright (c) 2026 Zetta Contributors
"""MuJoCo execution tests using the authoritative converted AC one model."""

from __future__ import annotations

from pathlib import Path
import json

import mujoco
import numpy as np
import pytest

from robots.arx.environment import ArxMujocoEnv

_ROOT = Path(__file__).resolve().parents[1]
_ZEVA_ASSET = _ROOT.parent / "Zeva_arx/assets/ac_one"


@pytest.mark.parametrize('locked', [True, False])
def test_absent_left_channels_require_lock(tmp_path: Path, locked: bool) -> None:
    root = _prepared(tmp_path)
    mapping = json.loads((_ZEVA_ASSET / 'ac_one_14d_mapping.json').read_text())
    mapping['joint_names'][:7] = [[] for _ in range(7)]
    mapping['actuator_names'][:7] = [[] for _ in range(7)]
    (root / 'mapping.json').write_text(json.dumps(mapping))
    task = json.loads((_ROOT / 'robots/arx/manifests/pickup_test_tube.yaml').read_text())
    task['control']['lock_left_arm'] = locked
    (root / 'task.yaml').write_text(json.dumps(task))
    kwargs = dict(prepared_scene_bundle=str(root), mapping_path=str(root/'mapping.json'),
                  task_manifest=str(root/'task.yaml'))
    if not locked:
        with pytest.raises(ValueError, match='task-locked left arm'):
            ArxMujocoEnv(**kwargs)
        return
    env = ArxMujocoEnv(**kwargs)
    try:
        obs, _ = env.reset()
        np.testing.assert_allclose(obs['state'][:7], task['start_state'][:7])
        assert np.isfinite(obs['state']).all()
    finally:
        env.close()


def _prepared(tmp_path: Path) -> Path:
    model = mujoco.MjModel.from_xml_path(str(_ZEVA_ASSET / "ac_one_14d.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_saveModel(model, str(tmp_path / "model.mjb"))
    np.savez(tmp_path / "reset_state.npz", qpos=data.qpos, qvel=data.qvel, time=data.time)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"logical_body_map": {"tube_01": "base_link"}}), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.skipif(not _ZEVA_ASSET.exists(), reason="../Zeva_arx checkout is required")
def test_reset_state_mapping_and_exact_policy_duration(tmp_path: Path) -> None:
    env = ArxMujocoEnv(
        prepared_scene_bundle=str(_prepared(tmp_path)),
        mapping_path=str(_ZEVA_ASSET / "ac_one_14d_mapping.json"),
        task_manifest=str(_ROOT / "robots/arx/manifests/pickup_test_tube.yaml"),
    )
    try:
        observation, info = env.reset(seed=3)
        assert observation["state"].shape == (14,)
        assert info["task"] == "pickup_test_tube"
        before = env.data.time
        observation, _, terminated, truncated, _ = env.step(observation["state"])
        assert env.data.time - before == pytest.approx(1.0 / 15.0, abs=1e-12)
        assert not terminated and not truncated
        assert np.isfinite(observation["state"]).all()
    finally:
        env.close()


@pytest.mark.skipif(not _ZEVA_ASSET.exists(), reason="../Zeva_arx checkout is required")
def test_invalid_action_does_not_advance_physics(tmp_path: Path) -> None:
    env = ArxMujocoEnv(
        prepared_scene_bundle=str(_prepared(tmp_path)),
        mapping_path=str(_ZEVA_ASSET / "ac_one_14d_mapping.json"),
        task_manifest=str(_ROOT / "robots/arx/manifests/pickup_test_tube.yaml"),
    )
    try:
        observation, _ = env.reset()
        before = env.data.time
        bad = observation["state"].copy()
        bad[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            env.step(bad)
        assert env.data.time == before
    finally:
        env.close()


@pytest.mark.skipif(not _ZEVA_ASSET.exists(), reason="../Zeva_arx checkout is required")
def test_gripper_observation_projects_solver_overshoot_to_hardware_domain(
    tmp_path: Path,
) -> None:
    env = ArxMujocoEnv(
        prepared_scene_bundle=str(_prepared(tmp_path)),
        mapping_path=str(_ZEVA_ASSET / "ac_one_14d_mapping.json"),
        task_manifest=str(_ROOT / "robots/arx/manifests/pickup_test_tube.yaml"),
    )
    try:
        env.reset()
        for joint_id in env._joint_ids[13]:
            env.data.qpos[env.model.jnt_qposadr[joint_id]] = 0.0442
        state = env._state()
        assert state[13] == pytest.approx(-3.4)
    finally:
        env.close()
