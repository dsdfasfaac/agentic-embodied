# Copyright (c) 2026 Zetta Contributors
"""Simulator-free contract tests for the pinned reBot G1-D provider."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.ids import EpisodeId, OperationSeq, RequestId, SessionId
from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends import build_policy_core
from rollout_runtime.backends.mujoco_env import MujocoEnvConfig
from rollout_runtime.backends.mujoco_session import MujocoSessionError
from rollout_runtime.backends.rebot_g1d_policy import (
    REBOT_G1D_SKILL_POLICY_FAMILY,
)
from rollout_runtime.backends.rebot_g1d_session import (
    REBOT_G1D_ACTION_NAMES,
    RebotG1DSession,
    compute_rebot_asset_manifest,
)
from rollout_runtime.core import payload as payload_module


def _fake_adapter_source() -> str:
    return f"""import numpy as np

CONTROLLED_JOINTS = {REBOT_G1D_ACTION_NAMES!r}

class JointIndex:
    def __init__(self):
        self.lower = -2.0
        self.upper = 2.0

class G1DMujocoEnv:
    def __init__(self, config, *, headless):
        self.config = config
        self.headless = headless
        self.targets = {{name: 0.0 for name in CONTROLLED_JOINTS}}
        self._joint_indices = {{name: JointIndex() for name in CONTROLLED_JOINTS}}
        self._step = 0
        self._bottle = np.asarray(config["scene"]["bottle_position"], dtype=float)
        self.contacts = []
        self.gripper_positions = {{
            "left": np.asarray([0.35, 0.18, 1.23], dtype=float),
            "right": np.asarray([0.35, -0.18, 1.23], dtype=float),
        }}
        self.last_render_camera = None
        self.closed = False

    def observation(self):
        return {{
            "qpos": np.full((len(CONTROLLED_JOINTS),), self._step, dtype=float),
            "qvel": np.zeros((len(CONTROLLED_JOINTS),), dtype=float),
        }}

    def reset(self):
        self._step = 0
        return self.observation()

    def set_free_body_pose(self, _name, position, _quaternion):
        self._bottle = np.asarray(position, dtype=float).copy()

    def get_body_position(self, _name):
        return self._bottle.copy()

    def set_joint_targets(self, targets):
        self.targets.update(targets)

    def step(self):
        self._step += 1
        return self.observation()

    def contact_pairs(self):
        return list(self.contacts)

    def get_gripper_pose(self, side):
        pose = np.eye(4)
        pose[:3, 3] = self.gripper_positions[side]
        return pose

    def render(self, camera_name=None):
        self.last_render_camera = camera_name
        render = self.config["render"]
        return {{
            "img": np.full(
                (render["height"], render["width"], 3), self._step, dtype=np.uint8
            )
        }}

    def close(self):
        self.closed = True
"""


def _fake_assets(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "assets"
    (root / "simulation").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "models/g1_d_description").mkdir(parents=True)
    (root / "simulation/g1d_mujoco_env.py").write_text(
        _fake_adapter_source(), encoding="utf-8"
    )
    (root / "simulation/g1d_mujoco_env_bin.py").write_text(
        _fake_adapter_source(), encoding="utf-8"
    )
    (root / "simulation/geometric_grasp.py").write_text(
        "# fake geometric detector\n", encoding="utf-8"
    )
    (root / "scripts").mkdir()
    (root / "scripts/sim_geometric_grasp.py").write_text(
        "# fake grasp skill\n", encoding="utf-8"
    )
    (root / "scripts/sim_fallen_bottle_to_bin.py").write_text(
        "# fake fallen skill\n", encoding="utf-8"
    )
    scene = """simulation:
  urdf_path: models/g1_d_description/g1_d_with_dex1_1.urdf
  render:
    width: 64
    height: 48
    camera: overview
  scene:
    bottle_position: [0.5, 0.2, 1.0]
    bottle_xy_jitter_m: [0.01, 0.02]
    table_position: [0.92, 0.0, 1.15]
    table_half_size: [0.42, 0.48, 0.05]
    trash_bin:
      position: [0.28, 0.62, 0.0]
      half_width_m: 0.18
      height_m: 1.0
      wall_thickness_m: 0.018
  motion:
    fixed_ready_tcp_position: [0.35, 0.18, 1.23]
"""
    (root / "config/g1d_mujoco.yaml").write_text(scene, encoding="utf-8")
    (root / "config/g1d_fallen_bottle_to_bin.yaml").write_text(scene, encoding="utf-8")
    (root / "models/g1_d_description/g1_d_with_dex1_1.urdf").write_text(
        '<robot name="fake"><link name="base"/></robot>\n', encoding="utf-8"
    )
    manifest = compute_rebot_asset_manifest(root)
    return root, str(manifest["sha256"])


def _session_config(root: Path, digest: str, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "provider": "rebot_g1d",
        "asset_root": str(root),
        "asset_manifest_sha256": digest,
        "rebot_scene_config": "config/g1d_mujoco.yaml",
        "rebot_randomize_bottle": True,
        "render_mode": "rgb_array",
        "render_backend": "egl",
        "camera_name": "overview",
        "image_width": 80,
        "image_height": 60,
        "max_episode_steps": 2,
    }
    config.update(overrides)
    return config


def test_rebot_config_requires_pinned_absolute_assets(tmp_path: Path) -> None:
    """The dedicated provider cannot become an arbitrary relative entrypoint."""
    digest = "a" * 64
    config = MujocoEnvConfig.from_mapping(
        {
            "provider": "rebot_g1d",
            "env_id": "reBot-DevArm-Grasp-v0",
            "asset_root": str(tmp_path),
            "asset_manifest_sha256": digest,
            "action_dim": 22,
        }
    )
    assert config.resolved_process_isolation is True
    assert config.rebot_scene_config == "config/g1d_mujoco.yaml"

    invalid = [
        {"asset_root": "relative/assets"},
        {"asset_manifest_sha256": "not-a-digest"},
        {"rebot_scene_config": "../scene.yaml"},
        {"action_dim": 21},
        {"process_isolation": False},
        {"env_kwargs": {"entrypoint": "arbitrary.module"}},
        {"rebot_task": "unknown"},
        {"rebot_task": "fallen"},
    ]
    baseline = {
        "provider": "rebot_g1d",
        "env_id": "reBot-DevArm-Grasp-v0",
        "asset_root": str(tmp_path),
        "asset_manifest_sha256": digest,
        "action_dim": 22,
    }
    for override in invalid:
        with pytest.raises(RuntimeApiError) as excinfo:
            MujocoEnvConfig.from_mapping({**baseline, **override})
        assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_asset_manifest_covers_adapter_config_urdf_and_mesh(tmp_path: Path) -> None:
    """Changing any executable or geometric input changes the aggregate pin."""
    root, original = _fake_assets(tmp_path)
    mesh = root / "models/g1_d_description/mesh.stl"
    mesh.write_bytes(b"mesh-v1")
    urdf = root / "models/g1_d_description/g1_d_with_dex1_1.urdf"
    urdf.write_text(
        '<robot name="fake"><link name="base"><visual><geometry>'
        '<mesh filename="mesh.stl"/></geometry></visual></link></robot>\n',
        encoding="utf-8",
    )
    with_mesh = compute_rebot_asset_manifest(root)
    assert with_mesh["file_count"] == 8
    assert with_mesh["sha256"] != original

    mesh.write_bytes(b"mesh-v2")
    changed = compute_rebot_asset_manifest(root)
    assert changed["sha256"] != with_mesh["sha256"]


def test_rebot_session_reset_step_render_and_close(tmp_path: Path) -> None:
    """The bridge exposes deterministic reset, 22 targets, RGB, and truncation."""
    root, digest = _fake_assets(tmp_path)
    session = RebotG1DSession(_session_config(root, digest))
    assert session.descriptor["action_shape"] == (22,)
    assert session.descriptor["action_names"] == REBOT_G1D_ACTION_NAMES

    first, first_info = session.reset(seed=7, options={})
    second, second_info = session.reset(seed=7, options={})
    np.testing.assert_allclose(
        first["water_bottle_position"], second["water_bottle_position"]
    )
    assert first_info["asset_manifest_sha256"] == digest
    assert first_info["bottle_position"] == second_info["bottle_position"]

    action = np.linspace(-1.0, 1.0, 22, dtype=np.float32)
    observation, reward, terminated, truncated, info = session.step(action)
    np.testing.assert_allclose(observation["joint_targets"], action)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["contact_count"] == 0
    frame = session.render()
    assert frame is not None
    assert frame.shape == (60, 80, 3)
    assert session._env.config["render"]["camera"] == "overview"
    assert session._env.last_render_camera == "overview"

    _observation, _reward, _terminated, truncated, _info = session.step(action)
    assert truncated is True
    env = session._env
    session.close()
    session.close()
    assert env.closed is True


def test_rebot_render_camera_does_not_rename_scene_camera(tmp_path: Path) -> None:
    """Selecting the layout view must not create duplicate MJCF camera names."""
    root, digest = _fake_assets(tmp_path)
    session = RebotG1DSession(
        _session_config(root, digest, camera_name="layout_overview")
    )
    try:
        assert session._env.config["render"]["camera"] == "overview"
        assert session.render() is not None
        assert session._env.last_render_camera == "layout_overview"
    finally:
        session.close()


def test_rebot_session_rejects_manifest_mismatch_before_loading(tmp_path: Path) -> None:
    """A modified external adapter is never executed under a stale pin."""
    root, digest = _fake_assets(tmp_path)
    adapter = root / "simulation/g1d_mujoco_env.py"
    adapter.write_text('raise RuntimeError("must not execute")\n', encoding="utf-8")
    with pytest.raises(MujocoSessionError, match="manifest mismatch"):
        RebotG1DSession(_session_config(root, digest))


def test_runtime_independently_latches_grasp_success(tmp_path: Path) -> None:
    """Policy output cannot self-report success; MuJoCo state must satisfy it."""
    root, digest = _fake_assets(tmp_path)
    session = RebotG1DSession(
        _session_config(root, digest, rebot_task="grasp", max_episode_steps=100)
    )
    try:
        session.reset(seed=7, options={})
        session._env.contacts = [
            ("water_bottle_body", "left_dex1_finger_link_1_pad_collision")
        ]
        session.step(np.zeros(22, dtype=np.float32))
        session._env._bottle = np.asarray([0.45, 0.2, 1.21])
        final = None
        for _ in range(25):
            final = session.step(np.zeros(22, dtype=np.float32))
        assert final is not None
        assert final[2] is True
        assert final[4]["is_success"] is True
        assert final[4]["payload_retained"] is True
        assert final[4]["max_bottle_lift_height"] > 0.08
    finally:
        session.close()


def test_runtime_independently_latches_fallen_deposit_success(tmp_path: Path) -> None:
    """A lift followed by stable in-bin release terminates the fallen task."""
    root, _digest = _fake_assets(tmp_path)
    digest = str(
        compute_rebot_asset_manifest(root, "config/g1d_fallen_bottle_to_bin.yaml")[
            "sha256"
        ]
    )
    session = RebotG1DSession(
        _session_config(
            root,
            digest,
            rebot_scene_config="config/g1d_fallen_bottle_to_bin.yaml",
            rebot_task="fallen",
            max_episode_steps=100,
        )
    )
    try:
        session.reset(seed=7, options={})
        session._env.contacts = [
            ("water_bottle_body", "left_dex1_finger_link_1_pad_collision")
        ]
        session.step(np.zeros(22, dtype=np.float32))
        session._env._bottle[2] += 0.12
        session.step(np.zeros(22, dtype=np.float32))
        session._env.contacts = []
        session._env._bottle = np.asarray([0.28, 0.62, 0.08])
        final = None
        for _ in range(25):
            final = session.step(np.zeros(22, dtype=np.float32))
        assert final is not None
        assert final[2] is True
        assert final[4]["is_success"] is True
        assert final[4]["deposited"] is True
        assert final[4]["final_gripper_bottle_contact"] is False
    finally:
        session.close()


def test_rebot_skill_policy_serves_step_aligned_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expert skill enters the normal inference plane as action chunks."""
    root, digest = _fake_assets(tmp_path)
    actions = np.arange(220, dtype=np.float32).reshape(10, 22)
    from rollout_runtime.backends import rebot_g1d_policy as policy_module

    monkeypatch.setattr(
        policy_module,
        "_compile_trajectory",
        lambda _config: (
            actions,
            {
                "task": "grasp",
                "trajectory_steps": 10,
                "trajectory_sha256": "b" * 64,
            },
        ),
    )
    core = build_policy_core(
        backend="rebot_g1d_skill",
        policy_family=REBOT_G1D_SKILL_POLICY_FAMILY,
        action_dim=22,
        actions_per_chunk=4,
        policy_config={
            "asset_root": str(root),
            "asset_manifest_sha256": digest,
            "task": "grasp",
            "scene_config": "config/g1d_mujoco.yaml",
            "seed": 7,
        },
    )
    core.load()
    observation = Observation(
        session_id=SessionId("session"),
        episode_id=EpisodeId(1),
        step_index=3,
        extras={
            "provider": "rebot_g1d",
            "action_names": list(REBOT_G1D_ACTION_NAMES),
        },
    )
    request = InferenceRequest(
        request_id=RequestId("request"),
        session_id=SessionId("session"),
        episode_id=EpisodeId(1),
        operation_seq=OperationSeq(1),
        policy_id="rebot_grasp_skill",
        observation=observation,
        routing_token="env:0",
        compat_key="key",
    )
    response = core.infer_batch([request])[0]
    assert response.error is None
    np.testing.assert_array_equal(
        payload_module.decode_payload(response.actions), actions[3:7]
    )
    assert response.auxiliary_outputs["trajectory_start"] == 3
    assert response.auxiliary_outputs["trajectory_stop"] == 7

    wrong = dataclasses.replace(request, policy_id="wrong")
    failure = core.infer_batch([wrong])[0]
    assert failure.error is not None
    assert failure.error.code is ErrorCode.POLICY_FAILURE


def test_asset_preparer_extracts_only_runtime_scene_files(tmp_path: Path) -> None:
    """The zip preparer excludes weights, outputs, bytecode, and unrelated code."""
    archive = tmp_path / "rebot.zip"
    prefix = "vendor-rebot/"
    config = """simulation:
  urdf_path: models/g1_d_description/g1_d_with_dex1_1.urdf
  scene:
    bottle_position: [0.5, 0.2, 1.0]
"""
    urdf = (
        '<robot name="fake"><link name="base"><visual><geometry>'
        '<mesh filename="meshes/used.stl"/></geometry></visual></link></robot>\n'
    )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(prefix + "simulation/g1d_mujoco_env.py", "# adapter\n")
        bundle.writestr(prefix + "simulation/g1d_mujoco_env_bin.py", "# bin adapter\n")
        bundle.writestr(prefix + "simulation/geometric_grasp.py", "# detector\n")
        bundle.writestr(prefix + "scripts/sim_geometric_grasp.py", "# grasp\n")
        bundle.writestr(prefix + "scripts/sim_fallen_bottle_to_bin.py", "# fallen\n")
        bundle.writestr(prefix + "config/g1d_mujoco.yaml", config)
        bundle.writestr(prefix + "config/g1d_fallen_bottle_to_bin.yaml", config)
        bundle.writestr(prefix + "models/g1_d_description/g1_d_with_dex1_1.urdf", urdf)
        bundle.writestr(prefix + "models/g1_d_description/meshes/used.stl", b"used")
        bundle.writestr(prefix + "models/g1_d_description/meshes/unused.stl", b"unused")
        bundle.writestr(prefix + "models/yolo.pt", b"weight")
        bundle.writestr(prefix + "outputs/episode.npy", b"output")
        bundle.writestr(prefix + "simulation/__pycache__/cached.pyc", b"bytecode")
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    output = tmp_path / "prepared"
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deployment/prepare_rebot_g1d_assets.py",
            str(archive),
            str(output),
            "--expected-archive-sha256",
            archive_digest,
        ],
        cwd=repository,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "asset-manifest.json").read_text())
    assert manifest["selected_file_count"] == 9
    assert (output / "models/g1_d_description/meshes/used.stl").is_file()
    assert not (output / "models/g1_d_description/meshes/unused.stl").exists()
    assert not (output / "models/yolo.pt").exists()
    assert not (output / "outputs").exists()
    assert not (output / "simulation/__pycache__").exists()
    assert (output / "scripts/sim_geometric_grasp.py").is_file()
    assert (output / "scripts/sim_fallen_bottle_to_bin.py").is_file()
