# Copyright (c) 2026 Zetta Contributors
"""Simulator-free tests for the planner-to-Runtime reBot integration."""

from __future__ import annotations

from types import SimpleNamespace

from robots.mujoco import _init_runtime, get_env_spec
from robots.mujoco.task_adapter import RebotAgenticTaskAdapter, RebotTaskConfig
from robots.mujoco.toolkit import MujocoToolkit


class _FakeTaskAdapter:
    def __init__(self, output_dir) -> None:
        self.config = SimpleNamespace(output_dir=output_dir)
        self.closed = False
        self.runs = 0

    def observe(self):
        return {
            "task": "grasp",
            "step_index": 0,
            "_image_bytes": b"png",
        }

    def run(self):
        self.runs += 1
        return {
            "success": True,
            "execution_path": [
                "zetta.planner",
                "RuntimeGateway.policy_step",
                "RebotG1DSession",
            ],
            "_image_bytes": b"final-png",
        }

    def close(self) -> None:
        self.closed = True


def test_mujoco_toolkit_exposes_only_agentic_task_surface(tmp_path) -> None:
    adapter = _FakeTaskAdapter(tmp_path)
    toolkit = MujocoToolkit(
        primitives_kwargs={},
        task_adapter=adapter,
    )
    names = {spec["name"] for spec in toolkit.get_tools_spec()}
    assert names == {
        "describe_tools",
        "finish",
        "observe_rebot_scene",
        "run_rebot_skill",
    }
    observed = toolkit.execute_tool("observe_rebot_scene", {})
    assert observed.result["step_index"] == 0
    assert any(block["type"] == "image" for block in observed.content_blocks)
    executed = toolkit.execute_tool("run_rebot_skill", {})
    assert executed.result["success"] is True
    assert "RuntimeGateway.policy_step" in executed.result["execution_path"]
    assert adapter.runs == 1
    toolkit.close()
    assert adapter.closed is True


def test_mujoco_env_prompt_requires_runtime_verdict() -> None:
    spec = get_env_spec()
    assert spec.name == "mujoco"
    system = spec.prompts.render(
        "system",
        variables={
            "task": "grasp",
            "seed": 17,
            "asset_manifest_sha256": "a" * 64,
            "instruction": "grasp",
        },
    )
    assert "Runtime success=true" in system
    assert "Gateway policy_step" in system
    user = spec.prompts.render(
        "user",
        variables={
            "task": "grasp",
            "seed": 17,
            "asset_manifest_sha256": "a" * 64,
            "instruction": "grasp the bottle",
        },
    )
    assert "grasp the bottle" in user
    assert "task=grasp; seed=17" in user
    assert "{{" not in user


def test_agentic_runtime_config_wires_skill_and_independent_success(tmp_path) -> None:
    config = RebotTaskConfig(
        task="fallen",
        asset_root=tmp_path.resolve(),
        asset_manifest_sha256="a" * 64,
        seed=17,
        output_dir=tmp_path / "run",
        video_path=tmp_path / "run/episode.mp4",
        camera_name="layout_overview",
    )
    adapter = object.__new__(RebotAgenticTaskAdapter)
    adapter.config = config
    runtime = adapter._runtime_config()
    assert runtime["rollout_worker"]["policy_backend"] == "rebot_g1d_skill"
    assert runtime["rollout_worker"]["policy_id"] == "rebot_fallen_skill"
    env = runtime["env_config"]
    assert env["rebot_task"] == "fallen"
    assert env["rebot_scene_config"] == "config/g1d_fallen_bottle_to_bin.yaml"
    assert env["success_mode"] == "info_key"
    assert env["success_info_key"] == "is_success"
    assert env["return_all_frames"] is False


def test_remote_runtime_keeps_server_side_asset_path(tmp_path) -> None:
    remote_asset_root = tmp_path / "not-mounted-on-client"
    args = SimpleNamespace(
        task="grasp",
        asset_root=remote_asset_root,
        asset_manifest_sha256="a" * 64,
        seed=17,
        camera=None,
        image_width=320,
        image_height=240,
        chunk_size=32,
        max_episode_steps=2500,
        video_fps=25.0,
        video_stride=4,
        runtime_timeout_s=900.0,
        runtime_endpoint="http://127.0.0.1:18710",
    )
    _, primitives = _init_runtime(args, tmp_path / "run")
    assert primitives["asset_root"] == remote_asset_root
    assert primitives["runtime_endpoint"] == "http://127.0.0.1:18710"
    assert primitives["image_width"] == 320
    assert primitives["image_height"] == 240
