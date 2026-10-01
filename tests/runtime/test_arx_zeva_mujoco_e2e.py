"""End-to-end Zetta Runtime + ARX MuJoCo test with a deterministic fake Zeva.

This exercises the real ARX environment/session, Runtime gateway, observation
packing, policy backend dispatch, and terminal plumbing. It substitutes only
the unavailable remote Cosmos process with the repository's fake policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg, PolicyRequest, ResetSpec
from rollout_runtime.api.result import unwrap
from rollout_runtime.launch.local import build_local_components
from rollout_runtime.backends.fake.policy import FakePolicyCore


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "runs/arx_pickup_test_tube/dark_silver_estimated_top_scene"
MAPPING = ROOT.parent / "Zeva_arx/assets/ac_one/ac_one_14d_mapping.json"
TASK = ROOT / "robots/arx/manifests/pickup_test_tube.yaml"


@pytest.mark.skipif(not SCENE.is_dir() or not MAPPING.is_file(), reason="prepared ARX scene/Zeva assets unavailable")
@pytest.mark.asyncio
async def test_zeva_mujoco_runtime_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the stand-in policy inside the ARX gripper domain. The real Zeva
    # client is tested separately; this test validates the Runtime/MuJoCo seam.
    monkeypatch.setattr(
        FakePolicyCore, "_actions_for",
        lambda self, request: np.zeros((32, 14), dtype=np.float32),
    )
    config = {
        "env_family": "mujoco",
        "env_config": {
            "provider": "arx_ac_one", "env_id": "ZettaArxManipulation-v0",
            "arx_prepared_scene_bundle": str(SCENE), "arx_mapping_path": str(MAPPING),
            "arx_task_manifest": str(TASK),
            "camera_names": {"front_rgb": "front_rgb", "left_rgb": "left_rgb", "right_rgb": "right_rgb"},
            "observation_mode": "rgb_state", "render_mode": "rgb_array",
            "render_backend": "egl", "image_width": 320, "image_height": 240,
            "action_dim": 14, "chunk_size": 32, "clip_actions": False,
            "max_episode_steps": 32, "core_form": "per_slot", "process_isolation": True,
            "rpc_timeout_s": 60.0, "instruction": "Pick up test tube with the pink label.",
            "success_mode": "info_key", "success_info_key": "is_success", "return_all_frames": True,
        },
        "transport": {"kind": "inproc", "command_timeout_seconds": 60.0},
        "env_worker": {"num_ranks": 1, "max_sessions_per_rank": 1, "has_accelerator": False},
        "rollout_worker": {"num_ranks": 1, "policy_id": "fake", "policy_family": "fake",
            "policy_backend": "fake", "device": "cpu", "dtype": "float32",
            "policy_config": {}, "scheduler": {"max_batch_size": 1, "max_wait_ms": 0.0}},
    }
    runtime = build_local_components(config)
    await runtime.start()
    try:
        gateway = runtime.gateway
        handle = unwrap((await gateway.create_sessions([CreateSessionRequest(
            application_id="arx-e2e-test", client_session_key="arx-e2e-17",
            env_spec=EnvSpecMsg(env_family="mujoco", env_config=config["env_config"], pool_size=1),
            default_policy_id="fake", lease_seconds=120.0,
        )]))[0])
        reset = unwrap((await gateway.reset([handle.session_id], ResetSpec(seed=17)))[0])
        assert reset.observation is not None
        assert reset.observation.step_index == 0
        assert reset.observation.state and len(reset.observation.state) == 14
        assert len(reset.observation.extra_view_images) == 1
        assert reset.observation.extras["camera_names"] == {
            "front_rgb": "main_image", "left_rgb": "wrist_image", "right_rgb": "extra_view_images.0"
        }

        result = unwrap((await gateway.policy_step(
            [handle.session_id], PolicyRequest(policy_id="fake", actions_per_chunk=32)
        ))[0])
        assert result.observation is not None
        assert result.executed_horizon == 32
        assert result.observation.step_index == 32
        assert result.per_step is not None and len(result.per_step) == 32
        assert all(record.info is not None for record in result.per_step)
        assert result.terminated is False
        assert result.truncated is False
        assert result.success is False
    finally:
        await runtime.gateway.stop()
        await runtime.aclose()
