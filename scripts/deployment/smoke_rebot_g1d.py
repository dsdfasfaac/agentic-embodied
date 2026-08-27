#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run a real reBot G1-D scene through the MuJoCo Runtime core."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
from rollout_runtime.backends.mujoco_env import MujocoEnvCore
from rollout_runtime.core import payload as payload_module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--scene-config", default="config/g1d_mujoco.yaml")
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-output", type=Path)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=float, default=25.0)
    return parser.parse_args()


def _joint_targets(observation: Any) -> np.ndarray:
    layout = observation.extras.get("observation_layout", ())
    for item in layout:
        if item.get("path") == "joint_targets":
            return np.asarray(
                observation.state[int(item["start"]) : int(item["stop"])],
                dtype=np.float32,
            )
    raise RuntimeError("reBot observation has no joint_targets state slice")


def main() -> int:
    args = _parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not np.isfinite(args.video_fps) or args.video_fps <= 0.0:
        raise ValueError("--video-fps must be finite and positive")
    spec = EnvSpecMsg(
        env_family="mujoco",
        env_config={
            "provider": "rebot_g1d",
            "env_id": "reBot-DevArm-Grasp-v0",
            "asset_root": str(args.asset_root.expanduser().resolve(strict=True)),
            "asset_manifest_sha256": args.manifest_sha256,
            "rebot_scene_config": args.scene_config,
            "rebot_randomize_bottle": True,
            "observation_mode": "rgb_state",
            "render_mode": "rgb_array",
            "render_backend": "egl",
            "camera_name": args.camera,
            "image_width": args.width,
            "image_height": args.height,
            "action_dim": 22,
            "chunk_size": args.steps,
            "clip_actions": True,
            "max_episode_steps": args.steps + 1,
            "process_isolation": True,
            "rpc_timeout_s": 180.0,
            "instruction": "Control G1-D and its Dex1-1 grippers in the grasp scene",
            "success_mode": "none",
            "return_all_frames": args.video_output is not None,
        },
        resource_hints={"accelerator": True},
    )
    core = MujocoEnvCore()
    child = None
    try:
        core.build(spec, num_envs=1, seed_offset=args.seed)
        child = core._slots[0].session._process
        initial = core.reset([0], ResetSpec(seed=args.seed))[0]
        action = _joint_targets(initial)
        outcome = core.chunk_step(
            [0], [np.repeat(action[None, :], args.steps, axis=0)]
        )[0]
        final = outcome.observation
        if final is None or final.main_image is None:
            raise RuntimeError("reBot smoke produced no final RGB observation")
        image = payload_module.decode_image(final.main_image)
        if args.frame_output is not None:
            args.frame_output.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(args.frame_output, image)
        video_details: dict[str, Any] = {}
        if args.video_output is not None:
            if initial.main_image is None or outcome.per_step is None:
                raise RuntimeError("reBot smoke produced no per-step RGB frames")
            video_frames = [payload_module.decode_image(initial.main_image)]
            for record in outcome.per_step:
                observation = record.observation
                if observation is None or observation.main_image is None:
                    raise RuntimeError("reBot smoke has a missing per-step RGB frame")
                video_frames.append(payload_module.decode_image(observation.main_image))
            args.video_output.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(
                args.video_output,
                np.stack(video_frames),
                plugin="FFMPEG",
                fps=float(args.video_fps),
                codec="libx264",
                pixelformat="yuv420p",
            )
            video_details = {
                "video_frame_count": len(video_frames),
                "video_fps": float(args.video_fps),
                "video_sha256": hashlib.sha256(
                    args.video_output.read_bytes()
                ).hexdigest(),
            }
        result = {
            "provider": "rebot_g1d",
            "env_id": "reBot-DevArm-Grasp-v0",
            "scene_config": args.scene_config,
            "asset_manifest_sha256": args.manifest_sha256,
            "seed": args.seed,
            "requested_steps": args.steps,
            "executed_steps": outcome.executed_horizon,
            "reward": outcome.reward,
            "terminated": outcome.terminated,
            "truncated": outcome.truncated,
            "action_dim": int(action.size),
            "action_names": initial.extras.get("action_names", []),
            "state_dim": len(final.state),
            "rgb_shape": list(image.shape),
            "rgb_dtype": str(image.dtype),
            "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "child_pid": child.pid,
            **video_details,
        }
    finally:
        core.close()
    result["child_reclaimed"] = child is not None and not child.is_alive()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
