# Copyright (c) 2026 Zetta Contributors
"""Zetta environment plugin for the reBot G1-D MuJoCo tasks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.mujoco.prompt_bundle import system_prompt, user_prompt
from zetta.envs.env_spec import EnvSpec, RunConfig
from zetta.envs.prompt_bundle import PromptBundle
from zetta.utils.config import get_repo_root


def get_env_spec() -> EnvSpec:
    """Return the MuJoCo prompt bundle and CLI lifecycle hooks."""
    return EnvSpec(
        name="mujoco",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Build the constrained MuJoCo planner toolkit."""
    from robots.mujoco.toolkit import MujocoToolkit

    return MujocoToolkit(
        primitives_kwargs=primitives_kwargs,
        video_path=video_path,
        dashboard=dashboard,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    if use_dashboard:
        raise ValueError("the MuJoCo Agentic experiment does not support --dashboard")
    parser.add_argument("--task", choices=("grasp", "fallen"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--asset-manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=2500)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--runtime-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--runtime-endpoint",
        default=None,
        help="Use an existing Runtime HTTP endpoint instead of launching locally.",
    )


def _instruction(task: str) -> str:
    if task == "grasp":
        return "Grasp the upright water bottle and return it safely to ready."
    return "Pick up the fallen water bottle and deposit it inside the bin."


def _parse_config(args: argparse.Namespace) -> RunConfig:
    args.asset_root = args.asset_root.expanduser()
    if args.runtime_endpoint is not None:
        args.runtime_endpoint = str(args.runtime_endpoint).strip()
        if not args.runtime_endpoint:
            raise ValueError("--runtime-endpoint must not be blank")
    if args.runtime_endpoint is None:
        args.asset_root = args.asset_root.resolve(strict=True)
    elif not args.asset_root.is_absolute():
        raise ValueError("--asset-root must be absolute for a remote Runtime")
    digest = str(args.asset_manifest_sha256)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("--asset-manifest-sha256 must be 64 lowercase hex")
    for name in (
        "image_width",
        "image_height",
        "chunk_size",
        "max_episode_steps",
        "video_stride",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.video_fps <= 0.0 or args.runtime_timeout_s <= 0.0:
        raise ValueError("--video-fps and --runtime-timeout-s must be positive")
    if args.task_memory_snapshot or args.episode_memory_frozen:
        raise ValueError("task memory snapshots are not enabled for MuJoCo yet")

    recipe_tag = f"rebot_{args.task}_s{args.seed}"
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{recipe_tag}"
    output_dir = Path(output_dir).expanduser().resolve()
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={
            "task": args.task,
            "seed": args.seed,
            "asset_manifest_sha256": digest,
            "instruction": _instruction(args.task),
        },
        dashboard_state=None,
        task_desc={
            "environment": "mujoco",
            "robot": "reBot G1-D + Dex1-1",
            "task": args.task,
            "seed": args.seed,
            "asset_manifest_sha256": digest,
        },
    )


def _init_runtime(
    args: argparse.Namespace, output_dir: Path
) -> tuple[list[Any], dict[str, Any]]:
    camera = args.camera
    if camera is None:
        camera = "overview" if args.task == "grasp" else "layout_overview"
    return [], {
        "task": args.task,
        "asset_root": args.asset_root,
        "asset_manifest_sha256": str(args.asset_manifest_sha256),
        "seed": int(args.seed),
        "output_dir": output_dir,
        "video_path": output_dir / "episode.mp4",
        "camera_name": str(camera),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "chunk_size": int(args.chunk_size),
        "max_episode_steps": int(args.max_episode_steps),
        "video_fps": float(args.video_fps),
        "video_stride": int(args.video_stride),
        "timeout_s": float(args.runtime_timeout_s),
        "runtime_endpoint": args.runtime_endpoint,
    }


__all__ = ["get_env_spec", "get_toolkit"]
