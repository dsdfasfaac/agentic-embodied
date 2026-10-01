# Copyright (c) 2026 Zetta Contributors
"""ARX AC one contracts and Zetta environment extension hooks."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.arx.contracts import (
    ArxCameraSpec,
    ArxModelContract,
    ArxTaskManifest,
    load_model_contract,
    load_task_manifest,
)
from robots.arx.prompt_bundle import system_prompt, user_prompt
from zetta.envs.env_spec import EnvSpec, RunConfig
from zetta.envs.prompt_bundle import PromptBundle
from zetta.utils.config import get_repo_root


def get_env_spec() -> EnvSpec:
    return EnvSpec(
        name="arx",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
    )


def get_toolkit(*, primitives_kwargs: dict[str, Any], video_path: str | None = None, dashboard: Any = None):
    del video_path
    from robots.arx.toolkit import ArxToolkit

    return ArxToolkit(primitives_kwargs=primitives_kwargs, dashboard=dashboard)


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    if use_dashboard:
        raise ValueError("the ARX Agentic experiment does not support --dashboard")
    parser.add_argument("--prepared-scene-bundle", type=Path, required=True)
    parser.add_argument("--prepared-bundle-digest", required=True)
    parser.add_argument("--arx-mapping", type=Path, required=True)
    parser.add_argument("--arx-task-manifest", type=Path, required=True)
    parser.add_argument("--arx-model-contract", type=Path, required=True)
    parser.add_argument("--front-camera", required=True)
    parser.add_argument("--left-camera", required=True)
    parser.add_argument("--right-camera", required=True)
    parser.add_argument("--zeva-host", default="127.0.0.1")
    parser.add_argument("--zeva-port", type=int, default=5581)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--runtime-timeout-s", type=float, default=600.0)
    parser.add_argument("--render-backend", choices=("egl", "glfw", "osmesa"), default="egl")
    parser.add_argument("--runtime-endpoint", default=None)


def _config_from_args(args: argparse.Namespace, output_dir: Path) -> ArxAgenticTaskConfig:
    from robots.arx.task_adapter import ArxAgenticTaskConfig
    endpoint = None if args.runtime_endpoint is None else str(args.runtime_endpoint).strip()
    if args.runtime_endpoint is not None and not endpoint:
        raise ValueError("--runtime-endpoint must not be blank")
    return ArxAgenticTaskConfig(
        prepared_scene_bundle=args.prepared_scene_bundle.expanduser().resolve(strict=True),
        prepared_bundle_digest=str(args.prepared_bundle_digest),
        mapping_path=args.arx_mapping.expanduser().resolve(strict=True),
        task_manifest=args.arx_task_manifest.expanduser().resolve(strict=True),
        model_contract=args.arx_model_contract.expanduser().resolve(strict=True),
        camera_names={
            "front_rgb": str(args.front_camera),
            "left_rgb": str(args.left_camera),
            "right_rgb": str(args.right_camera),
        },
        output_dir=output_dir,
        seed=int(args.seed),
        zeva_host=str(args.zeva_host),
        zeva_port=int(args.zeva_port),
        timeout_s=float(args.runtime_timeout_s),
        render_backend=str(args.render_backend),
        runtime_endpoint=endpoint,
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if args.task_memory_snapshot or args.episode_memory_frozen:
        raise ValueError("task memory snapshots are not enabled for ARX")
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_arx_task7_s{args.seed}"
    output_dir = Path(output_dir).expanduser().resolve()
    config = _config_from_args(args, output_dir)
    task = config.task
    return RunConfig(
        recipe_tag=f"arx_{task.name}_s{config.seed}",
        output_dir=output_dir,
        prompt_vars={
            "task": task.name,
            "seed": config.seed,
            "instruction": task.instruction,
            "prepared_bundle_digest": config.prepared_bundle_digest,
        },
        dashboard_state=None,
        task_desc={
            "environment": "mujoco",
            "robot": "ARX AC one bilateral",
            "policy": "Zeva Cosmos3-Edge Task7",
            "task": task.name,
            "seed": config.seed,
            "prepared_bundle_digest": config.prepared_bundle_digest,
        },
    )


def _init_runtime(args: argparse.Namespace, output_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    config = _config_from_args(args, output_dir)
    return [], {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}

__all__ = [
    "ArxCameraSpec",
    "ArxModelContract",
    "ArxTaskManifest",
    "load_model_contract",
    "load_task_manifest",
    "get_env_spec",
    "get_toolkit",
]
