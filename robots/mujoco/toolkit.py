# Copyright (c) 2026 Zetta Contributors
"""Planner-visible tools for the reBot MuJoCo Agentic experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robots.mujoco.task_adapter import RebotAgenticTaskAdapter, RebotTaskConfig
from zetta.tools.contracts import READ_ONLY_CONTRACT, ToolContract
from zetta.tools.toolkit import Toolkit

__all__ = ["MujocoToolkit"]


class MujocoToolkit(Toolkit):
    """Expose observation and one audited task skill to a Zetta planner."""

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        video_path: str | None = None,
        dashboard: Any = None,
        task_adapter: Any | None = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        config = dict(primitives_kwargs)
        if video_path is not None:
            config["video_path"] = Path(video_path)
        self._adapter = task_adapter or RebotAgenticTaskAdapter(
            RebotTaskConfig(**config)
        )
        self.add_tool(
            "observe_rebot_scene",
            {
                "name": "observe_rebot_scene",
                "description": (
                    "Observe the initialized reBot MuJoCo scene without changing it."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            self._adapter.observe,
            contract=READ_ONLY_CONTRACT,
        )
        self.add_tool(
            "run_rebot_skill",
            {
                "name": "run_rebot_skill",
                "description": (
                    "Execute the configured reBot task through Gateway policy_step; "
                    "returns the independent Runtime verdict and MP4 audit."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            self._adapter.run,
            contract=ToolContract(
                capabilities=("motion", "grasp", "simulation"),
                risk_level="medium",
                irreversible=False,
                requires_reobservation=True,
                requirements=(
                    "hash-pinned reBot assets",
                    "independent Runtime success evaluator",
                ),
                notes=(
                    "The skill is one-shot per episode and all physical state "
                    "changes pass through RuntimeGateway.policy_step."
                ),
            ),
        )
        self.retain_tools(
            {
                "describe_tools",
                "finish",
                "observe_rebot_scene",
                "run_rebot_skill",
            }
        )

    def close(self) -> None:
        self._adapter.close()

    def write_recipe(self, recipe_tag: str) -> str | None:
        del recipe_tag
        path = getattr(self._adapter.config, "output_dir", None)
        return None if path is None else str(path / "agentic-summary.json")
