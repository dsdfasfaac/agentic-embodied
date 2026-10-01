# Copyright (c) 2026 Zetta Contributors
"""Constrained planner tools for the ARX Zeva episode."""

from __future__ import annotations

from typing import Any

from robots.arx.task_adapter import ArxAgenticTaskAdapter, ArxAgenticTaskConfig
from zetta.tools.contracts import READ_ONLY_CONTRACT, ToolContract
from zetta.tools.toolkit import Toolkit


class ArxToolkit(Toolkit):
    def __init__(self, *, primitives_kwargs: dict[str, Any], dashboard: Any = None, task_adapter: Any = None) -> None:
        super().__init__(dashboard=dashboard)
        self._adapter = task_adapter or ArxAgenticTaskAdapter(ArxAgenticTaskConfig(**primitives_kwargs))
        self.add_tool("observe_arx_scene", {
            "name": "observe_arx_scene", "description": "Observe all three calibrated ARX views before motion.",
            "input_schema": {"type": "object", "properties": {}},
        }, self._adapter.observe, contract=READ_ONLY_CONTRACT)
        self.add_tool("run_zeva_policy", {
            "name": "run_zeva_policy", "description": "Run the immutable Zeva policy until the Runtime-owned terminal verdict.",
            "input_schema": {"type": "object", "properties": {}},
        }, self._adapter.run, contract=ToolContract(
            capabilities=("motion", "grasp", "simulation"), risk_level="medium",
            irreversible=False, requires_reobservation=True,
            requirements=("prior observe_arx_scene", "hash-pinned scene and policy contracts", "Runtime evaluator"),
        ))
        # Replace generic finish: a planner cannot assert success against Runtime.
        self.add_tool("finish", self._tools["finish"][0], self._finish, contract=self._contracts["finish"])
        self.retain_tools({"describe_tools", "observe_arx_scene", "run_zeva_policy", "finish"})

    def _finish(self, status: str, summary: str) -> dict[str, Any]:
        if status.lower() == "success" and self._adapter.runtime_success is not True:
            return {"error": "success rejected: Runtime has not validated ARX task success"}
        return {"_finish": True, "status": status, "summary": summary, "runtime_success": self._adapter.runtime_success}

    def close(self) -> None:
        self._adapter.close()

    def write_recipe(self, recipe_tag: str) -> str | None:
        del recipe_tag
        return str(self._adapter.config.output_dir / "agentic-summary.json")
