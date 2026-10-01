# Copyright (c) 2026 Zetta Contributors
"""Semantic invariants for the Zetta-facing ARX agent tools."""

from __future__ import annotations

from pathlib import Path

from robots.arx.toolkit import ArxToolkit
from zetta.envs.base import get_env_spec


class _Adapter:
    def __init__(self) -> None:
        self.runtime_success = None
        self.observed = False
        self.runs = 0
        self.config = type("Config", (), {"output_dir": Path("/tmp/arx-test")})()

    def observe(self):
        self.observed = True
        return {"step_index": 0, "_image_bytes": b"png"}

    def run(self):
        if not self.observed:
            raise RuntimeError("observe_arx_scene must be called before motion")
        self.runs += 1
        if self.runs > 1:
            raise RuntimeError("the ARX episode has already been executed")
        self.runtime_success = True
        return {"success": True, "terminal_reason": "success"}

    def close(self):
        pass


def test_planner_has_no_direct_action_or_evaluator_tools() -> None:
    toolkit = ArxToolkit(primitives_kwargs={}, task_adapter=_Adapter())
    names = {item["name"] for item in toolkit.get_tools_spec()}
    assert names == {"describe_tools", "observe_arx_scene", "run_zeva_policy", "finish"}
    assert not {"step", "action_step", "set_evaluator", "write_text_file"} & names
    motion = next(item for item in toolkit.get_tools_spec() if item["name"] == "run_zeva_policy")
    assert motion["input_schema"] == {"type": "object", "properties": {}}


def test_arx_is_a_resolvable_zetta_environment() -> None:
    spec = get_env_spec("arx")
    assert spec.name == "arx"
    assert callable(spec.add_cli_args)
    assert callable(spec.parse_config)
    assert callable(spec.init_runtime)


def test_observation_precedes_motion_and_runtime_gates_finish() -> None:
    adapter = _Adapter()
    toolkit = ArxToolkit(primitives_kwargs={}, task_adapter=adapter)
    premature_motion = toolkit.execute_tool("run_zeva_policy", {}).result
    assert "observe_arx_scene" in premature_motion["error"]
    premature_finish = toolkit.execute_tool("finish", {"status": "success", "summary": "looks done"})
    assert not premature_finish.is_finish
    assert "Runtime" in premature_finish.result["error"]
    toolkit.execute_tool("observe_arx_scene", {})
    result = toolkit.execute_tool("run_zeva_policy", {}).result
    assert result["success"] is True
    finish = toolkit.execute_tool("finish", {"status": "success", "summary": "Runtime passed"})
    assert finish.is_finish and finish.result["runtime_success"] is True


def test_failure_finish_remains_available_without_motion() -> None:
    toolkit = ArxToolkit(primitives_kwargs={}, task_adapter=_Adapter())
    finish = toolkit.execute_tool("finish", {"status": "failure", "summary": "camera unavailable"})
    assert finish.is_finish
    assert finish.result["runtime_success"] is None
