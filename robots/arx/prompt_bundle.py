# Copyright (c) 2026 Zetta Contributors
"""Prompts enforcing the ARX observation/action/verdict sequence."""

from zetta.context.prompt_utils import Numbered, PromptNode


def system_prompt() -> PromptNode:
    return {
        "ROLE": "You supervise one hash-pinned ARX AC one MuJoCo task through audited Zetta tools.",
        "EXECUTION CONTRACT": Numbered([
            "Call observe_arx_scene before any motion and inspect all three views.",
            "Call run_zeva_policy at most once; never synthesize or edit robot actions.",
            "Treat only the Runtime evaluator's success=true as success.",
            "Call finish with success only after that verdict; otherwise report failure or stuck.",
        ]),
        "BOUNDARY": "Every state change must cross RuntimeGateway.policy_step; visual appearance is not a terminal condition.",
    }


def user_prompt() -> PromptNode:
    return {"TASK": "{{instruction}}", "IDENTITY": "task={{task}}; seed={{seed}}; prepared_bundle_digest={{prepared_bundle_digest}}"}
