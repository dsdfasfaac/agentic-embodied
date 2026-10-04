"""Compile frozen CandidateBundle recovery steps into bounded gateway calls."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from robots.arx.gateway.contracts import RecoveryBinding
from .real_input import _TOOL_MODELS


@dataclass(frozen=True)
class ProgramCall:
    step_index: int
    call_index: int
    tool: str
    arguments: dict
    stop_when: str


@dataclass(frozen=True)
class RecoveryProgram:
    binding: RecoveryBinding
    calls: tuple[ProgramCall, ...]


def _eef_budget(args):
    distance = math.sqrt(sum(x * x for x in args["delta_xyz_m"]))
    rotation = math.sqrt(sum(x * x for x in args.get("delta_rotvec_rad", (0, 0, 0))))
    # CommandKinematics.plan rotates the vector before taking its NumPy norm.
    # Reserve one action for floating-point rounding at exact 1 cm boundaries.
    actions = max(1, math.ceil(distance / min(.001, args.get("speed_m_s", .01) / 15)),
                  math.ceil(rotation / .01)) + 16
    if actions > 60:
        raise ValueError("EEF call exceeds planner's 60-action horizon")
    return actions


def compile_programs(bundle, *, max_tool_calls=64, max_physical_steps=None,
                     nominal_chunk_steps=16):
    """Symbolic values are resolved only against live review/observation results."""
    if type(nominal_chunk_steps) is not int or nominal_chunk_steps < 1:
        raise ValueError("nominal VLA chunk length must be positive")
    programs = {}
    for rule in bundle.recovery_rules:
        if not rule.fallback.strip():
            raise ValueError(f"missing recovery fallback: {rule.recovery_id}")
        calls, physical = [], 0
        reviewed = False
        resumed = False
        for step_index, step in enumerate(rule.steps):
            if resumed:
                raise ValueError("no recovery step may follow VLA reentry")
            if step.tool not in _TOOL_MODELS or step.tool == "arx.finish":
                raise ValueError(f"unsupported recovery tool: {step.tool}")
            args = deepcopy(step.parameters)
            count = 1
            if step.tool == "arx.move_eef":
                vector = args.get("delta_xyz_m")
                rotation = args.get("delta_rotvec_rad", [0., 0., 0.])
                if not isinstance(vector, list) or len(vector) != 3 or not all(
                    type(v) in (int, float) and math.isfinite(v) for v in vector + rotation
                ):
                    raise ValueError("invalid EEF displacement")
                count = max(1, math.ceil((math.sqrt(sum(v*v for v in vector)) - 1e-12) / .01),
                            math.ceil((math.sqrt(sum(v*v for v in rotation)) - 1e-12) / .1))
                args["delta_xyz_m"] = [v / count for v in vector]
                args["delta_rotvec_rad"] = [v / count for v in rotation]
                physical += count * _eef_budget(args)
                reviewed = False
            elif step.tool == "arx.set_gripper":
                physical += args.get("max_steps", 0)
                reviewed = False
            elif step.tool == "arx.hold":
                physical += args.get("steps", 0)
                reviewed = False
            elif step.tool == "arx.review_reentry":
                if args.get("observation_ids") != ["post-recovery"]:
                    raise ValueError("review must bind the fresh post-recovery observation")
                reviewed = True
                args["observation_ids"] = ["obs-preflight"]
            elif step.tool == "arx.zeva":
                if args.get("reentry_token") not in (None, "token-from-review"):
                    raise ValueError("VLA reentry token must come from a fresh review")
                if not reviewed:
                    # A provisional bundle may omit the read-only review. The
                    # executable program inserts it immediately before VLA.
                    review_args = {"observation_ids": ["obs-preflight"]}
                    _TOOL_MODELS["arx.review_reentry"].model_validate(review_args)
                    calls.append(ProgramCall(
                        step_index, -1, "arx.review_reentry", review_args,
                        "fresh measured reentry assessment is eligible",
                    ))
                    reviewed = True
                args["reentry_token"] = "token-preflight"
                physical += args["max_chunks"] * nominal_chunk_steps
                resumed = True
            _TOOL_MODELS[step.tool].model_validate(args)
            for call_index in range(count):
                calls.append(ProgramCall(step_index, call_index, step.tool, deepcopy(args), step.stop_when))
        if not reviewed or not resumed:
            raise ValueError("recovery must review real evidence before VLA reentry")
        if len(calls) > max_tool_calls or (max_physical_steps is not None and physical > max_physical_steps):
            raise ValueError(f"recovery exceeds frozen budget: {rule.recovery_id}")
        binding = RecoveryBinding(
            binding_id=rule.recovery_id, failure_modes=list(rule.trigger_rule_ids),
            skill_entrypoint="bundle:" + rule.recovery_id,
            allowed_tools=list(dict.fromkeys(call.tool for call in calls)),
            max_recovery_steps=physical, max_agent_decisions=len(calls),
            monitor_policy="recovery_local", reentry_policy_id=rule.recovery_id,
        )
        programs[rule.recovery_id] = RecoveryProgram(binding, tuple(calls))
    return programs


def resolve_call(call, observation_id, review_token):
    args = deepcopy(call.arguments)
    if call.tool == "arx.review_reentry":
        args["observation_ids"] = [observation_id]
    elif call.tool == "arx.zeva":
        if not review_token:
            raise ValueError("reentry review did not grant a token")
        args["reentry_token"] = review_token
    return args


def verify_call_result(call, result, *, real):
    if result["status"] != "completed":
        raise ValueError(f"recovery call did not complete: {call.tool}")
    output = result.get("result") or {}
    if output.get("completion") == "task_success":
        if real and output.get("physical_arrival_verified") is not True:
            raise ValueError("physical arrival unverified at task success")
        return None
    if call.tool in ("arx.move_eef", "arx.set_gripper"):
        if output.get("command_target_reached") is not True:
            raise ValueError(f"recovery target not reached: {call.tool}")
    if call.tool in ("arx.move_eef", "arx.hold", "arx.set_gripper", "arx.zeva"):
        if real and output.get("physical_arrival_verified") is not True:
            raise ValueError(f"physical arrival unverified: {call.tool}")
    if call.tool == "arx.review_reentry":
        if output.get("assessment", {}).get("status") != "eligible" or not output.get("reentry_token"):
            raise ValueError("real reentry conditions are not satisfied")
    return output.get("reentry_token") if call.tool == "arx.review_reentry" else None
