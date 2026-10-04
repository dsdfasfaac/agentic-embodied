# Copyright (c) 2026 Zetta Contributors
"""Frozen tool catalog and proposal-only handlers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel

from .contracts import (
    EefArgs,
    ExecutionOutput,
    FinishArgs,
    FinishOutput,
    GatewayError,
    GripperArgs,
    HoldArgs,
    ReviewArgs,
    ReviewOutput,
    ZevaArgs,
    digest,
)

RECOVERY = ("INTERRUPTED", "RECOVERING")


@dataclass(frozen=True)
class ApprovedToolContext:
    command: np.ndarray
    observation: dict


@dataclass(frozen=True)
class ArxToolSpec:
    name: str
    version: int
    description: str
    kind: str
    capabilities: tuple[str, ...]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    allowed_states: tuple[str, ...]
    requires_reobservation: bool
    requirements: tuple[str, ...]

    def public(self):
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "kind": self.kind,
            "capabilities": list(self.capabilities),
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
            "allowed_states": list(self.allowed_states),
            "requires_reobservation": self.requires_reobservation,
            "requirements": list(self.requirements),
        }


@dataclass(frozen=True)
class RegisteredTool:
    spec: ArxToolSpec
    handler: Any


class ArxToolRegistry:
    def __init__(self, dependencies=()):
        self._tools = {}
        self._dependencies = frozenset(dependencies)
        self._catalog = None

    def register(self, spec, handler):
        if self._catalog is not None or spec.name in self._tools:
            raise ValueError("duplicate or frozen registration")
        protocol = {
            "execution": "prepare",
            "read_only": "inspect",
            "episode_control": None,
        }
        if (
            spec.kind not in protocol
            or not spec.name.startswith("arx.")
            or spec.version < 1
        ):
            raise ValueError("invalid tool metadata")
        if (
            not spec.description
            or not spec.capabilities
            or not set(spec.requirements) <= self._dependencies
        ):
            raise ValueError("missing description, capabilities, or dependencies")
        method = protocol[spec.kind]
        if method and not callable(getattr(handler, method, None)):
            raise ValueError("handler kind mismatch")
        if spec.kind == "episode_control" and (
            spec.name != "arx.finish" or handler is not None
        ):
            raise ValueError("unknown episode control")
        for model in (spec.input_model, spec.output_model):
            if (
                not issubclass(model, BaseModel)
                or model.model_json_schema().get("additionalProperties") is not False
            ):
                raise ValueError("strict object model required")
        self._tools[spec.name] = RegisteredTool(spec, handler)

    def freeze(self):
        if self._catalog is None:
            specs = [self._tools[key].spec.public() for key in sorted(self._tools)]
            self._catalog = {
                "schema_version": "arx.tool.catalog.v1",
                "tools": specs,
                "catalog_sha256": digest(specs),
            }
        return self.describe()

    def describe(self):
        if self._catalog is None:
            raise RuntimeError("catalog must be frozen")
        return deepcopy(self._catalog)

    def resolve(self, name):
        if self._catalog is None:
            raise RuntimeError("catalog must be frozen")
        if name not in self._tools:
            raise GatewayError("UNKNOWN_TOOL")
        return self._tools[name]


class ArrayPlan:
    def __init__(self, targets, *, convergence_target=None, tolerance=1e-4):
        self.targets = targets
        self.planned_steps = len(targets)
        self.limit = len(targets)
        self.target = convergence_target
        self.tolerance = tolerance
        self.reached = None if self.target is None else False
        self.sent = False

    def next_targets(self, context):
        if self.sent:
            return None
        self.sent = True
        if (
            self.target is not None
            and np.max(np.abs(context.command - self.target)) <= self.tolerance
        ):
            self.reached = True
            return None
        return self.targets

    def on_commit(self, context):
        if self.target is not None:
            self.reached = bool(
                np.max(np.abs(context.command - self.target)) <= self.tolerance
            )


class HoldPlanner:
    def prepare(self, args, context):
        return ArrayPlan(np.repeat(context.command[None], args.steps, axis=0))


class GripperPlanner:
    def __init__(
        self, mapping, *, command_offsets=(0.0, 0.0), processor_applies_offsets=False
    ):
        self.mapping = mapping
        # Current ActionProcessor consumes hardware commands directly. Keep the
        # offset contract explicit for a future processor version, never apply twice.
        self.offset = command_offsets[1] if processor_applies_offsets else 0.0
        if self.offset:
            raise ValueError(
                "offset-applying processors are not supported by this gateway version"
            )

    def prepare(self, args, context):
        low, high = self.mapping.finger_position_ranges[1]
        target = context.command.copy()
        target[13] = self.mapping.finger_to_hardware_gripper(
            13, low + args.opening * (high - low)
        )
        return ArrayPlan(
            np.repeat(target[None], args.max_steps, axis=0), convergence_target=target
        )


class EefPlanner:
    def __init__(self, kinematics):
        self.kinematics = kinematics

    def prepare(self, args, context):
        targets = self.kinematics.plan(context.command, args)
        return ArrayPlan(targets, convergence_target=targets[-1])


def default_registry(*, zeva, gripper=None, eef=None, reentry=None):
    motion_states = ("READY", "RUNNING_NOMINAL", *RECOVERY)
    dependencies = {"command_state", "policy_observation", "public_frames"}
    registry = ArxToolRegistry(dependencies)
    entries = [
        (
            "zeva",
            ZevaArgs,
            zeva,
            ("READY", "RUNNING_NOMINAL", *RECOVERY),
            "Infer fresh bounded Zeva targets.",
            "policy_observation",
        ),
        (
            "hold",
            HoldArgs,
            HoldPlanner(),
            motion_states,
            "Hold the committed command while physics advances.",
            "command_state",
        ),
    ]
    if gripper is not None:
        entries.append(
            (
                "set_gripper",
                GripperArgs,
                gripper,
                motion_states,
                "Set right gripper opening in command space.",
                "command_state",
            )
        )
    if eef is not None:
        entries.append(
            (
                "move_eef",
                EefArgs,
                eef,
                motion_states,
                "Plan a bounded relative right TCP command; arrival is not measured.",
                "command_state",
            )
        )
    for name, model, handler, states, description, dependency in entries:
        registry.register(
            ArxToolSpec(
                "arx." + name,
                1,
                description,
                "execution",
                ("motion." + name,),
                model,
                ExecutionOutput,
                states,
                True,
                (dependency,),
            ),
            handler,
        )
    if reentry is not None:
        registry.register(
            ArxToolSpec(
                "arx.review_reentry",
                1,
                "Review public evidence for reentry.",
                "read_only",
                ("review.reentry",),
                ReviewArgs,
                ReviewOutput,
                RECOVERY,
                False,
                ("public_frames",),
            ),
            reentry,
        )
    registry.register(
        ArxToolSpec(
            "arx.finish",
            1,
            "Close this episode without asserting success.",
            "episode_control",
            ("episode.finish",),
            FinishArgs,
            FinishOutput,
            ("READY", "RUNNING_NOMINAL", *RECOVERY, "ENDED"),
            False,
            (),
        ),
        None,
    )
    registry.freeze()
    return registry


class PolicyGripperPlanner:
    """Right gripper opening in the calibrated 14D policy coordinate."""

    def __init__(self, *, closed_policy: float, open_policy: float,
                 max_policy_step: float | None = None):
        if not np.isfinite([closed_policy, open_policy]).all() or closed_policy == open_policy:
            raise ValueError("distinct finite gripper endpoints required")
        if max_policy_step is not None and (not np.isfinite(max_policy_step) or max_policy_step <= 0):
            raise ValueError("positive gripper policy step limit required")
        self.closed_policy = float(closed_policy)
        self.open_policy = float(open_policy)
        self.max_policy_step = max_policy_step

    def prepare(self, args, context):
        target = context.command.copy()
        target[13] = self.closed_policy + args.opening * (
            self.open_policy - self.closed_policy
        )
        if (self.max_policy_step is not None and
                abs(float(target[13] - context.command[13])) >
                args.max_steps * self.max_policy_step + 1e-6):
            raise ValueError("gripper recovery step budget cannot reach target")
        return ArrayPlan(
            np.repeat(target[None], args.max_steps, axis=0),
            convergence_target=target,
        )
