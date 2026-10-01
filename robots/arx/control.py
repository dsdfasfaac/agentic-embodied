# Copyright (c) 2026 Zetta Contributors
"""Pure ARX X5 14-D state and action processing."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from robots.arx.contracts import ARX_ACTION_DIM, ArxControlSpec, ArxTaskManifest

ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])
GRIPPER_PERIOD = float(2.0 * np.pi)

__all__ = [
    "ARM_INDICES",
    "GRIPPER_INDICES",
    "ActionProcessor",
    "FilterInfo",
    "apply_task_locks",
    "as_action",
    "canonicalize_gripper",
    "interpolate_commands",
    "prepare_model_state",
]


def as_action(values: Any) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float32)
    if result.shape != (ARX_ACTION_DIM,):
        raise ValueError(f"expected a 14D vector, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("state/action contains a non-finite value")
    return result


def canonicalize_gripper(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"gripper value must be finite, got {value}")
    return float(((float(value) + GRIPPER_PERIOD) % GRIPPER_PERIOD) - GRIPPER_PERIOD)


def _control(task: ArxTaskManifest | ArxControlSpec) -> ArxControlSpec:
    return task.control if isinstance(task, ArxTaskManifest) else task


def prepare_model_state(
    measured_state: Any, task: ArxTaskManifest | ArxControlSpec
) -> np.ndarray:
    spec = _control(task)
    result = as_action(measured_state).copy()
    result[6] = canonicalize_gripper(float(result[6]))
    result[13] = canonicalize_gripper(float(result[13]))
    if spec.model_left_gripper is not None:
        result[6] = float(spec.model_left_gripper)
    if spec.model_right_gripper is not None:
        result[13] = float(spec.model_right_gripper)
    if spec.zero_left_model_state:
        result[:7] = 0.0
    if spec.zero_right_model_state:
        result[7:] = 0.0
    return result


def apply_task_locks(action: Any, task: ArxTaskManifest) -> np.ndarray:
    result = as_action(action).copy()
    start = np.asarray(task.start_state, dtype=np.float32)
    spec = task.control
    if spec.lock_left_arm:
        result[:7] = start[:7]
    if spec.lock_right_arm:
        result[7:] = start[7:]
    if spec.lock_right_gripper:
        result[13] = start[13]
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class FilterInfo:
    raw_arm_step: float
    raw_gripper_step: float
    output_arm_step: float
    output_gripper_step: float


class ActionProcessor:
    """Stateful task locks, low-pass filter, step limits, and jump guard."""

    def __init__(self, task: ArxTaskManifest, initial_command: Any) -> None:
        self.task = task
        self.previous = apply_task_locks(initial_command, task)

    def process(self, model_action: Any) -> tuple[np.ndarray, FilterInfo]:
        spec = self.task.control
        target = apply_task_locks(model_action, self.task)
        base = self.previous
        raw_step = target - base
        result = target.copy()
        result[ARM_INDICES] = (
            base[ARM_INDICES] + spec.arm_filter_alpha * raw_step[ARM_INDICES]
        )
        result[GRIPPER_INDICES] = (
            base[GRIPPER_INDICES]
            + spec.gripper_filter_alpha * raw_step[GRIPPER_INDICES]
        )
        filtered = result - base
        if spec.max_joint_step > 0:
            result[ARM_INDICES] = base[ARM_INDICES] + np.clip(
                filtered[ARM_INDICES], -spec.max_joint_step, spec.max_joint_step
            )
        if spec.max_gripper_step > 0:
            result[GRIPPER_INDICES] = base[GRIPPER_INDICES] + np.clip(
                filtered[GRIPPER_INDICES],
                -spec.max_gripper_step,
                spec.max_gripper_step,
            )
        result = apply_task_locks(result, self.task)
        output = result - base
        arm_jump = float(np.max(np.abs(output[ARM_INDICES])))
        gripper_jump = float(np.max(np.abs(output[GRIPPER_INDICES])))
        if arm_jump > spec.max_joint_delta:
            raise RuntimeError(
                f"unsafe arm command jump: {arm_jump:.4f} > {spec.max_joint_delta:.4f}"
            )
        if gripper_jump > spec.max_gripper_delta:
            raise RuntimeError(
                "unsafe gripper command jump: "
                f"{gripper_jump:.4f} > {spec.max_gripper_delta:.4f}"
            )
        self.previous = np.ascontiguousarray(result, dtype=np.float32)
        return self.previous.copy(), FilterInfo(
            raw_arm_step=float(np.max(np.abs(raw_step[ARM_INDICES]))),
            raw_gripper_step=float(np.max(np.abs(raw_step[GRIPPER_INDICES]))),
            output_arm_step=arm_jump,
            output_gripper_step=gripper_jump,
        )


def interpolate_commands(start: Any, target: Any, substeps: int) -> list[np.ndarray]:
    if substeps < 1:
        raise ValueError("substeps must be positive")
    start_array = as_action(start)
    target_array = as_action(target)
    commands: list[np.ndarray] = []
    for index in range(1, substeps + 1):
        phase = index / substeps
        alpha = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        command = start_array.copy()
        command[ARM_INDICES] = start_array[ARM_INDICES] + alpha * (
            target_array[ARM_INDICES] - start_array[ARM_INDICES]
        )
        command[GRIPPER_INDICES] = target_array[GRIPPER_INDICES]
        commands.append(np.ascontiguousarray(command, dtype=np.float32))
    return commands
