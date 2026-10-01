# Copyright (c) 2026 Zetta Contributors
"""Parity tests for ARX action processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robots.arx.contracts import load_task_manifest
from robots.arx.control import (
    ARM_INDICES,
    GRIPPER_INDICES,
    ActionProcessor,
    as_action,
    canonicalize_gripper,
    interpolate_commands,
    prepare_model_state,
)

TASK = load_task_manifest(
    Path(__file__).resolve().parents[1]
    / "robots/arx/manifests/pickup_test_tube.yaml"
)


def test_prepare_model_state_matches_pickup_contract() -> None:
    measured = np.asarray(TASK.start_state, dtype=np.float32)
    output = prepare_model_state(measured, TASK)
    np.testing.assert_array_equal(output[:7], np.zeros(7, dtype=np.float32))
    assert -2 * np.pi <= output[13] < 0
    np.testing.assert_array_equal(measured, np.asarray(TASK.start_state, dtype=np.float32))


@pytest.mark.parametrize("value", [0.0, -2 * np.pi, 2 * np.pi, -4 * np.pi])
def test_gripper_canonical_interval(value: float) -> None:
    result = canonicalize_gripper(value)
    assert -2 * np.pi <= result < 0


def test_action_processor_locks_left_and_limits_steps() -> None:
    initial = np.asarray(TASK.start_state, dtype=np.float32)
    processor = ActionProcessor(TASK, initial)
    target = initial + 2.0
    output, info = processor.process(target)
    np.testing.assert_array_equal(output[:7], initial[:7])
    assert np.max(np.abs(output[ARM_INDICES] - initial[ARM_INDICES])) <= 0.035001
    assert np.max(np.abs(output[GRIPPER_INDICES] - initial[GRIPPER_INDICES])) <= 0.080001
    assert info.output_arm_step <= 0.035001


def test_interpolation_uses_quintic_arms_and_immediate_grippers() -> None:
    start = np.zeros(14, dtype=np.float32)
    target = np.ones(14, dtype=np.float32)
    commands = interpolate_commands(start, target, 4)
    assert len(commands) == 4
    np.testing.assert_array_equal(commands[0][GRIPPER_INDICES], np.ones(2))
    np.testing.assert_allclose(commands[-1], target)
    assert 0 < commands[0][0] < commands[1][0] < commands[2][0] < 1


def test_invalid_action_does_not_advance_processor() -> None:
    initial = np.asarray(TASK.start_state, dtype=np.float32)
    processor = ActionProcessor(TASK, initial)
    with pytest.raises(ValueError, match="14D"):
        processor.process(np.zeros(13))
    np.testing.assert_array_equal(processor.previous, initial)
    with pytest.raises(ValueError, match="non-finite"):
        as_action([float("nan")] * 14)
