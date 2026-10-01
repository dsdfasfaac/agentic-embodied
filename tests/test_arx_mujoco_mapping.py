# Copyright (c) 2026 Zetta Contributors
"""Tests for the explicit ARX-to-MuJoCo mapping."""

from __future__ import annotations

from pathlib import Path

import pytest

from robots.arx.mujoco_mapping import (
    MappingValidationError,
    MujocoMapping,
    SimulationMappingError,
)

MAPPING = Path("../Zeva_arx/assets/ac_one/ac_one_14d_mapping.json")


def test_checked_in_arx_mapping_has_paired_fingers() -> None:
    mapping = MujocoMapping.from_json(MAPPING)
    assert len(mapping.joint_names) == len(mapping.actuator_names) == 14
    assert len(mapping.joint_names[6]) == len(mapping.joint_names[13]) == 2
    assert len(mapping.actuator_names[6]) == len(mapping.actuator_names[13]) == 2
    assert len(mapping.flat_joint_names) == len(mapping.flat_actuator_names) == 16


@pytest.mark.parametrize("channel", [6, 13])
def test_gripper_mapping_endpoints_and_roundtrip(channel: int) -> None:
    mapping = MujocoMapping.from_json(MAPPING)
    assert mapping.hardware_gripper_to_finger(channel, -3.4) == pytest.approx(0.044)
    assert mapping.hardware_gripper_to_finger(channel, 0.0) == pytest.approx(0.0)
    for hardware in (-3.4, -2.1, -1.7, -0.2, 0.0):
        finger = mapping.hardware_gripper_to_finger(channel, hardware)
        assert mapping.finger_to_hardware_gripper(channel, finger) == pytest.approx(hardware)


def test_mapping_rejects_single_finger_and_duplicates() -> None:
    mapping = MujocoMapping.from_json(MAPPING)
    payload = {
        "joint_names": [list(item) for item in mapping.joint_names],
        "actuator_names": [list(item) for item in mapping.actuator_names],
    }
    payload["joint_names"][6] = ["only_one"]
    with pytest.raises(MappingValidationError, match="2 non-empty"):
        MujocoMapping.from_mapping(payload)
    payload["joint_names"][6] = list(mapping.joint_names[6])
    payload["joint_names"][1] = list(mapping.joint_names[0])
    with pytest.raises(MappingValidationError, match="unique"):
        MujocoMapping.from_mapping(payload)


def test_mapping_rejects_out_of_range_and_non_gripper() -> None:
    mapping = MujocoMapping.from_json(MAPPING)
    with pytest.raises(SimulationMappingError, match="outside"):
        mapping.hardware_gripper_to_finger(6, 0.1)
    with pytest.raises(MappingValidationError, match="not a gripper"):
        mapping.hardware_gripper_to_finger(0, -1.0)


def test_finger_readback_clamps_only_within_explicit_solver_tolerance() -> None:
    mapping = MujocoMapping.from_json(MAPPING)

    assert mapping.finger_to_hardware_gripper(
        13, 0.0440324, tolerance=1e-4
    ) == pytest.approx(-3.4)
    with pytest.raises(SimulationMappingError, match="outside"):
        mapping.finger_to_hardware_gripper(13, 0.0442, tolerance=1e-4)
