# Copyright (c) 2026 Zetta Contributors
"""Terminal evaluator tests independent of policy inference."""

from __future__ import annotations

import mujoco

from robots.arx.evaluators import PickupTestTubeEvaluator


def _evaluator() -> tuple[PickupTestTubeEvaluator, mujoco.MjData]:
    xml = """<mujoco><worldbody>
      <body name='target'><freejoint/><geom name='target_geom' type='sphere' size='.02'/></body>
      <body name='right_finger_a' pos='0 -.015 0'><geom name='a' type='sphere' size='.02'/></body>
      <body name='right_finger_b' pos='0 .015 0'><geom name='b' type='sphere' size='.02'/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    evaluator = PickupTestTubeEvaluator(
        model, data, {"tube_2": "target"},
        {"target_logical_id": "tube_2", "allowed_gripper": "right", "lift_height_m": .01, "hold_steps": 2, "require_bilateral_finger_contact": True},
        {"workspace_min": [-1, -1, -.2], "workspace_max": [1, 1, 1], "unrecoverable_drop_z": -.1},
    )
    evaluator.reset()
    return evaluator, data


def test_success_requires_lift_contacts_and_hold() -> None:
    evaluator, data = _evaluator()
    data.qpos[2] = .02
    mujoco.mj_forward(evaluator.model, data)
    first = evaluator.evaluate()
    second = evaluator.evaluate()
    assert first.finger_contacts == (True, True)
    assert not first.success
    assert second.success and second.reason == "success"


def test_drop_is_authoritative_failure() -> None:
    evaluator, data = _evaluator()
    data.qpos[2] = -.15
    mujoco.mj_forward(evaluator.model, data)
    result = evaluator.evaluate()
    assert result.failure and not result.success
    assert result.reason == "unsafe_target_pose"
