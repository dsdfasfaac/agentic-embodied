# Copyright (c) 2026 Zetta Contributors
"""Environment-owned terminal evaluators for ARX manipulation tasks."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import mujoco
import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class Evaluation:
    success: bool
    failure: bool
    reason: str | None
    progress: float
    hold_steps: int
    target_height: float
    lift: float
    finger_contacts: tuple[bool, bool]


def _vector(value: Any, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


class PickupTestTubeEvaluator:
    """Require target lift plus simultaneous contact by both allowed fingers."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        logical_body_map: Mapping[str, str],
        success: Mapping[str, Any],
        failure: Mapping[str, Any],
    ) -> None:
        expected_success = {
            "target_logical_id", "allowed_gripper", "lift_height_m", "hold_steps",
            "require_bilateral_finger_contact",
        }
        expected_failure = {"workspace_min", "workspace_max", "unrecoverable_drop_z"}
        if set(success) != expected_success or set(failure) != expected_failure:
            raise ValueError("pickup evaluator parameters must match the pinned schema")
        logical_id = str(success["target_logical_id"])
        body_name = logical_body_map.get(logical_id)
        if not body_name:
            raise ValueError(f"target logical ID has no prepared body mapping: {logical_id}")
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"target body is missing from model: {body_name}")
        side = str(success["allowed_gripper"])
        if side not in {"left", "right"}:
            raise ValueError("allowed_gripper must be left or right")
        finger_bodies = (f"{side}_finger_a", f"{side}_finger_b")
        self.finger_body_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in finger_bodies
        )
        if any(item < 0 for item in self.finger_body_ids):
            raise ValueError(f"allowed gripper bodies are missing: {finger_bodies}")
        self.model, self.data = model, data
        self.lift_height = float(success["lift_height_m"])
        self.required_hold = int(success["hold_steps"])
        self.require_contacts = bool(success["require_bilateral_finger_contact"])
        if not math.isfinite(self.lift_height) or self.lift_height <= 0 or self.required_hold < 1:
            raise ValueError("lift height and hold steps must be positive")
        self.workspace_min = _vector(failure["workspace_min"], "workspace_min")
        self.workspace_max = _vector(failure["workspace_max"], "workspace_max")
        if np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("workspace_min must be below workspace_max")
        self.drop_z = float(failure["unrecoverable_drop_z"])
        self.initial_z = 0.0
        self.hold = 0

    def reset(self) -> Evaluation:
        self.initial_z = float(self.data.xpos[self.body_id, 2])
        self.hold = 0
        return self.evaluate(update_hold=False)

    def _contacts(self) -> tuple[bool, bool]:
        touched = [False, False]
        for contact in self.data.contact[: self.data.ncon]:
            bodies = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            for index, finger in enumerate(self.finger_body_ids):
                touched[index] |= self.body_id in bodies and finger in bodies
        return bool(touched[0]), bool(touched[1])

    def evaluate(self, *, update_hold: bool = True) -> Evaluation:
        position = self.data.xpos[self.body_id].copy()
        lift = float(position[2] - self.initial_z)
        contacts = self._contacts()
        candidate = lift >= self.lift_height and (
            not self.require_contacts or all(contacts)
        )
        if update_hold:
            self.hold = self.hold + 1 if candidate else 0
        failure = bool(
            position[2] < self.drop_z
            or np.any(position < self.workspace_min)
            or np.any(position > self.workspace_max)
        )
        success = self.hold >= self.required_hold and not failure
        reason = "unsafe_target_pose" if failure else "success" if success else None
        return Evaluation(
            success, failure, reason,
            min(max(lift / self.lift_height, 0.0), 1.0), self.hold,
            float(position[2]), lift, contacts,
        )
