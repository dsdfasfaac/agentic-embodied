# Copyright (c) 2026 Zetta Contributors
"""Explicit ARX 14-D policy-to-MuJoCo joint and actuator mapping."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robots.arx.contracts import ARX_ACTION_DIM, ARX_GRIPPER_INDICES

__all__ = ["MappingValidationError", "MujocoMapping", "SimulationMappingError"]


class MappingValidationError(ValueError):
    """An ARX mapping file is structurally invalid."""


class SimulationMappingError(RuntimeError):
    """A policy value cannot be represented by the mapped simulator."""


@dataclasses.dataclass(frozen=True, slots=True)
class MujocoMapping:
    """Map twelve arm joints and two coupled grippers by explicit names."""

    joint_names: tuple[tuple[str, ...], ...]
    actuator_names: tuple[tuple[str, ...], ...]
    gripper_hardware_ranges: tuple[tuple[float, float], ...] = (
        (-3.4, 0.0),
        (-3.4, 0.0),
    )
    finger_position_ranges: tuple[tuple[float, float], ...] = (
        (0.0, 0.044),
        (0.0, 0.044),
    )

    @staticmethod
    def _channels(value: Sequence[Any], field: str) -> tuple[tuple[str, ...], ...]:
        if isinstance(value, (str, bytes)):
            raise MappingValidationError(f"{field} must be an array")
        items = tuple(value)
        if len(items) != ARX_ACTION_DIM:
            raise MappingValidationError(f"{field} must contain 14 channels")
        result: list[tuple[str, ...]] = []
        for index, item in enumerate(items):
            names = (item,) if isinstance(item, str) else tuple(item)
            # A reconstructed scene may omit the task-locked left arm. The
            # environment validates that these channels cannot be commanded.
            if not names and index < 7:
                result.append(())
                continue
            expected = 2 if index in ARX_GRIPPER_INDICES else 1
            if len(names) != expected or any(
                not isinstance(name, str) or not name.strip() for name in names
            ):
                raise MappingValidationError(
                    f"{field}[{index}] must contain {expected} non-empty names"
                )
            result.append(tuple(name.strip() for name in names))
        flat = [name for channel in result for name in channel]
        if len(flat) != len(set(flat)):
            raise MappingValidationError(f"{field} must contain unique names")
        return tuple(result)

    @staticmethod
    def _ranges(value: Any, field: str) -> tuple[tuple[float, float], ...]:
        ranges = tuple(value)
        if len(ranges) != 2:
            raise MappingValidationError(f"{field} must contain two ranges")
        result: list[tuple[float, float]] = []
        for index, pair in enumerate(ranges):
            try:
                low, high = (float(item) for item in pair)
            except (TypeError, ValueError) as exc:
                raise MappingValidationError(f"{field}[{index}] must be [low,high]") from exc
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise MappingValidationError(
                    f"{field}[{index}] must be finite with low < high"
                )
            result.append((low, high))
        return tuple(result)

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", self._channels(self.joint_names, "joint_names"))
        object.__setattr__(
            self, "actuator_names", self._channels(self.actuator_names, "actuator_names")
        )
        object.__setattr__(
            self,
            "gripper_hardware_ranges",
            self._ranges(self.gripper_hardware_ranges, "gripper_hardware_ranges"),
        )
        object.__setattr__(
            self,
            "finger_position_ranges",
            self._ranges(self.finger_position_ranges, "finger_position_ranges"),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> MujocoMapping:
        if not isinstance(value, Mapping):
            raise MappingValidationError("mapping must be an object")
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(value) - known - {"schema_version", "policy_order", "source_contract"})
        if unknown:
            raise MappingValidationError(f"unknown mapping keys: {unknown}")
        missing = [name for name in ("joint_names", "actuator_names") if name not in value]
        if missing:
            raise MappingValidationError(f"mapping is missing keys: {missing}")
        return cls(
            joint_names=tuple(value["joint_names"]),
            actuator_names=tuple(value["actuator_names"]),
            gripper_hardware_ranges=tuple(
                value.get("gripper_hardware_ranges", ((-3.4, 0.0), (-3.4, 0.0)))
            ),
            finger_position_ranges=tuple(
                value.get("finger_position_ranges", ((0.0, 0.044), (0.0, 0.044)))
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> MujocoMapping:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MappingValidationError(f"cannot read mapping {source}: {exc}") from exc
        return cls.from_mapping(payload)

    @property
    def flat_joint_names(self) -> tuple[str, ...]:
        return tuple(name for channel in self.joint_names for name in channel)

    @property
    def flat_actuator_names(self) -> tuple[str, ...]:
        return tuple(name for channel in self.actuator_names for name in channel)

    @staticmethod
    def _gripper_slot(channel: int) -> int:
        if channel == 6:
            return 0
        if channel == 13:
            return 1
        raise MappingValidationError(f"channel {channel} is not a gripper")

    def hardware_gripper_to_finger(
        self, channel: int, value: float, *, tolerance: float = 0.0
    ) -> float:
        slot = self._gripper_slot(channel)
        hardware_low, hardware_high = self.gripper_hardware_ranges[slot]
        finger_low, finger_high = self.finger_position_ranges[slot]
        value = self._bounded(value, hardware_low, hardware_high, tolerance, channel)
        phase = (value - hardware_low) / (hardware_high - hardware_low)
        return float(finger_high - phase * (finger_high - finger_low))

    def finger_to_hardware_gripper(
        self, channel: int, value: float, *, tolerance: float = 0.0
    ) -> float:
        slot = self._gripper_slot(channel)
        finger_low, finger_high = self.finger_position_ranges[slot]
        hardware_low, hardware_high = self.gripper_hardware_ranges[slot]
        value = self._bounded(value, finger_low, finger_high, tolerance, channel)
        phase = (value - finger_low) / (finger_high - finger_low)
        return float(hardware_high - phase * (hardware_high - hardware_low))

    @staticmethod
    def _bounded(
        value: float, low: float, high: float, tolerance: float, channel: int
    ) -> float:
        value = float(value)
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        boundary = max(tolerance, 1e-6)
        if not math.isfinite(value) or value < low - boundary or value > high + boundary:
            raise SimulationMappingError(
                f"channel {channel} value {value:.6g} is outside [{low},{high}]"
            )
        return min(max(value, low), high)
