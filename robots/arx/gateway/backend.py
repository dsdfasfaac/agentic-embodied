# Copyright (c) 2026 Zetta Contributors
"""Direct binding; private data never becomes a public observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .public import CAMERAS


@dataclass(frozen=True)
class PolicyObservation:
    images: dict[str, np.ndarray]
    state: np.ndarray


@dataclass(frozen=True)
class PrivilegedObservation:
    """Bounded, trusted simulator projection for explicitly privileged runs."""

    schema_version: str
    selected: dict[str, Any]
    interaction: dict[str, Any]
    contact_summary: dict[str, Any]
    joint_summary: dict[str, Any]
    simulation: dict[str, Any]


@dataclass(frozen=True)
class HardwareEvidence:
    observation: dict[str, Any]
    command_receipt: dict[str, Any] | None
    arrival_verified: bool | None
    feature_frames: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class StepCommit:
    policy: PolicyObservation
    command: np.ndarray
    simulation_time_s: float
    environment_ended: bool
    private_evaluation: dict[str, Any]
    privileged: PrivilegedObservation | None = None
    hardware: HardwareEvidence | None = None


class Backend(Protocol):
    def reset(self) -> StepCommit: ...
    def step(self, raw_target: np.ndarray) -> StepCommit: ...
    def close(self) -> None: ...


class DirectBackend:
    """Construct this only inside the episode executor process."""

    def __init__(self, *, scene, mapping, task, seed):
        from robots.arx.environment import ArxMujocoEnv

        self.env = ArxMujocoEnv(
            prepared_scene_bundle=str(scene),
            mapping_path=str(mapping),
            task_manifest=str(task),
            camera_names={name: name for name in CAMERAS},
        )
        self.seed = seed

    def _commit(self, observation, command, ended, private):
        privileged = self.env.privileged_observation() if hasattr(self.env, "privileged_observation") else None
        return StepCommit(
            PolicyObservation(
                {k: observation[k].copy() for k in CAMERAS}, observation["state"].copy()
            ),
            np.asarray(command, dtype=np.float32).copy(),
            float(self.env.data.time),
            ended,
            private,
            privileged,
        )

    def reset(self):
        observation, info = self.env.reset(seed=self.seed)
        start = np.asarray(self.env.task.start_state, dtype=np.float32)
        if not np.allclose(observation["state"], start, atol=1e-5, rtol=0):
            raise ValueError("unsupported reset: configured command invariant failed")
        return self._commit(observation, start, False, info)

    def step(self, raw_target):
        observation, reward, terminated, truncated, info = self.env.step(raw_target)
        return self._commit(
            observation,
            info["processed_action"],
            bool(terminated or truncated),
            {
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                **info,
            },
        )

    def close(self):
        self.env.close()
