# Copyright (c) 2026 Zetta Contributors
"""Hash-pinned expert-skill policy for the reBot G1-D MuJoCo tasks.

The supplied reBot bundle contains two complete, deterministic state machines.
This backend audits their source through the same asset manifest as the Runtime,
compiles one state machine into absolute joint targets in a private planning
simulation, and serves those targets through ``PolicyInferenceCore``.  The
actual episode is still advanced only by Gateway ``policy_step`` calls, and the
environment independently determines success from MuJoCo state and contacts.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.backends.rebot_g1d_session import (
    REBOT_G1D_ACTION_NAMES,
    REBOT_G1D_TASK_SPECS,
    compute_rebot_asset_manifest,
)
from rollout_runtime.core import payload as payload_module

__all__ = [
    "REBOT_G1D_SKILL_POLICY_FAMILY",
    "RebotG1DSkillPolicyConfig",
    "RebotG1DSkillPolicyCore",
]

REBOT_G1D_SKILL_POLICY_FAMILY = "rebot_g1d_skill"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass(kw_only=True)
class RebotG1DSkillPolicyConfig:
    """Configuration for one immutable, task-specific reBot expert skill."""

    asset_root: str
    asset_manifest_sha256: str
    task: str
    scene_config: str
    seed: int = 0
    action_dim: int = 22
    actions_per_chunk: int = 32
    device: str = "cpu"
    dtype: str = "float32"
    policy_family: str = REBOT_G1D_SKILL_POLICY_FAMILY
    policy_id: str | None = None
    model_version: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RebotG1DSkillPolicyConfig:
        """Build a strict config, rejecting unknown or inconsistent fields."""
        raw = dict(value)
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown reBot skill policy_config keys: {unknown}")
        try:
            result = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid reBot skill policy config: {exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        """Validate the fixed task, source pin, and output shape."""
        task_spec = REBOT_G1D_TASK_SPECS.get(self.task)
        if task_spec is None:
            raise ValueError(
                f"task must be one of {sorted(REBOT_G1D_TASK_SPECS)}, got {self.task!r}"
            )
        root = Path(self.asset_root).expanduser()
        if not root.is_absolute():
            raise ValueError("reBot skill asset_root must be absolute")
        if not _SHA256_PATTERN.fullmatch(self.asset_manifest_sha256):
            raise ValueError(
                "reBot skill asset_manifest_sha256 must be 64 lowercase hex characters"
            )
        if self.scene_config != task_spec["scene_config"]:
            raise ValueError(
                f"task {self.task!r} requires scene_config "
                f"{task_spec['scene_config']!r}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("reBot skill seed must be an integer")
        if self.action_dim != len(REBOT_G1D_ACTION_NAMES):
            raise ValueError(
                f"reBot skill action_dim must be {len(REBOT_G1D_ACTION_NAMES)}"
            )
        if (
            isinstance(self.actions_per_chunk, bool)
            or not isinstance(self.actions_per_chunk, int)
            or self.actions_per_chunk < 1
        ):
            raise ValueError("reBot skill actions_per_chunk must be positive")
        if self.policy_family != REBOT_G1D_SKILL_POLICY_FAMILY:
            raise ValueError(
                f"reBot skill policy_family must be {REBOT_G1D_SKILL_POLICY_FAMILY!r}"
            )
        expected_policy_id = f"rebot_{self.task}_skill"
        if self.policy_id is None:
            self.policy_id = expected_policy_id
        elif self.policy_id != expected_policy_id:
            raise ValueError(
                f"task {self.task!r} requires policy_id={expected_policy_id!r}"
            )

    @property
    def resolved_model_version(self) -> str:
        """Return a source/seed-bound immutable model-version label."""
        if self.model_version:
            return self.model_version
        return (
            f"rebot-{self.task}-skill-"
            f"{self.asset_manifest_sha256[:12]}-seed-{self.seed}"
        )


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _pinned_simulation_modules(
    root: Path, task: str, fingerprint: str
) -> Iterator[tuple[ModuleType, ModuleType]]:
    """Load only manifest-covered simulation modules, restoring globals after."""
    task_spec = REBOT_G1D_TASK_SPECS[task]
    adapter_path = str(task_spec["adapter_path"])
    adapter_name = Path(adapter_path).stem
    module_names = (
        "simulation",
        f"simulation.{adapter_name}",
        "simulation.geometric_grasp",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    previous_path = list(sys.path)
    try:
        package = ModuleType("simulation")
        package.__path__ = [str(root / "simulation")]  # type: ignore[attr-defined]
        package.__package__ = "simulation"
        sys.modules["simulation"] = package
        adapter = _load_source_module(f"simulation.{adapter_name}", root / adapter_path)
        _load_source_module(
            "simulation.geometric_grasp", root / "simulation/geometric_grasp.py"
        )
        script = _load_source_module(
            f"_agentic_rebot_skill_{task}_{fingerprint}",
            root / str(task_spec["skill_path"]),
        )
        yield adapter, script
    finally:
        sys.path[:] = previous_path
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop(f"_agentic_rebot_skill_{task}_{fingerprint}", None)


def _compile_trajectory(
    config: RebotG1DSkillPolicyConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute the pinned planner state machine in a private recording scene."""
    root = Path(config.asset_root).expanduser().resolve(strict=True)
    manifest = compute_rebot_asset_manifest(root, config.scene_config)
    actual_digest = str(manifest["sha256"])
    if actual_digest != config.asset_manifest_sha256:
        raise RuntimeError(
            "reBot skill asset manifest mismatch: "
            f"expected {config.asset_manifest_sha256}, calculated {actual_digest}"
        )

    recorded: list[np.ndarray] = []
    output_log = io.StringIO()
    fingerprint = hashlib.sha256(
        f"{root}\0{actual_digest}\0{config.task}".encode("utf-8")
    ).hexdigest()[:16]
    with _pinned_simulation_modules(root, config.task, fingerprint) as (
        adapter,
        script,
    ):
        base_env = getattr(adapter, "G1DMujocoEnv")
        controlled_joints = tuple(getattr(adapter, "CONTROLLED_JOINTS"))
        if controlled_joints != REBOT_G1D_ACTION_NAMES:
            raise RuntimeError("reBot skill adapter has an unexpected joint order")

        class RecordingEnv(base_env):  # type: ignore[misc, valid-type]
            """Record the target command immediately before every physics step."""

            def step(self, frame_skip: int | None = None) -> Any:
                recorded.append(
                    np.asarray(
                        [self.targets[name] for name in controlled_joints],
                        dtype=np.float32,
                    )
                )
                return super().step(frame_skip=frame_skip)

        script.G1DMujocoEnv = RecordingEnv
        old_argv = list(sys.argv)
        try:
            with tempfile.TemporaryDirectory(prefix="rebot-agentic-plan-") as temp:
                output_dir = Path(temp)
                sys.argv = [
                    str(root / str(REBOT_G1D_TASK_SPECS[config.task]["skill_path"])),
                    "--config",
                    str(root / config.scene_config),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    str(config.seed),
                    "--arm",
                    "auto",
                ]
                with (
                    contextlib.redirect_stdout(output_log),
                    contextlib.redirect_stderr(output_log),
                ):
                    return_code = int(script.main())
                summary_path = output_dir / "summary.json"
                if not summary_path.is_file():
                    raise RuntimeError("reBot skill compiler produced no summary.json")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
        finally:
            sys.argv = old_argv

    if return_code != 0 or not bool(summary.get("success")):
        raise RuntimeError(
            f"reBot {config.task} skill planning failed with code {return_code}; "
            f"log tail: {output_log.getvalue()[-2000:]}"
        )
    actions = np.asarray(recorded, dtype=np.float32)
    expected_shape = (len(recorded), len(REBOT_G1D_ACTION_NAMES))
    if (
        not recorded
        or actions.shape != expected_shape
        or not np.isfinite(actions).all()
    ):
        raise RuntimeError(
            f"reBot skill compiler produced invalid actions with shape {actions.shape}"
        )
    summary_bytes = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    audit = {
        "task": config.task,
        "seed": config.seed,
        "asset_manifest_sha256": actual_digest,
        "trajectory_steps": int(actions.shape[0]),
        "trajectory_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "planner_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "planner_reported_success": True,
        "planner_arm": summary.get("arm"),
    }
    return actions, audit


class RebotG1DSkillPolicyCore:
    """Serve one compiled expert trajectory through the Runtime policy plane."""

    def __init__(self, config: RebotG1DSkillPolicyConfig) -> None:
        self.config = config
        self._model_version = config.resolved_model_version
        self._actions: np.ndarray | None = None
        self._audit: dict[str, Any] = {}
        self.loaded = False
        self.closed = False

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def device(self) -> str:
        return self.config.device

    @property
    def dtype(self) -> str:
        return self.config.dtype

    @property
    def policy_family(self) -> str:
        return self.config.policy_family

    @property
    def audit(self) -> dict[str, Any]:
        """Return the immutable compilation audit after ``load``."""
        return dict(self._audit)

    def load(self) -> None:
        if self.loaded:
            return
        self._actions, self._audit = _compile_trajectory(self.config)
        self.loaded = True

    def update_weights(self, model_version: str) -> None:
        if model_version != self._model_version:
            raise ValueError("the hash-pinned reBot expert skill is immutable")

    def close(self) -> None:
        self._actions = None
        self.closed = True

    def _error(self, request: InferenceRequest, message: str) -> ActionResponse:
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            model_version=self._model_version,
            error=make_error(ErrorCode.POLICY_FAILURE, message),
        )

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        responses: list[ActionResponse] = []
        for request in requests:
            if not self.loaded or self._actions is None:
                responses.append(
                    self._error(request, "reBot skill policy is not loaded")
                )
                continue
            if request.policy_id != self.config.policy_id:
                responses.append(
                    self._error(
                        request,
                        f"reBot {self.config.task} skill rejects policy_id "
                        f"{request.policy_id!r}",
                    )
                )
                continue
            extras = request.observation.extras
            if (
                extras.get("provider") != "rebot_g1d"
                or tuple(extras.get("action_names", ())) != REBOT_G1D_ACTION_NAMES
            ):
                responses.append(
                    self._error(
                        request, "observation is not the pinned reBot G1-D schema"
                    )
                )
                continue
            start = int(request.observation.step_index)
            if start < 0 or start >= int(self._actions.shape[0]):
                responses.append(
                    self._error(
                        request,
                        f"reBot skill exhausted at observation step {start} without "
                        "Runtime success",
                    )
                )
                continue
            stop = min(start + self.config.actions_per_chunk, self._actions.shape[0])
            actions = np.ascontiguousarray(self._actions[start:stop])
            responses.append(
                ActionResponse(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    binding_token=request.binding_token,
                    episode_id=request.episode_id,
                    operation_seq=request.operation_seq,
                    actions=payload_module.encode_array(actions),
                    model_version=self._model_version,
                    auxiliary_outputs={
                        **self._audit,
                        "trajectory_start": start,
                        "trajectory_stop": int(stop),
                    },
                )
            )
        return responses
