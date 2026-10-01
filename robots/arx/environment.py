# Copyright (c) 2026 Zetta Contributors
"""Gymnasium environment for a prepared Real2Sim scene and bilateral AC one."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np

from robots.arx.contracts import ARX_CAMERA_NAMES, ArxTaskManifest, load_task_manifest
from robots.arx.control import ActionProcessor, as_action, interpolate_commands
from robots.arx.evaluators import PickupTestTubeEvaluator
from robots.arx.mujoco_mapping import MujocoMapping

__all__ = ["ArxMujocoEnv"]


class ArxMujocoEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Execute raw 14-D Zeva position actions in a prepared MuJoCo model.

    Terminal success is intentionally evaluator-owned. Until a task evaluator
    is registered, this environment only produces safety and time-limit exits.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        prepared_scene_bundle: str,
        mapping_path: str,
        task_manifest: str,
        camera_names: Mapping[str, str] | None = None,
        policy_hz: float = 15.0,
        command_hz: float = 60.0,
        physics_steps_per_command: int = 10,
        image_width: int = 320,
        image_height: int = 240,
        trace_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        self._trace_hook = trace_hook
        self.root = Path(prepared_scene_bundle).expanduser().resolve(strict=True)
        self.task: ArxTaskManifest = load_task_manifest(task_manifest)
        self.mapping = MujocoMapping.from_json(mapping_path)
        for channel, (joints, actuators) in enumerate(zip(
            self.mapping.joint_names, self.mapping.actuator_names
        )):
            if not joints or not actuators:
                if joints or actuators or channel >= 7 or not self.task.control.lock_left_arm:
                    raise ValueError("absent channels require matching empty mappings and a task-locked left arm")
        if policy_hz <= 0 or command_hz <= 0 or physics_steps_per_command < 1:
            raise ValueError("control frequencies and physics steps must be positive")
        ratio = command_hz / policy_hz
        if not float(ratio).is_integer():
            raise ValueError("command_hz must be an integer multiple of policy_hz")
        self.commands_per_action = int(ratio)
        self.physics_steps_per_command = int(physics_steps_per_command)
        self.model = mujoco.MjModel.from_binary_path(str(self.root / "model.mjb"))
        self.data = mujoco.MjData(self.model)
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        logical_body_map = metadata.get("logical_body_map")
        if not isinstance(logical_body_map, dict):
            raise ValueError("prepared scene metadata must contain logical_body_map")
        # AC one source uses 2 ms, which cannot represent 15/60 Hz exactly.
        # Pinning the runtime step yields exact command and policy durations.
        if metadata.get("preserve_physics_timestep", False):
            substeps = 1.0 / (command_hz * self.model.opt.timestep)
            if abs(substeps - round(substeps)) > 1e-8:
                raise ValueError("source timestep must divide the command interval")
            self.physics_steps_per_command = int(round(substeps))
        else:
            self.model.opt.timestep = 1.0 / (command_hz * physics_steps_per_command)
        self._render_options = None
        if "render_geom_groups" in metadata:
            self._render_options = mujoco.MjvOption()
            self._render_options.geomgroup[:] = 0
            self._render_options.geomgroup[metadata["render_geom_groups"]] = 1
        self._reset = np.load(self.root / "reset_state.npz")
        self._validate_reset()
        self._joint_ids, self._actuator_ids = self._resolve_mapping()
        if self.task.success["evaluator"] != "pickup_test_tube_v1" or self.task.failure["evaluator"] != "arx_safety_v1":
            raise ValueError("unsupported ARX evaluator configuration")
        self._evaluator = PickupTestTubeEvaluator(
            self.model,
            self.data,
            logical_body_map,
            self.task.success.get("parameters", {}),
            self.task.failure.get("parameters", {}),
        )
        self.camera_names = self._validate_cameras(camera_names)
        self.image_width, self.image_height = int(image_width), int(image_height)
        if self.image_width < 1 or self.image_height < 1:
            raise ValueError("image dimensions must be positive")
        # Keep an independent native renderer per camera. Some MuJoCo EGL
        # builds abort when one Renderer is repeatedly retargeted across
        # cameras in a single context.
        self._renderers: dict[str, mujoco.Renderer] = {}
        limit = np.finfo(np.float32).max
        self.action_space = gym.spaces.Box(-limit, limit, (14,), np.float32)
        spaces: dict[str, gym.Space[Any]] = {
            "state": gym.spaces.Box(-limit, limit, (14,), np.float32)
        }
        for semantic in self.camera_names:
            spaces[semantic] = gym.spaces.Box(
                0, 255, (self.image_height, self.image_width, 3), np.uint8
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self._processor: ActionProcessor | None = None
        self._last_command = np.zeros(14, dtype=np.float32)
        self._steps = 0
        self._finished = False
        self._trace(
            "environment_initialized",
            nq=self.model.nq,
            nv=self.model.nv,
            nu=self.model.nu,
            timestep=float(self.model.opt.timestep),
        )

    def _trace(self, event: str, **payload: Any) -> None:
        if self._trace_hook is None:
            return
        try:
            self._trace_hook(event, payload)
        except Exception:
            # Diagnostics must never change simulation behavior.
            pass

    def _validate_reset(self) -> None:
        if set(self._reset.files) != {"qpos", "qvel", "time"}:
            raise ValueError("prepared reset state must contain qpos, qvel, and time")
        if self._reset["qpos"].shape != (self.model.nq,) or self._reset["qvel"].shape != (self.model.nv,):
            raise ValueError("prepared reset state dimensions do not match model")

    def _resolve_mapping(self) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        def resolve(kind: mujoco.mjtObj, channels: tuple[tuple[str, ...], ...], label: str) -> tuple[tuple[int, ...], ...]:
            result = []
            for channel in channels:
                ids = tuple(mujoco.mj_name2id(self.model, kind, name) for name in channel)
                if any(index < 0 for index in ids):
                    missing = [name for name, index in zip(channel, ids) if index < 0]
                    raise ValueError(f"mapped {label} names are missing: {missing}")
                result.append(ids)
            return tuple(result)
        return (
            resolve(mujoco.mjtObj.mjOBJ_JOINT, self.mapping.joint_names, "joint"),
            resolve(mujoco.mjtObj.mjOBJ_ACTUATOR, self.mapping.actuator_names, "actuator"),
        )

    def _validate_cameras(self, cameras: Mapping[str, str] | None) -> dict[str, str]:
        result = dict(cameras or {})
        if result and tuple(result) != ARX_CAMERA_NAMES:
            raise ValueError(f"camera semantic order must be {ARX_CAMERA_NAMES}")
        if len(set(result.values())) != len(result):
            raise ValueError("MuJoCo camera names must be unique")
        for name in result.values():
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"MuJoCo camera is missing: {name}")
        return result

    def _state(self) -> np.ndarray:
        values = np.empty(14, dtype=np.float32)
        for channel, ids in enumerate(self._joint_ids):
            if not ids:
                values[channel] = self.task.start_state[channel]
                continue
            qpos = [float(self.data.qpos[self.model.jnt_qposadr[index]]) for index in ids]
            if channel in (6, 13):
                # Limited MuJoCo joints are constraint-solved and their qpos
                # may transiently cross a hard stop. The policy contract,
                # however, requires a representable hardware-domain state.
                # Project finite simulator readback onto the declared finger
                # interval; reset and command conversion keep their separate
                # input-domain validation paths.
                slot = 0 if channel == 6 else 1
                finger_low, finger_high = self.mapping.finger_position_ranges[slot]
                finger_position = float(np.mean(qpos))
                if np.isfinite(finger_position):
                    raw_finger_position = finger_position
                    finger_position = float(
                        np.clip(finger_position, finger_low, finger_high)
                    )
                    if finger_position != raw_finger_position:
                        self._trace(
                            "finger_readback_clipped",
                            channel=channel,
                            raw=raw_finger_position,
                            clipped=finger_position,
                            low=finger_low,
                            high=finger_high,
                        )
                values[channel] = self.mapping.finger_to_hardware_gripper(
                    channel, finger_position
                )
            else:
                values[channel] = qpos[0]
        return values

    def _set_command(self, command: np.ndarray) -> None:
        for channel, ids in enumerate(self._actuator_ids):
            value = float(command[channel])
            if channel in (6, 13):
                value = self.mapping.hardware_gripper_to_finger(
                    channel, value, tolerance=0.1
                )
            for actuator_id in ids:
                self.data.ctrl[actuator_id] = value

    def _initialize_robot(self) -> None:
        start = as_action(self.task.start_state)
        for channel, ids in enumerate(self._joint_ids):
            value = float(start[channel])
            if channel in (6, 13):
                # A task reset is configuration, not noisy simulator readback:
                # reject an invalid hardware-domain start instead of silently
                # commanding a target beyond the MuJoCo finger joint stop.
                value = self.mapping.hardware_gripper_to_finger(
                    channel, value
                )
            for joint_id in ids:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        self._set_command(start)
        mujoco.mj_forward(self.model, self.data)

    def _observation(self) -> dict[str, np.ndarray]:
        self._trace("observation_begin", step=self._steps, sim_time=float(self.data.time))
        result = {"state": self._state()}
        self._trace("state_ready", step=self._steps, state=result["state"].tolist())
        if self.camera_names:
            for semantic, camera in self.camera_names.items():
                self._trace("camera_render_begin", semantic=semantic, camera=camera)
                renderer = self._renderers.get(semantic)
                if renderer is None:
                    self._trace("renderer_create_begin", semantic=semantic)
                    renderer = mujoco.Renderer(
                        self.model, height=self.image_height, width=self.image_width
                    )
                    self._renderers[semantic] = renderer
                    self._trace("renderer_create_end", semantic=semantic)
                renderer.update_scene(self.data, camera=camera, scene_option=self._render_options)
                # Own the pixels before the renderer is reused or destroyed;
                # render() may expose a view onto a recycled native buffer.
                result[semantic] = np.array(renderer.render(), copy=True, order="C")
                self._trace("camera_render_end", semantic=semantic, camera=camera)
        self._trace("observation_end", step=self._steps)
        return result

    def privileged_observation(self):
        """Return the bounded task projection used by privileged ARX runs."""
        from robots.arx.gateway.backend import PrivilegedObservation

        position = np.asarray(self.data.xpos[self._evaluator.body_id], dtype=float)
        success = self.task.success.get("parameters", self.task.success)
        target = {"logical_id": str(success["target_logical_id"]),
                  "position_m": position.tolist()}
        eef = np.asarray(self.data.xpos[self._evaluator.finger_body_ids[0]], dtype=float)
        distance = float(np.linalg.norm(position - eef))
        evaluation = self._evaluator.evaluate(update_hold=False)
        contacts = evaluation.finger_contacts
        joints = {f"joint_{i}": float(value) for i, value in enumerate(self._state())}
        state = self._state()
        gripper_closed = bool(float(state[-1]) < -2.0)
        return PrivilegedObservation(
            schema_version="arx.privileged.observation.v1",
            selected={"manipulated_object": target, "target": target,
                      "target_gripper_distance_m": distance},
            interaction={"gripper_closed": gripper_closed,
                         "gripper_contact": bool(any(contacts)),
                         "robot_contact": bool(any(contacts)),
                         "grasped": bool(all(contacts)),
                         "retained": bool(all(contacts)),
                         "released_now": False, "released_ever": False,
                         "in_target": bool(evaluation.success),
                         "lift_m": float(evaluation.lift),
                         "progress": float(evaluation.progress),
                         "stage": "complete" if evaluation.success else "transport" if all(contacts) else "pregrasp",
                         "success": bool(evaluation.success), "failure": bool(evaluation.failure)},
            contact_summary={"robot_count": int(sum(contacts)), "gripper_count": int(sum(contacts)),
                             "force_available": False, "max_normal_force_n": 0.0},
            joint_summary={"count": len(joints), "normalized": joints},
            simulation={"time_s": float(self.data.time)},
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._trace("reset_begin", seed=seed)
        super().reset(seed=seed)
        if options:
            raise ValueError(f"unknown ARX reset options: {sorted(options)}")
        self.data.qpos[:] = self._reset["qpos"]
        self.data.qvel[:] = self._reset["qvel"]
        self.data.time = float(self._reset["time"])
        self._initialize_robot()
        state = self._state()
        self._processor = ActionProcessor(self.task, state)
        self._last_command = state
        self._steps = 0
        self._finished = False
        evaluation = self._evaluator.reset()
        observation = self._observation()
        self._trace("reset_end", state=observation["state"].tolist())
        return observation, {
            "task": self.task.name,
            "terminal_reason": evaluation.reason,
            "progress": evaluation.progress,
            "evaluation": dataclasses.asdict(evaluation),
        }

    def step(self, action: np.ndarray):
        if self._finished or self._processor is None:
            raise RuntimeError("reset is required before step, and stepping after terminal is invalid")
        self._trace(
            "policy_step_begin",
            step=self._steps,
            raw_action=np.asarray(action).tolist(),
        )
        target, diagnostic = self._processor.process(action)
        self._trace("policy_action_processed", step=self._steps, target=target.tolist())
        commands = interpolate_commands(self._last_command, target, self.commands_per_action)
        for command_index, command in enumerate(commands):
            self._set_command(command)
            self._trace(
                "physics_step_begin",
                step=self._steps,
                command_index=command_index,
                sim_time=float(self.data.time),
                command=command.tolist(),
                ctrl=self.data.ctrl.tolist(),
                qpos_min=float(np.min(self.data.qpos)),
                qpos_max=float(np.max(self.data.qpos)),
                qvel_abs_max=float(np.max(np.abs(self.data.qvel))),
            )
            mujoco.mj_step(self.model, self.data, self.physics_steps_per_command)
            self._trace(
                "physics_step_end",
                step=self._steps,
                command_index=command_index,
                sim_time=float(self.data.time),
                qpos_min=float(np.min(self.data.qpos)),
                qpos_max=float(np.max(self.data.qpos)),
                qvel_abs_max=float(np.max(np.abs(self.data.qvel))),
            )
        self._last_command = target
        self._steps += 1
        evaluation = self._evaluator.evaluate()
        terminated = evaluation.success or evaluation.failure
        truncated = self._steps >= self.task.max_steps and not terminated
        self._finished = terminated or truncated
        info = {
            "task": self.task.name,
            "terminal_reason": evaluation.reason or ("time_limit" if truncated else None),
            "progress": evaluation.progress,
            "evaluation": dataclasses.asdict(evaluation),
            "processed_action": target.copy(),
            "filter": diagnostic.__dict__ if hasattr(diagnostic, "__dict__") else {
                name: getattr(diagnostic, name) for name in diagnostic.__slots__
            },
        }
        observation = self._observation()
        self._trace(
            "policy_step_end",
            step=self._steps,
            terminated=terminated,
            truncated=truncated,
            success=evaluation.success,
        )
        return observation, float(evaluation.success), terminated, truncated, info

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._reset.close()
