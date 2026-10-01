#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run an immutable Real2Sim scene through the MuJoCo Rollout Runtime.

The referenced Real2Sim attempt contains objects but no robot or actuators. This
integration composes the actual reBot G1-D + Dex1-1 model from its external
asset bundle, restores the attempt's settled object state, and exposes the
result as a Gymnasium environment. It never writes into either input tree.

This is an integration/controller smoke test, not a trained manipulation
policy. Actions are the reBot adapter's 22 absolute joint targets.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

ENV_ID = "Real2SimRebotG1D-v0"
MODULE_NAME = "scripts.deployment.run_real2sim_mujoco_scene"
DEFAULT_ATTEMPT = Path(
    "/data4/zhengyikai/Real2Sim/runs/tubes_1_dr_test/attempts/attempt_002"
)
DEFAULT_REBOT_ROOT = Path("/data4/zhengyikai/reBot-DevArm-Grasp")


def _require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Real2Sim attempt is missing {relative}: {path}")
    return path


def build_combined_xml(source_xml: Path) -> str:
    """Return MJCF with a minimal Cartesian gripper, without editing source_xml."""
    root = ET.parse(source_xml).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {source_xml}")
    if root.find("actuator") is not None:
        raise ValueError("the minimal importer expects a scene without actuators")

    x_body = ET.SubElement(worldbody, "body", name="r2s_robot_x", pos="0 -0.45 1.1")
    ET.SubElement(
        x_body,
        "joint",
        name="r2s_robot_x_joint",
        type="slide",
        axis="1 0 0",
        range="-0.5 0.5",
        damping="20",
    )
    ET.SubElement(
        x_body,
        "inertial",
        pos="0 0 0",
        mass="0.05",
        diaginertia="0.0001 0.0001 0.0001",
    )
    y_body = ET.SubElement(x_body, "body", name="r2s_robot_y")
    ET.SubElement(
        y_body,
        "joint",
        name="r2s_robot_y_joint",
        type="slide",
        axis="0 1 0",
        range="-0.25 0.85",
        damping="20",
    )
    ET.SubElement(
        y_body,
        "inertial",
        pos="0 0 0",
        mass="0.05",
        diaginertia="0.0001 0.0001 0.0001",
    )
    z_body = ET.SubElement(y_body, "body", name="r2s_robot_z")
    ET.SubElement(
        z_body,
        "joint",
        name="r2s_robot_z_joint",
        type="slide",
        axis="0 0 1",
        range="-0.45 0.45",
        damping="20",
    )
    ET.SubElement(
        z_body,
        "geom",
        name="r2s_robot_palm",
        type="box",
        size="0.055 0.025 0.018",
        rgba="0.15 0.25 0.8 1",
        mass="0.4",
        contype="1",
        conaffinity="1",
    )
    for side, sign in (("left", 1), ("right", -1)):
        finger = ET.SubElement(
            z_body,
            "body",
            name=f"r2s_robot_{side}_finger",
            pos=f"0 {0.03 * sign} -0.065",
        )
        ET.SubElement(
            finger,
            "joint",
            name=f"r2s_robot_{side}_finger_joint",
            type="slide",
            axis=f"0 {sign} 0",
            range="0 0.055",
            damping="4",
        )
        ET.SubElement(
            finger,
            "geom",
            name=f"r2s_robot_{side}_finger_geom",
            type="box",
            size="0.012 0.008 0.055",
            rgba="0.1 0.1 0.15 1",
            mass="0.08",
            friction="1.5 0.01 0.001",
            contype="1",
            conaffinity="1",
        )

    actuator = ET.SubElement(root, "actuator")
    for axis in ("x", "y", "z"):
        ET.SubElement(
            actuator,
            "position",
            name=f"r2s_robot_{axis}_actuator",
            joint=f"r2s_robot_{axis}_joint",
            kp="800",
        )
    for side in ("left", "right"):
        ET.SubElement(
            actuator,
            "position",
            name=f"r2s_robot_{side}_finger_actuator",
            joint=f"r2s_robot_{side}_finger_joint",
            kp="250",
        )
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _load_dependencies() -> tuple[Any, Any]:
    try:
        import gymnasium
        import mujoco
    except ImportError as exc:
        raise RuntimeError("install this repository with the 'mujoco' extra") from exc
    return gymnasium, mujoco


class Real2SimCartesianGripperEnv(gym.Env):
    """Gymnasium-compatible environment for the immutable Real2Sim attempt."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        *,
        attempt_root: str,
        render_mode: str | None = None,
        width: int = 320,
        height: int = 240,
        camera_name: str = "overview",
        frame_skip: int = 10,
        max_episode_steps: int = 500,
    ) -> None:
        gymnasium, mujoco = _load_dependencies()
        self._gymnasium = gymnasium
        self._mujoco = mujoco
        self.attempt_root = Path(attempt_root).expanduser().resolve(strict=True)
        scene_xml = _require_file(self.attempt_root, "scene/scene.xml")
        settled_path = _require_file(self.attempt_root, "output/settled_state.npz")
        self._source_model = mujoco.MjModel.from_xml_path(str(scene_xml))
        combined_xml = build_combined_xml(scene_xml)
        self.model = mujoco.MjModel.from_xml_string(combined_xml)
        self.data = mujoco.MjData(self.model)
        with np.load(settled_path) as state:
            self._settled_qpos = np.asarray(state["qpos"], dtype=np.float64).copy()
            self._settled_qvel = np.asarray(state["qvel"], dtype=np.float64).copy()
        if self._settled_qpos.shape != (self._source_model.nq,) or self._settled_qvel.shape != (
            self._source_model.nv,
        ):
            raise ValueError("settled state dimensions do not match the source MJCF")
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)
        self.camera_name = str(camera_name)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0
        self._renderer = None
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-0.5, -0.25, -0.45, 0.0], dtype=np.float32),
            high=np.array([0.5, 0.85, 0.45, 0.055], dtype=np.float32),
            dtype=np.float32,
        )
        state_dim = self.model.nq + self.model.nv + 2 * self.model.nu
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float64
        )

    def _observation(self) -> np.ndarray:
        # qpos/qvel plus two copies of the five actuator targets. Keeping this
        # fixed numeric vector makes the smoke environment's schema explicit.
        target = np.asarray(self.data.ctrl, dtype=np.float64)
        return np.concatenate((self.data.qpos, self.data.qvel, target, target))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        if options:
            unknown = sorted(options)
            raise ValueError(f"unsupported reset options: {unknown}")
        self._mujoco.mj_resetData(self.model, self.data)
        # The robot is appended after all original bodies, preserving this prefix.
        self.data.qpos[: self._source_model.nq] = self._settled_qpos
        self.data.qvel[: self._source_model.nv] = self._settled_qvel
        self.data.ctrl[:] = np.array([0.0, 0.45, 0.15, 0.04, 0.04])
        self._mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._observation(), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        command = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError("action must be a finite [x, y, z, gripper_opening] vector")
        self.data.ctrl[:] = np.array([*command[:3], command[3], command[3]])
        for _ in range(self.frame_skip):
            self._mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        truncated = self._step_count >= self.max_episode_steps
        return self._observation(), 0.0, not finite, truncated, {
            "is_success": False,
            "control_mode": "absolute_cartesian_gripper",
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _load_rebot_adapter(asset_root: Path) -> Any:
    adapter = asset_root / "simulation/g1d_mujoco_env.py"
    if not adapter.is_file():
        raise FileNotFoundError(f"reBot adapter does not exist: {adapter}")
    module_name = f"_real2sim_rebot_{hashlib.sha256(str(adapter).encode()).hexdigest()[:12]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, adapter)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reBot adapter: {adapter}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _name_real2sim_free_joints(root: ET.Element) -> None:
    """Give unnamed Real2Sim free joints stable names for state restoration."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Real2Sim MJCF has no worldbody")
    for body in worldbody.iter("body"):
        body_name = body.get("name")
        if not body_name:
            continue
        for joint in list(body.findall("freejoint")) + list(body.findall("joint")):
            if not joint.get("name"):
                joint.set("name", f"real2sim_{body_name}_free")


def _camera_xyaxes(position: np.ndarray, target: np.ndarray) -> str:
    forward = target - position
    forward /= np.linalg.norm(forward)
    camera_z = -forward
    camera_x = np.cross(np.array([0.0, 0.0, 1.0]), camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    values = np.concatenate((camera_x, camera_y))
    return " ".join(f"{value:.9g}" for value in values)


class Real2SimRebotEnv(gym.Env):
    """Actual reBot G1-D + Dex1-1 composed with an immutable Real2Sim scene."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        *,
        attempt_root: str,
        rebot_asset_root: str,
        robot_base_position: list[float] | tuple[float, float, float] = (0.0, -0.85, 0.12),
        robot_base_yaw_deg: float = 90.0,
        render_mode: str | None = None,
        width: int = 320,
        height: int = 240,
        camera_name: str = "overview",
        frame_skip: int = 5,
        max_episode_steps: int = 500,
    ) -> None:
        import yaml

        self.attempt_root = Path(attempt_root).expanduser().resolve(strict=True)
        self.rebot_asset_root = Path(rebot_asset_root).expanduser().resolve(strict=True)
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)
        self.camera_name = str(camera_name)
        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0
        scene_xml = _require_file(self.attempt_root, "scene/scene.xml")
        settled_path = _require_file(self.attempt_root, "output/settled_state.npz")
        source_model = self._load_source_model(scene_xml)
        with np.load(settled_path) as state:
            settled_qpos = np.asarray(state["qpos"], dtype=np.float64).copy()
            settled_qvel = np.asarray(state["qvel"], dtype=np.float64).copy()
        if settled_qpos.shape != (source_model.nq,) or settled_qvel.shape != (
            source_model.nv,
        ):
            raise ValueError("settled state dimensions do not match the Real2Sim MJCF")

        config_path = self.rebot_asset_root / "config/g1d_mujoco.yaml"
        with config_path.open("r", encoding="utf-8") as stream:
            config = copy.deepcopy(yaml.safe_load(stream)["simulation"])
        config["render"] = {
            **dict(config.get("render") or {}),
            "width": self.width,
            "height": self.height,
            "camera": self.camera_name,
        }
        config["frame_skip"] = int(frame_skip)
        adapter = _load_rebot_adapter(self.rebot_asset_root)
        base_position = np.asarray(robot_base_position, dtype=np.float64)
        if base_position.shape != (3,) or not np.isfinite(base_position).all():
            raise ValueError("robot_base_position must contain three finite values")
        base_yaw_deg = float(robot_base_yaw_deg)
        if not np.isfinite(base_yaw_deg):
            raise ValueError("robot_base_yaw_deg must be finite")
        source_joint_state = self._source_joint_state(
            source_model, settled_qpos, settled_qvel
        )

        parent = self

        class ComposedG1DEnv(adapter.G1DMujocoEnv):
            def _write_scene(self, converted_mjcf: Path, scene_path: Path) -> None:
                super()._write_scene(converted_mjcf, scene_path)
                parent._compose_scene(
                    scene_path, scene_xml, base_position, base_yaw_deg
                )

        self._env = ComposedG1DEnv(config, headless=True)
        self.model = self._env.model
        self.data = self._env.data
        self._action_names = tuple(adapter.CONTROLLED_JOINTS)
        self._restore_source_state(source_joint_state)
        low = np.asarray(
            [self._env._joint_indices[name].lower for name in self._action_names],
            dtype=np.float32,
        )
        high = np.asarray(
            [self._env._joint_indices[name].upper for name in self._action_names],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        state_dim = self.model.nq + self.model.nv + len(self._action_names)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float64
        )

    @staticmethod
    def _load_source_model(scene_xml: Path) -> Any:
        import mujoco

        return mujoco.MjModel.from_xml_path(str(scene_xml))

    @staticmethod
    def _source_joint_state(
        model: Any, qpos: np.ndarray, qvel: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        import mujoco

        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            body_id = int(model.jnt_bodyid[joint_id])
            if name is None:
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                name = f"real2sim_{body_name}_free"
            qadr = int(model.jnt_qposadr[joint_id])
            dadr = int(model.jnt_dofadr[joint_id])
            next_qadr = int(model.jnt_qposadr[joint_id + 1]) if joint_id + 1 < model.njnt else model.nq
            next_dadr = int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < model.njnt else model.nv
            result[name] = (qpos[qadr:next_qadr].copy(), qvel[dadr:next_dadr].copy())
        return result

    @staticmethod
    def _compose_scene(
        rebot_scene: Path,
        real2sim_scene: Path,
        base_position: np.ndarray,
        base_yaw_deg: float,
    ) -> None:
        rebot_tree = ET.parse(rebot_scene)
        rebot_root = rebot_tree.getroot()
        rebot_world = rebot_root.find("worldbody")
        rebot_asset = rebot_root.find("asset")
        if rebot_world is None:
            raise ValueError("generated reBot scene has no worldbody")
        robot = next(
            (body for body in rebot_world.findall("body") if body.get("name") == "AGV_link"),
            None,
        )
        if robot is None:
            raise ValueError("generated reBot scene has no AGV_link root")
        # Remove the adapter's demonstration scene, retaining only the robot.
        for child in list(rebot_world):
            rebot_world.remove(child)
        robot.set("pos", " ".join(f"{value:.9g}" for value in base_position))
        half_yaw = np.deg2rad(base_yaw_deg) * 0.5
        robot.set(
            "quat",
            f"{np.cos(half_yaw):.9g} 0 0 {np.sin(half_yaw):.9g}",
        )
        rebot_world.append(robot)

        source_tree = ET.parse(real2sim_scene)
        source_root = source_tree.getroot()
        _name_real2sim_free_joints(source_root)
        source_asset = source_root.find("asset")
        source_world = source_root.find("worldbody")
        if source_world is None:
            raise ValueError("Real2Sim scene has no worldbody")
        # The source overview tightly frames the table and crops the robot.
        # Retarget only this camera to show the complete imported setup.
        overview = next(
            (
                camera
                for camera in source_world.findall("camera")
                if camera.get("name") == "overview"
            ),
            None,
        )
        if overview is not None:
            camera_position = np.array([-1.9, -2.35, 1.85], dtype=np.float64)
            camera_target = np.array([0.0, -0.1, 0.95], dtype=np.float64)
            overview.set("pos", " ".join(f"{value:.9g}" for value in camera_position))
            overview.set("xyaxes", _camera_xyaxes(camera_position, camera_target))
            overview.set("fovy", "48")
        if source_asset is not None:
            if rebot_asset is None:
                rebot_asset = ET.Element("asset")
                rebot_root.insert(0, rebot_asset)
            for child in list(source_asset):
                rebot_asset.append(copy.deepcopy(child))
        for child in list(source_world):
            rebot_world.append(copy.deepcopy(child))
        body_names = {
            body.get("name") for body in rebot_world.iter("body") if body.get("name")
        }
        contact = rebot_root.find("contact")
        if contact is not None:
            for exclude in list(contact.findall("exclude")):
                if exclude.get("body1") not in body_names or exclude.get("body2") not in body_names:
                    contact.remove(exclude)
        ET.indent(rebot_tree, space="  ")
        rebot_tree.write(rebot_scene, encoding="utf-8", xml_declaration=True)

    def _restore_source_state(
        self, state: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        import mujoco

        for name, (qpos, qvel) in state.items():
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"composed model is missing Real2Sim joint {name!r}")
            qadr = int(self.model.jnt_qposadr[joint_id])
            dadr = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qadr : qadr + len(qpos)] = qpos
            self.data.qvel[dadr : dadr + len(qvel)] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _observation(self) -> np.ndarray:
        targets = np.asarray(
            [self._env.targets[name] for name in self._action_names], dtype=np.float64
        )
        return np.concatenate((self.data.qpos, self.data.qvel, targets))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        if options:
            raise ValueError(f"unsupported reset options: {sorted(options)}")
        self._env.reset()
        source_model = self._load_source_model(
            _require_file(self.attempt_root, "scene/scene.xml")
        )
        with np.load(_require_file(self.attempt_root, "output/settled_state.npz")) as state:
            source_state = self._source_joint_state(source_model, state["qpos"], state["qvel"])
        self._restore_source_state(source_state)
        self._step_count = 0
        return self._observation(), {"is_success": False, "robot": "reBot G1-D + Dex1-1"}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        vector = np.asarray(action, dtype=np.float64)
        if vector.shape != (len(self._action_names),) or not np.isfinite(vector).all():
            raise ValueError(f"action must be a finite {(len(self._action_names),)} vector")
        vector = np.clip(vector, self.action_space.low, self.action_space.high)
        self._env.set_joint_targets(dict(zip(self._action_names, vector, strict=True)))
        self._env.step()
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        return self._observation(), 0.0, False, truncated, {
            "is_success": False,
            "robot": "reBot G1-D + Dex1-1",
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return np.asarray(self._env.render(camera_name=self.camera_name)["img"], dtype=np.uint8)

    def close(self) -> None:
        self._env.close()


def _register_environment() -> None:
    gymnasium, _mujoco = _load_dependencies()
    if ENV_ID not in gymnasium.registry:
        gymnasium.register(
            id=ENV_ID,
            entry_point=f"{MODULE_NAME}:Real2SimRebotEnv",
        )


async def _run_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
    from rollout_runtime.api.messages import EnvSpecMsg
    from rollout_runtime.config.schema import load_config
    from rollout_runtime.core import payload as payload_module
    from rollout_runtime.launch.local import build_local_components

    config = load_config("local_fake")
    env_config = {
        "provider": "gymnasium",
        "env_id": f"{MODULE_NAME}:{ENV_ID}",
        "env_kwargs": {
            "attempt_root": str(args.attempt_root),
            "rebot_asset_root": str(args.rebot_asset_root),
            "robot_base_position": args.robot_base_position,
            "robot_base_yaw_deg": args.robot_base_yaw_deg,
            "frame_skip": args.frame_skip,
        },
        "observation_mode": "rgb_state",
        "render_mode": "rgb_array",
        "render_backend": args.render_backend,
        "camera_name": args.camera,
        "image_width": args.width,
        "image_height": args.height,
        "action_dim": 22,
        "chunk_size": 1,
        "max_episode_steps": args.steps,
        "process_isolation": True,
        "success_mode": "info_key",
        "success_info_key": "is_success",
    }
    config.env_family = "mujoco"
    config.env_config = env_config
    runtime = build_local_components(config)
    print("[real2sim] starting LocalRuntime", flush=True)
    await runtime.start()
    facade = RuntimeGymEnv(
        runtime.gateway,
        EnvSpecMsg(env_family="mujoco", env_config=env_config),
        application_id="real2sim-mujoco-smoke",
    )
    executed = 0
    frame_count = 0
    last_info: dict[str, Any] = {}
    writer = imageio.get_writer(
        args.video_output,
        fps=args.video_fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    def append_frame(observation: Any) -> None:
        nonlocal frame_count
        if observation.main_image is None:
            raise RuntimeError("Runtime observation is missing its main RGB image")
        frame = payload_module.decode_image(observation.main_image)
        if frame.shape != (args.height, args.width, 3):
            raise RuntimeError(
                "Runtime frame has unexpected shape: "
                f"expected {(args.height, args.width, 3)}, got {frame.shape}"
            )
        writer.append_data(frame)
        frame_count += 1

    try:
        print("[real2sim] resetting imported scene", flush=True)
        observation, _reset_info = await facade.reset(
            seed=args.seed, options={"instruction": args.instruction}
        )
        append_frame(observation)
        hold_action = np.asarray(observation.state[-22:], dtype=np.float32)
        print("[real2sim] executing explicit reBot joint-target actions", flush=True)
        while executed < args.steps:
            observation, _reward, terminated, truncated, last_info = await facade.step(
                hold_action
            )
            append_frame(observation)
            executed += int(last_info["executed_horizon"])
            if terminated or truncated:
                break
    finally:
        writer.close()
        print("[real2sim] closing Runtime session", flush=True)
        await facade.close()
        print("[real2sim] stopping Runtime gateway", flush=True)
        await runtime.gateway.stop()
        print("[real2sim] closing LocalRuntime", flush=True)
        await runtime.aclose()
    video_bytes = args.video_output.read_bytes()
    return {
        "mode": "runtime",
        "executed_steps": executed,
        "last_info": last_info,
        "video": {
            "path": str(args.video_output),
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "fps": args.video_fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / args.video_fps,
            "width": args.width,
            "height": args.height,
            "size_bytes": len(video_bytes),
            "sha256": hashlib.sha256(video_bytes).hexdigest(),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--rebot-asset-root", type=Path, default=DEFAULT_REBOT_ROOT)
    parser.add_argument(
        "--robot-base-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, -0.85, 0.12),
    )
    parser.add_argument("--robot-base-yaw-deg", type=float, default=90.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--render-backend", choices=("egl", "osmesa", "glfw"), default="egl")
    parser.add_argument(
        "--instruction",
        default="Inspect the imported tube scene with reBot G1-D and Dex1-1.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.attempt_root = args.attempt_root.expanduser().resolve(strict=True)
    args.rebot_asset_root = args.rebot_asset_root.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.video_output is None:
        args.video_output = args.output_dir / "episode.mp4"
    else:
        args.video_output = args.video_output.expanduser().resolve()
    if args.steps < 1 or args.frame_skip < 1 or args.width < 1 or args.height < 1:
        raise ValueError("steps, frame-skip, width, and height must be positive")
    if args.video_fps <= 0.0:
        raise ValueError("video-fps must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("width and height must be even for yuv420p MP4 output")
    # Ensure all generated files are outside the immutable input tree.
    if args.output_dir == args.attempt_root or args.attempt_root in args.output_dir.parents:
        raise ValueError("--output-dir must not be inside --attempt-root")
    if args.video_output == args.attempt_root or args.attempt_root in args.video_output.parents:
        raise ValueError("--video-output must not be inside --attempt-root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_output.parent.mkdir(parents=True, exist_ok=True)
    _register_environment()
    result = asyncio.run(_run_runtime(args))
    result.update(
        {
            "attempt_root": str(args.attempt_root),
            "source_scene": str(_require_file(args.attempt_root, "scene/scene.xml")),
            "robot": "reBot G1-D + Dex1-1",
            "robot_base_position": list(args.robot_base_position),
            "robot_base_yaw_deg": args.robot_base_yaw_deg,
            "action_contract": "22 absolute joint targets in reBot CONTROLLED_JOINTS order",
            "input_was_modified": False,
        }
    )
    output = args.output_dir / "real2sim-runtime-summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**result, "summary": str(output)}, indent=2, sort_keys=True))
    return 0


_register_environment()


if __name__ == "__main__":
    raise SystemExit(main())
