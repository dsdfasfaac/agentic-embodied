# Copyright (c) 2026 Zetta Contributors
"""Staged MuJoCo/Gymnasium/Runtime acceptance probes.

The script imports neither Gymnasium nor MuJoCo until after the requested
render backend has been exported. Every stage emits one JSON object to stdout
and can additionally persist it with ``--output``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _configure_egl() -> None:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def _versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for module_name in ("mujoco", "gymnasium", "ray"):
        try:
            module = __import__(module_name)
        except ImportError:
            versions[module_name] = None
        else:
            versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def _gpu_snapshot() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    snapshots = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            snapshots.append(
                dict(
                    zip(
                        (
                            "index",
                            "name",
                            "driver",
                            "memory_used_mb",
                            "memory_total_mb",
                        ),
                        fields,
                        strict=True,
                    )
                )
            )
    return snapshots


def _rss_bytes(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _git_snapshot() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return ""
        return result.stdout.rstrip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(status),
        "changed_paths": [line[3:] for line in status.splitlines() if len(line) > 3],
    }


def _base_result(stage: str, started: float) -> dict[str, Any]:
    return {
        "stage": stage,
        "ok": True,
        "recorded_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "git": _git_snapshot(),
        "versions": _versions(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
            )
        },
        "gpu": _gpu_snapshot(),
    }


def native_probe() -> dict[str, Any]:
    """Run raw MuJoCo physics and EGL rendering without Gymnasium."""
    started = time.perf_counter()
    _configure_egl()
    import mujoco

    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="1 1 .1" rgba=".3 .3 .3 1"/>
        <body pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -.5" size=".05"
                rgba=".8 .2 .2 1"/>
        </body>
      </worldbody>
      <actuator><motor joint="hinge" ctrlrange="-1 1"/></actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    try:
        renderer.update_scene(data)
        image = np.asarray(renderer.render())
    finally:
        renderer.close()
    if not np.isfinite(data.qpos).all():
        raise RuntimeError("native MuJoCo state contains NaN or Inf")
    if image.shape != (120, 160, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"unexpected native RGB: {image.shape} {image.dtype}")
    return {
        **_base_result("native", started),
        "effective_config": {"steps": 100, "height": 120, "width": 160},
        "steps": 100,
        "sim_time": float(data.time),
        "state_finite": True,
        "rgb_shape": list(image.shape),
        "rgb_dtype": str(image.dtype),
        "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _gym_rollout(seed: int, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import gymnasium as gym

    env = gym.make("InvertedPendulum-v5")
    try:
        observation, _info = env.reset(seed=seed)
        states = [np.asarray(observation, dtype=np.float64)]
        rewards: list[float] = []
        for action in actions:
            observation, reward, terminated, truncated, _info = env.step(action)
            states.append(np.asarray(observation, dtype=np.float64))
            rewards.append(float(reward))
            if terminated or truncated:
                break
    finally:
        env.close()
    return np.stack(states), np.asarray(rewards, dtype=np.float64)


def gymnasium_probe() -> dict[str, Any]:
    """Verify Gymnasium lifecycle and deterministic replay."""
    started = time.perf_counter()
    _configure_egl()
    actions = np.zeros((100, 1), dtype=np.float32)
    first_states, first_rewards = _gym_rollout(23, actions)
    second_states, second_rewards = _gym_rollout(23, actions)
    state_delta = float(np.max(np.abs(first_states - second_states)))
    reward_delta = float(np.max(np.abs(first_rewards - second_rewards)))
    if state_delta > 1e-12 or reward_delta > 1e-12:
        raise RuntimeError(
            f"Gymnasium replay diverged: state={state_delta}, reward={reward_delta}"
        )

    import gymnasium as gym

    rendered = gym.make(
        "InvertedPendulum-v5", render_mode="rgb_array", width=160, height=120
    )
    try:
        rendered.reset(seed=23)
        image = np.asarray(rendered.render())
    finally:
        rendered.close()
    if image.shape != (120, 160, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"unexpected Gymnasium RGB: {image.shape} {image.dtype}")
    return {
        **_base_result("gymnasium", started),
        "effective_config": {
            "env_id": "InvertedPendulum-v5",
            "seed": 23,
            "action": "zeros[100,1]",
            "height": 120,
            "width": 160,
        },
        "steps": int(first_rewards.size),
        "state_max_abs_diff": state_delta,
        "reward_max_abs_diff": reward_delta,
        "state_sha256": hashlib.sha256(first_states.tobytes()).hexdigest(),
        "reward_sha256": hashlib.sha256(first_rewards.tobytes()).hexdigest(),
        "rgb_shape": list(image.shape),
        "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _env_config(*, rgb: bool, max_episode_steps: int = 1000) -> dict[str, Any]:
    return {
        "provider": "gymnasium",
        "env_id": "InvertedPendulum-v5",
        "observation_mode": "rgb_state" if rgb else "state",
        "render_mode": "rgb_array" if rgb else None,
        "render_backend": "egl",
        "camera_name": None,
        "image_width": 160,
        "image_height": 120,
        "action_dim": 1,
        "chunk_size": 4,
        "clip_actions": True,
        "max_episode_steps": max_episode_steps,
        "core_form": "per_slot",
        "process_isolation": rgb,
        "rpc_timeout_s": 120.0,
        "instruction": "Keep the pendulum upright",
        "success_mode": "none",
        "return_all_frames": False,
    }


def core_probe() -> dict[str, Any]:
    """Exercise the real backend directly, including dynamic RGB slots."""
    started = time.perf_counter()
    _configure_egl()
    from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
    from rollout_runtime.backends.mujoco_env import MujocoEnvCore
    from rollout_runtime.core import payload as payload_module

    state_core = MujocoEnvCore()
    state_core.build(
        EnvSpecMsg(env_family="mujoco", env_config=_env_config(rgb=False)),
        num_envs=1,
        seed_offset=0,
    )
    try:
        first = state_core.reset([0], ResetSpec(seed=31))[0]
        outcome = state_core.chunk_step([0], [np.zeros((4, 1), dtype=np.float32)])[0]
        if outcome.executed_horizon != 4 or len(outcome.per_step or ()) != 4:
            raise RuntimeError("backend core reported an incorrect executed horizon")
        if outcome.success is not None:
            raise RuntimeError("return-only MuJoCo task fabricated binary success")
        state_dim = len(first.state)
        episode_return = outcome.reward
    finally:
        state_core.close()

    rgb_core = MujocoEnvCore()
    rgb_core.build(
        EnvSpecMsg(env_family="mujoco", env_config=_env_config(rgb=True)),
        num_envs=1,
        seed_offset=0,
    )
    processes = [rgb_core._slots[0].session._process]
    try:
        rgb_core.add_slot(100)
        processes.append(rgb_core._slots[1].session._process)
        observations = rgb_core.reset([0, 1], ResetSpec(seed=32))
        images = [
            payload_module.decode_image(observation.main_image)
            for observation in observations
            if observation.main_image is not None
        ]
        if len(images) != 2 or any(image.shape != (120, 160, 3) for image in images):
            raise RuntimeError("backend RGB observation shape mismatch")
        rgb_core.remove_slot(1)
    finally:
        rgb_core.close()
    if any(process.is_alive() for process in processes):
        raise RuntimeError("backend close left a MuJoCo slot process alive")
    return {
        **_base_result("core", started),
        "effective_config": {
            "state": _env_config(rgb=False),
            "rgb": _env_config(rgb=True),
            "seeds": [31, 32],
        },
        "state_dim": state_dim,
        "episode_return": episode_return,
        "executed_horizon": 4,
        "rgb_slots": 2,
        "children_reclaimed": True,
    }


async def local_runtime_probe(total_steps: int) -> dict[str, Any]:
    """Run explicit actions and fake-policy actions through LocalRuntime."""
    started = time.perf_counter()
    _configure_egl()
    from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
    from rollout_runtime.api.messages import EnvSpecMsg
    from rollout_runtime.config.schema import load_config
    from rollout_runtime.launch.local import build_local_components

    config = load_config("local_fake")
    config.env_family = "mujoco"
    config.env_config = _env_config(rgb=True, max_episode_steps=25)
    config.env_resource_hints = {"accelerator": True}
    config.env_worker.has_accelerator = True
    runtime = build_local_components(config)
    await runtime.start()
    spec = EnvSpecMsg(
        env_family="mujoco",
        env_config=dict(config.env_config),
        resource_hints={"accelerator": True},
    )
    facade = RuntimeGymEnv(runtime.gateway, spec, policy_id="fake")
    explicit_steps = 0
    policy_horizon = 0
    session_id = None
    try:
        await facade.reset(seed=41)
        session_id = str(facade.session_id)
        episode = 0
        while explicit_steps < total_steps:
            _obs, _reward, terminated, truncated, info = await facade.step(
                np.zeros((1,), dtype=np.float32)
            )
            explicit_steps += int(info["executed_horizon"])
            if terminated or truncated:
                episode += 1
                await facade.reset(seed=41 + episode)
        await facade.reset(seed=99)
        while True:
            _obs, _reward, terminated, truncated, info = await facade.step()
            policy_horizon += int(info["executed_horizon"])
            if terminated or truncated:
                break
        if len(runtime.env_workers[0].sessions) != 1:
            raise RuntimeError("LocalRuntime did not reuse exactly one session")
    finally:
        await facade.close()
        sessions_after_close = len(runtime.env_workers[0].sessions)
        await runtime.gateway.stop()
        await runtime.aclose()
    if sessions_after_close:
        raise RuntimeError("LocalRuntime leaked a session after close")
    return {
        **_base_result("local", started),
        "effective_config": {
            "env": _env_config(rgb=True, max_episode_steps=25),
            "resource_hints": {"accelerator": True},
        },
        "session_id": session_id,
        "explicit_horizon": explicit_steps,
        "policy_horizon": policy_horizon,
        "sessions_after_close": sessions_after_close,
    }


async def ray_probe(concurrency: int) -> dict[str, Any]:
    """Run concurrent RGB sessions in separately launched Ray actors."""
    started = time.perf_counter()
    _configure_egl()
    from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
    from rollout_runtime.api.messages import EnvSpecMsg
    from rollout_runtime.config.schema import load_config
    from rollout_runtime.launch.ray_launch import build_ray_components

    config = load_config("rtx4090_mujoco")
    suffix = uuid.uuid4().hex[:8]
    config.env_worker.group_name = f"mujoco-env-{suffix}"
    config.rollout_worker.group_name = f"mujoco-rollout-{suffix}"
    config.env_worker.max_sessions_per_rank = max(concurrency, 1)
    config.env_config = _env_config(rgb=True, max_episode_steps=32)
    runtime = build_ray_components(config)
    await runtime.start()
    spec = EnvSpecMsg(
        env_family="mujoco",
        env_config=dict(config.env_config),
        pool_size=concurrency,
        resource_hints={"accelerator": True},
    )
    facades = [
        RuntimeGymEnv(
            runtime.gateway,
            spec,
            application_id="mujoco-ray-smoke",
            client_session_key=f"ray-{suffix}-{index}",
            policy_id="fake",
        )
        for index in range(concurrency)
    ]
    try:
        await asyncio.gather(
            *(facade.reset(seed=100 + index) for index, facade in enumerate(facades))
        )
        results = await asyncio.gather(*(facade.step() for facade in facades))
        if any(result[4]["executed_horizon"] != 4 for result in results):
            raise RuntimeError("Ray policy step returned an unexpected horizon")
        registered = runtime.observed_ranks.copy()
        worker_entries = runtime.gateway.workers.snapshot().values()
        if not worker_entries or not all(
            entry.info.has_accelerator for entry in worker_entries
        ):
            raise RuntimeError("RGB EnvWorker was not registered with an accelerator")
    finally:
        await asyncio.gather(
            *(facade.close() for facade in facades), return_exceptions=True
        )
        await runtime.gateway.stop()
        await runtime.aclose()
    return {
        **_base_result("ray", started),
        "effective_config": {
            "preset": "rtx4090_mujoco",
            "env": _env_config(rgb=True, max_episode_steps=32),
            "resource_hints": {"accelerator": True},
        },
        "concurrency": concurrency,
        "observed_ranks": registered,
        "executed_horizon_per_session": 4,
    }


def soak_probe(
    episodes: int, steps_per_episode: int, concurrency: int
) -> dict[str, Any]:
    """Run a multi-slot EGL soak and record current RSS/GPU samples."""
    started = time.perf_counter()
    _configure_egl()
    from rollout_runtime.api.messages import EnvSpecMsg, ResetSpec
    from rollout_runtime.backends.mujoco_env import MujocoEnvCore

    config = _env_config(rgb=True, max_episode_steps=steps_per_episode)
    core = MujocoEnvCore()
    core.build(
        EnvSpecMsg(env_family="mujoco", env_config=config, pool_size=concurrency),
        num_envs=concurrency,
        seed_offset=0,
    )
    processes = [slot.session._process for slot in core._slots]
    samples: list[dict[str, Any]] = []
    total_steps = 0
    try:
        for episode in range(episodes):
            slot = episode % concurrency
            core.reset([slot], ResetSpec(seed=1000 + episode))
            episode_steps = 0
            while episode_steps < steps_per_episode:
                horizon = min(10, steps_per_episode - episode_steps)
                outcome = core.chunk_step(
                    [slot], [np.zeros((horizon, 1), dtype=np.float32)]
                )[0]
                episode_steps += outcome.executed_horizon
                total_steps += outcome.executed_horizon
                if outcome.terminated or outcome.truncated:
                    break
            if (episode + 1) % max(1, episodes // 10) == 0:
                rss_values = [
                    value
                    for value in (
                        _rss_bytes(os.getpid()),
                        *(_rss_bytes(p.pid) for p in processes),
                    )
                    if value is not None
                ]
                samples.append(
                    {
                        "episode": episode + 1,
                        "total_steps": total_steps,
                        "rss_bytes": sum(rss_values) if rss_values else None,
                        "gpu": _gpu_snapshot(),
                    }
                )
    finally:
        core.close()
    alive = [process.pid for process in processes if process.is_alive()]
    if alive:
        raise RuntimeError(f"soak left child processes alive: {alive}")
    rss_samples = [
        int(sample["rss_bytes"])
        for sample in samples
        if sample["rss_bytes"] is not None
    ]
    rss_growth_bytes = rss_samples[-1] - rss_samples[0] if len(rss_samples) > 1 else 0
    rss_growth_limit_bytes = 256 * 1024 * 1024
    if rss_growth_bytes > rss_growth_limit_bytes:
        raise RuntimeError(
            "soak RSS grew beyond the acceptance limit: "
            f"growth={rss_growth_bytes}, limit={rss_growth_limit_bytes}"
        )
    return {
        **_base_result("soak", started),
        "effective_config": config,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "total_steps": total_steps,
        "concurrency": concurrency,
        "resource_samples": samples,
        "rss_growth_bytes": rss_growth_bytes,
        "rss_growth_limit_bytes": rss_growth_limit_bytes,
        "infrastructure_invalid_episodes": 0,
        "children_reclaimed": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("native", "gymnasium", "core", "local", "ray", "soak")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.steps < 1 or args.episodes < 1 or args.steps_per_episode < 1:
        raise SystemExit("step and episode counts must be positive")
    if args.concurrency < 1:
        raise SystemExit("concurrency must be positive")
    if args.stage == "native":
        result = native_probe()
    elif args.stage == "gymnasium":
        result = gymnasium_probe()
    elif args.stage == "core":
        result = core_probe()
    elif args.stage == "local":
        result = asyncio.run(local_runtime_probe(args.steps))
    elif args.stage == "ray":
        result = asyncio.run(ray_probe(args.concurrency))
    else:
        result = soak_probe(args.episodes, args.steps_per_episode, args.concurrency)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
