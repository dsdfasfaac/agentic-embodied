# Copyright (c) 2026 Zetta Contributors
"""MuJoCo sessions and an optional process-isolated proxy.

The Runtime executes synchronous environment cores through ``asyncio.to_thread``.
That is safe for state-only MuJoCo, but RGB rendering owns native EGL contexts
whose lifetime must not bounce between arbitrary worker threads. This module
therefore offers a small blocking session interface and a ``spawn``-based proxy;
the child creates, steps, renders, and closes one Gymnasium environment on one
thread for its entire lifetime.

Neither Gymnasium nor MuJoCo is imported at module import time. The optional
``mujoco`` extra is required only when a real session is constructed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback
from multiprocessing import connection
from typing import Any

import numpy as np

__all__ = [
    "GymMujocoSession",
    "MujocoSessionError",
    "RemoteMujocoSession",
    "create_mujoco_session",
    "spawn_mujoco_session",
]

_SPAWN_CONTEXT = mp.get_context("spawn")
_SHUTDOWN = "__shutdown__"


class MujocoSessionError(RuntimeError):
    """A real or remote Gymnasium MuJoCo session failed."""


def _configure_render_environment(config: dict[str, Any]) -> None:
    """Select the native renderer before Gymnasium can import MuJoCo."""
    if config.get("render_mode") != "rgb_array":
        return
    render_backend = str(config.get("render_backend") or "egl")
    os.environ["MUJOCO_GL"] = render_backend
    if render_backend == "egl":
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif render_backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _load_gymnasium() -> Any:
    """Import Gymnasium lazily with an actionable optional-extra error."""
    try:
        import gymnasium
    except ImportError as exc:  # pragma: no cover - exercised in isolated envs
        raise MujocoSessionError(
            "Gymnasium MuJoCo dependencies are unavailable; install "
            "the project with the 'mujoco' extra"
        ) from exc
    return gymnasium


class GymMujocoSession:
    """One in-process Gymnasium environment with a blocking API."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create the environment and freeze its action-space descriptor.

        Args:
            config: Plain configuration produced by ``MujocoEnvConfig``.

        Raises:
            MujocoSessionError: The environment cannot be created or has an
                unsupported action space.
        """
        self._config = dict(config)
        self._closed = False
        _configure_render_environment(self._config)
        gymnasium = _load_gymnasium()
        kwargs = dict(self._config.get("env_kwargs") or {})
        render_mode = self._config.get("render_mode")
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
            kwargs["width"] = int(self._config["image_width"])
            kwargs["height"] = int(self._config["image_height"])
            camera_name = self._config.get("camera_name")
            if camera_name is not None:
                kwargs["camera_name"] = str(camera_name)
        max_episode_steps = self._config.get("max_episode_steps")
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = int(max_episode_steps)
        try:
            self._env = gymnasium.make(str(self._config["env_id"]), **kwargs)
        except BaseException as exc:  # noqa: BLE001 - native constructor boundary
            raise MujocoSessionError(
                f"cannot construct Gymnasium environment "
                f"{self._config['env_id']!r}: {type(exc).__name__}: {exc}"
            ) from exc

        camera_name = self._config.get("camera_name")
        unwrapped = getattr(self._env, "unwrapped", self._env)
        model = getattr(unwrapped, "model", None)
        if camera_name is not None and model is not None:
            try:
                import mujoco

                camera_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    str(camera_name),
                )
            except BaseException as exc:  # noqa: BLE001 - native model boundary
                self.close()
                raise MujocoSessionError(
                    f"cannot validate MuJoCo camera {camera_name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if camera_id < 0:
                self.close()
                raise MujocoSessionError(
                    f"MuJoCo camera {camera_name!r} does not exist in "
                    f"environment {self._config['env_id']!r}"
                )

        action_space = getattr(self._env, "action_space", None)
        shape = tuple(int(item) for item in getattr(action_space, "shape", ()) or ())
        if len(shape) != 1 or shape[0] < 1:
            self.close()
            raise MujocoSessionError(
                "MuJoCo v1 requires a one-dimensional continuous action space; "
                f"got shape={shape!r}"
            )
        try:
            low = np.broadcast_to(
                np.asarray(action_space.low, dtype=np.float32), shape
            ).copy()
            high = np.broadcast_to(
                np.asarray(action_space.high, dtype=np.float32), shape
            ).copy()
        except (AttributeError, TypeError, ValueError) as exc:
            self.close()
            raise MujocoSessionError(
                "MuJoCo v1 requires a continuous Box-like action space with "
                "numeric low/high bounds"
            ) from exc
        if np.isnan(low).any() or np.isnan(high).any() or np.any(low > high):
            self.close()
            raise MujocoSessionError("action-space bounds are invalid")
        self.descriptor = {
            "action_shape": shape,
            "action_low": low,
            "action_high": high,
        }

    def reset(
        self, *, seed: int | None, options: dict[str, Any] | None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the Gymnasium environment."""
        try:
            result = self._env.reset(seed=seed, options=dict(options or {}))
        except BaseException as exc:  # noqa: BLE001 - native simulator boundary
            raise MujocoSessionError(
                f"Gymnasium reset failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(result, tuple) or len(result) != 2:
            raise MujocoSessionError("Gymnasium reset must return (observation, info)")
        observation, info = result
        return observation, dict(info or {})

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one Gymnasium control step."""
        try:
            result = self._env.step(np.asarray(action, dtype=np.float32))
        except BaseException as exc:  # noqa: BLE001 - native simulator boundary
            raise MujocoSessionError(
                f"Gymnasium step failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(result, tuple) or len(result) != 5:
            raise MujocoSessionError(
                "Gymnasium step must return "
                "(observation, reward, terminated, truncated, info)"
            )
        observation, reward, terminated, truncated, info = result
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info or {}),
        )

    def render(self) -> np.ndarray | None:
        """Render the current RGB frame when configured."""
        if self._config.get("render_mode") != "rgb_array":
            return None
        try:
            frame = self._env.render()
        except BaseException as exc:  # noqa: BLE001 - native renderer boundary
            raise MujocoSessionError(
                f"Gymnasium render failed: {type(exc).__name__}: {exc}"
            ) from exc
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise MujocoSessionError(
                "Gymnasium rgb_array render must return uint8 HWC with 3 or 4 "
                f"channels, got shape={array.shape}, dtype={array.dtype}"
            )
        if array.shape[-1] == 4:
            array = array[..., :3]
        return np.ascontiguousarray(array)

    def close(self) -> None:
        """Close the environment idempotently."""
        if self._closed:
            return
        self._closed = True
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()


def create_mujoco_session(config: dict[str, Any]) -> Any:
    """Create the fixed session implementation selected by ``provider``."""
    provider = str(config.get("provider") or "gymnasium")
    if provider == "gymnasium":
        return GymMujocoSession(config)
    if provider == "rebot_g1d":
        from rollout_runtime.backends.rebot_g1d_session import RebotG1DSession

        return RebotG1DSession(config)
    if provider == "arx_ac_one":
        return ArxAcOneSession(config)
    raise MujocoSessionError(f"unsupported MuJoCo provider {provider!r}")


class ArxAcOneSession:
    """Blocking adapter around the prepared ARX Gymnasium environment."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from robots.arx.environment import ArxMujocoEnv

            self._env = ArxMujocoEnv(
                prepared_scene_bundle=str(config["arx_prepared_scene_bundle"]),
                mapping_path=str(config["arx_mapping_path"]),
                task_manifest=str(config["arx_task_manifest"]),
                camera_names=dict(config["camera_names"]),
                image_width=int(config["image_width"]),
                image_height=int(config["image_height"]),
            )
        except BaseException as exc:
            raise MujocoSessionError(f"cannot construct ARX AC one environment: {type(exc).__name__}: {exc}") from exc
        self.descriptor = {
            "action_shape": (14,),
            "action_low": np.asarray(self._env.action_space.low, dtype=np.float32),
            "action_high": np.asarray(self._env.action_space.high, dtype=np.float32),
            "action_names": [f"arx_policy_{index}" for index in range(14)],
        }
        self._last = None

    def reset(self, *, seed: int | None, options: dict[str, Any] | None):
        self._last, info = self._env.reset(seed=seed, options=options)
        return self._last, info

    def step(self, action: np.ndarray):
        result = self._env.step(action)
        self._last = result[0]
        info = dict(result[4])
        info["is_success"] = bool(result[2] and info.get("terminal_reason") == "success")
        return result[:4] + (info,)

    def render(self) -> np.ndarray | None:
        if self._last is None:
            return None
        return np.asarray(self._last["front_rgb"])

    def close(self) -> None:
        self._env.close()


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _worker_main(conn: connection.Connection, config: dict[str, Any]) -> None:
    """Own one session in a spawned process and serve blocking RPCs."""
    _configure_render_environment(config)
    session: Any = None
    try:
        session = create_mujoco_session(config)
    except BaseException as exc:  # noqa: BLE001 - child construction boundary
        try:
            conn.send(["error", _error_payload(exc)])
        except (BrokenPipeError, EOFError, OSError):
            pass
        conn.close()
        return
    try:
        conn.send(["ready", session.descriptor])
        while True:
            try:
                command, payload = conn.recv()
            except EOFError:
                break
            try:
                if command == _SHUTDOWN:
                    conn.send(["ok", None])
                    break
                if command == "reset":
                    result = session.reset(
                        seed=payload.get("seed"), options=payload.get("options")
                    )
                elif command == "step":
                    result = session.step(payload["action"])
                elif command == "render":
                    result = session.render()
                else:
                    raise ValueError(f"unknown MuJoCo session command {command!r}")
                conn.send(["ok", result])
            except BaseException as exc:  # noqa: BLE001 - RPC error boundary
                conn.send(["error", _error_payload(exc)])
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        session.close()
        conn.close()


class RemoteMujocoSession:
    """Parent-side proxy for a process-isolated MuJoCo session."""

    def __init__(
        self,
        process: mp.process.BaseProcess,
        conn: connection.Connection,
        descriptor: dict[str, Any],
        *,
        rpc_timeout_s: float,
    ) -> None:
        self._process = process
        self._conn = conn
        self._rpc_timeout_s = float(rpc_timeout_s)
        self._lock = threading.Lock()
        self._closed = False
        self.descriptor = descriptor

    def _call(self, command: str, payload: dict[str, Any]) -> Any:
        with self._lock:
            if self._closed:
                raise MujocoSessionError("MuJoCo subprocess session is closed")
            if not self._process.is_alive():
                raise MujocoSessionError(
                    f"MuJoCo subprocess pid={self._process.pid} is not alive; "
                    f"exitcode={self._process.exitcode}"
                )
            try:
                self._conn.send([command, payload])
                if not self._conn.poll(self._rpc_timeout_s):
                    self._abort_locked()
                    raise MujocoSessionError(
                        f"MuJoCo subprocess pid={self._process.pid} did not reply "
                        f"to {command!r} within {self._rpc_timeout_s}s"
                    )
                status, result = self._conn.recv()
            except (EOFError, OSError, ValueError) as exc:
                self._abort_locked()
                raise MujocoSessionError(
                    f"MuJoCo subprocess RPC {command!r} failed: {exc}"
                ) from exc
            if status != "ok":
                raise MujocoSessionError(
                    f"MuJoCo subprocess raised {result['type']} during {command!r}: "
                    f"{result['message']}\n{result['traceback']}"
                )
            return result

    def _abort_locked(self) -> None:
        """Close IPC and terminate the owned child while holding ``_lock``."""
        self._closed = True
        try:
            self._conn.close()
        except OSError:
            pass
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5.0)

    def reset(
        self, *, seed: int | None, options: dict[str, Any] | None
    ) -> tuple[Any, dict[str, Any]]:
        return self._call("reset", {"seed": seed, "options": dict(options or {})})

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self._call("step", {"action": np.asarray(action, dtype=np.float32)})

    def render(self) -> np.ndarray | None:
        return self._call("render", {})

    def close(self) -> None:
        """Request orderly shutdown, then terminate only if the child hangs."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.is_alive():
                try:
                    self._conn.send([_SHUTDOWN, {}])
                    if self._conn.poll(min(self._rpc_timeout_s, 10.0)):
                        self._conn.recv()
                except (EOFError, OSError, ValueError):
                    pass
            self._conn.close()
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)


def spawn_mujoco_session(config: dict[str, Any]) -> RemoteMujocoSession:
    """Spawn and validate one process-isolated MuJoCo session."""
    parent, child = _SPAWN_CONTEXT.Pipe(duplex=True)
    process = _SPAWN_CONTEXT.Process(
        target=_worker_main,
        args=(child, dict(config)),
        name="mujoco-env-slot",
        daemon=True,
    )
    try:
        process.start()
    except BaseException:
        parent.close()
        child.close()
        raise
    child.close()
    timeout = float(config.get("rpc_timeout_s", 120.0))
    if not parent.poll(timeout):
        process.terminate()
        process.join(timeout=5.0)
        parent.close()
        raise MujocoSessionError(
            f"MuJoCo subprocess did not initialize within {timeout}s"
        )
    try:
        status, payload = parent.recv()
    except (EOFError, OSError) as exc:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        parent.close()
        raise MujocoSessionError(
            "MuJoCo subprocess exited before reporting readiness"
        ) from exc
    if status != "ready":
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        parent.close()
        raise MujocoSessionError(
            f"MuJoCo subprocess initialization failed with {payload['type']}: "
            f"{payload['message']}\n{payload['traceback']}"
        )
    return RemoteMujocoSession(process, parent, payload, rpc_timeout_s=timeout)
