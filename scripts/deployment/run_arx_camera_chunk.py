#!/usr/bin/env python3
"""Run one or more receding-horizon Zeva chunks in an ARX MuJoCo scene."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

from robots.arx.contracts import load_model_contract, load_task_manifest
from robots.arx.control import prepare_model_state
from robots.arx.cosmos_edge_client import CosmosEdgeClient
from robots.arx.environment import ArxMujocoEnv


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5581)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--inference-seed", type=int, help="override backend seed for every chunk")
    parser.add_argument("--offline-test", action="store_true", help="use a small deterministic right-arm test chunk")
    parser.add_argument("--overview-camera", help="optional diagnostic MuJoCo camera to render separately")
    parser.add_argument("--chunks", type=int, default=1, help="number of receding-horizon Zeva chunks")
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    if args.inference_seed is not None and not 0 <= args.inference_seed < 2**31:
        parser.error("--inference-seed must be in [0, 2**31)")
    args.output.mkdir(parents=True, exist_ok=False)
    fault_log = (args.output / "native_fault.log").open("w", buffering=1)
    faulthandler.enable(file=fault_log, all_threads=True)
    trace_path = args.output / "diagnostic_trace.jsonl"

    def trace(event: str, payload: dict | None = None) -> None:
        record = {
            "wall_time": time.time(),
            "monotonic": time.monotonic(),
            "event": event,
            **(payload or {}),
        }
        line = json.dumps(record, default=str, separators=(",", ":"))
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"[arx-trace] {line}", flush=True)

    trace("client_start", {"arguments": vars(args)})
    trace("contracts_load_begin")
    task, contract = load_task_manifest(args.task), load_model_contract(args.contract)
    trace("contracts_load_end", {"task": task.name, "execution_steps": task.execution_steps})
    trace("environment_create_begin")
    env = ArxMujocoEnv(
        prepared_scene_bundle=str(args.scene), mapping_path=str(args.mapping),
        task_manifest=str(args.task), camera_names={name: name for name in ("front_rgb", "left_rgb", "right_rgb")},
        trace_hook=trace,
    )
    trace("environment_create_end")
    observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    vla_targets, step_chunks, step_indices = [], [], []
    sim_states = []

    def capture_state() -> None:
        sim_states.append({"time": float(env.data.time), "qpos": env.data.qpos.copy(),
                           "qvel": env.data.qvel.copy(), "ctrl": env.data.ctrl.copy()})
    overview_frames: list[np.ndarray] = []
    overview_renderer = mujoco.Renderer(env.model, height=240, width=320) if args.overview_camera else None
    def capture_overview() -> None:
        if overview_renderer is not None:
            overview_renderer.update_scene(env.data, camera=args.overview_camera)
            overview_frames.append(np.array(overview_renderer.render(), copy=True, order="C"))
    try:
        trace("reset_call_begin")
        observation, reset_info = env.reset(seed=args.seed)
        trace("reset_call_end")
        observations.append({key: value.copy() for key, value in observation.items()})
        capture_state()
        capture_overview()
        terminated = truncated = False
        info = reset_info
        all_raw_actions: list[np.ndarray] = []
        roundtrip_sec, prediction_metadata = 0.0, {}
        for chunk_index in range(args.chunks):
            trace("chunk_begin", {"chunk_index": chunk_index, "step": len(actions)})
            if args.offline_test:
                raw_actions = np.repeat(observation["state"][None], contract.action_horizon, axis=0)
            # Visible, smooth right-arm and gripper motion, bounded by the task's
            # delta and per-step safety filters.
                phase = np.linspace(0.0, 1.0, contract.action_horizon, dtype=np.float32)
                for channel, delta in ((7, 0.45), (8, -0.40), (9, 0.35), (10, 0.30), (11, -0.25), (12, 0.30), (13, 0.60)):
                    raw_actions[:, channel] += phase * delta
                prediction_metadata = {"source": "deterministic_safe_test_chunk"}
            else:
                trace("inference_connect_begin", {"host": args.host, "port": args.port})
                with CosmosEdgeClient(args.host, args.port, contract, timeout_sec=300.0) as client:
                    trace("inference_request_begin")
                    prediction = client.predict(
                        {name: observation[name] for name in ("front_rgb", "left_rgb", "right_rgb")},
                        prepare_model_state(observation["state"], task), task.instruction,
                        seed=args.inference_seed,
                    )
                    trace("inference_request_end", {"roundtrip_sec": prediction.roundtrip_sec})
                raw_actions = prediction.actions
                roundtrip_sec += prediction.roundtrip_sec
                prediction_metadata = prediction.metadata
            all_raw_actions.append(raw_actions.copy())
            trace(
                "actions_ready",
                {
                    "chunk_index": chunk_index, "shape": list(raw_actions.shape),
                    "finite": bool(np.isfinite(raw_actions).all()),
                    "minimum": float(np.min(raw_actions)),
                    "maximum": float(np.max(raw_actions)),
                },
            )
            for action_index, raw_action in enumerate(raw_actions[: task.execution_steps]):
                trace("env_step_call_begin", {"action_index": action_index})
                observation, _, terminated, truncated, info = env.step(raw_action)
                trace(
                    "env_step_call_end",
                    {
                        "action_index": action_index,
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                )
                observations.append({key: value.copy() for key, value in observation.items()})
                capture_state()
                capture_overview()
                actions.append(np.asarray(info["processed_action"]).copy())
                vla_targets.append(np.asarray(raw_action).copy())
                step_chunks.append(chunk_index)
                step_indices.append(action_index)
                if terminated or truncated:
                    break
            trace("chunk_end", {"chunk_index": chunk_index, "step": len(actions), "terminal": bool(terminated or truncated)})
            if terminated or truncated:
                break
        raw_actions = np.concatenate(all_raw_actions, axis=0)
        np.save(args.output / "raw_actions.npy", raw_actions, allow_pickle=False)
        names = ("front_rgb", "left_rgb", "right_rgb")
        for name in names:
            imageio.imwrite(args.output / f"{name}_frame0.png", observations[0][name])
        ffmpeg_options = {"codec": "libx264", "pixelformat": "yuv420p", "macro_block_size": 1}
        for name in names:
            imageio.mimwrite(args.output / f"{name}.mp4", [item[name] for item in observations], fps=15, **ffmpeg_options)
        mosaics = [np.concatenate([item[name] for name in names], axis=1) for item in observations]
        imageio.imwrite(args.output / "three_view_frame0.png", mosaics[0])
        imageio.mimwrite(args.output / "three_view.mp4", mosaics, fps=15, **ffmpeg_options)
        if overview_frames:
            imageio.imwrite(args.output / f"{args.overview_camera}_frame0.png", overview_frames[0])
            imageio.mimwrite(args.output / f"{args.overview_camera}.mp4", overview_frames, fps=15, **ffmpeg_options)
        np.save(args.output / "executed_actions.npy", np.asarray(actions), allow_pickle=False)
        states = np.asarray([item["state"] for item in observations])
        np.savez_compressed(
            args.output / "trajectory.npz",
            vla_action_targets=np.asarray(vla_targets),
            processed_action_targets=np.asarray(actions),
            actual_tracked_actions=states[1:],
            robot_states=states,
            chunk_index=np.asarray(step_chunks), action_index=np.asarray(step_indices),
            **{key: np.asarray([item[key] for item in sim_states])
               for key in ("time", "qpos", "qvel", "ctrl")},
        )
        audit = {
            "schema_version": "zetta_arx_camera_chunk_audit_v1", "seed": args.seed,
            "inference_seed": args.inference_seed,
            "instruction": task.instruction, "requested_shape": list(raw_actions.shape),
            "executed_steps": len(actions), "roundtrip_sec": roundtrip_sec,
            "executed_chunks": len(all_raw_actions),
            "success": bool(info["evaluation"]["success"]),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "terminal_reason": info.get("terminal_reason") or "chunk_limit",
            "final_evaluation": info["evaluation"],
            "prediction_metadata": prediction_metadata, "reset_info": reset_info,
            "initial_state_sha256": _digest(observations[0]["state"]),
            "initial_image_sha256": {name: _digest(observations[0][name]) for name in names},
            "final_state_sha256": _digest(observations[-1]["state"]),
        }
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
        trace("client_complete", {"executed_steps": len(actions)})
        print(json.dumps(audit, indent=2, default=str))
    except BaseException as exc:
        trace("python_exception", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        trace("cleanup_begin")
        if overview_renderer is not None:
            overview_renderer.close()
        env.close()
        trace("cleanup_end")
        faulthandler.disable()
        fault_log.close()


if __name__ == "__main__":
    main()
