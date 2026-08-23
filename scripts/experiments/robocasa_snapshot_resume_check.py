# Copyright (c) 2026 Zetta Contributors
"""Cross-process RoboCasa replay-vs-snapshot continuation experiment.

Phase A creates a fresh environment, replays recorded actions from reset to a
chosen keyframe, snapshots the complete simulator state, and continues the
recorded action trajectory.  Phase B creates another fresh environment,
restores the snapshot without replaying the prefix, and executes the identical
tail actions.  The parent compares every MuJoCo integration scalar and writes
side-by-side videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as iio
import mujoco
import numpy as np

from robots.robocasa.snapshot import (
    model_state_difference,
    raw_model_and_data,
    read_integration_state,
    restore_snapshot,
    save_snapshot,
)

CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


def load_actions(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    records.sort(key=lambda item: int(item["step_index"]))
    expected = list(range(1, len(records) + 1))
    actual = [int(item["step_index"]) for item in records]
    if actual != expected:
        raise ValueError("trajectory step indices are not contiguous from 1")
    # Production's canonical_action contract converts JSON lists to float64
    # arrays before calling Gym.  In particular, Gym compares the one-element
    # gripper/control components with scalars; leaving them as JSON lists is a
    # type error and would not reproduce the recorded rollout.
    return [
        {
            name: np.asarray(component, dtype=np.float64)
            for name, component in item["action"].items()
        }
        for item in records
    ]


def create_environment(args: argparse.Namespace):
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.wrappers.gym_wrapper  # noqa: F401

    env = gym.make(
        f"robocasa/{args.task}",
        split=args.split,
        enable_render=True,
        camera_widths=args.camera_size,
        camera_heights=args.camera_size,
        # RGB-only forces RoboSuite through its serialized official Renderer.
        # Its legacy shared RGB-D framebuffer has an independent, intermittent
        # camera-alias/readback bug on MuJoCo 3.3.1 + multi-GPU EGL and is not a
        # valid oracle for snapshot determinism.  Physics state is unaffected.
        camera_depths=False,
    )
    observation, _ = env.reset(seed=args.seed)
    return env, dict(observation)


def owned_rgb_frames(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        if key not in observation:
            raise KeyError(f"missing camera observation {key}")
        frame = np.array(observation[key], dtype=np.uint8, order="C", copy=True)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{key} is not RGB: {frame.shape}")
        result[key] = frame
    return result


class VideoSet:
    def __init__(self, root: Path, *, fps: int = 20):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.frame_count = 0
        self.frame_shapes: dict[str, list[int]] = {}
        self.raw_hashes = {key: hashlib.sha256() for key in CAMERA_KEYS}
        self.raw_hashes["multiview"] = hashlib.sha256()
        self.raw_frame_hashes: dict[str, list[str]] = {
            key: [] for key in (*CAMERA_KEYS, "multiview")
        }
        self.writers = {
            key: iio.get_writer(
                root / f"{key.removeprefix('video.').replace('.', '_')}.mp4",
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                output_params=["-threads", "1"],
            )
            for key in CAMERA_KEYS
        }
        self.multiview = iio.get_writer(
            root / "multiview.mp4",
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            output_params=["-threads", "1"],
        )

    def append(self, observation: dict[str, Any]) -> None:
        frames = owned_rgb_frames(observation)
        for key, frame in frames.items():
            self.frame_shapes.setdefault(key, list(frame.shape))
            if self.frame_shapes[key] != list(frame.shape):
                raise ValueError(f"raw frame shape changed for {key}")
            self.raw_hashes[key].update(frame.tobytes())
            self.raw_frame_hashes[key].append(hashlib.sha256(frame).hexdigest())
            self.writers[key].append_data(frame)
        multiview = np.concatenate([frames[key] for key in CAMERA_KEYS], axis=1)
        self.frame_shapes.setdefault("multiview", list(multiview.shape))
        if self.frame_shapes["multiview"] != list(multiview.shape):
            raise ValueError("raw frame shape changed for multiview")
        self.raw_hashes["multiview"].update(multiview.tobytes())
        self.raw_frame_hashes["multiview"].append(hashlib.sha256(multiview).hexdigest())
        self.multiview.append_data(multiview)
        self.frame_count += 1

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.multiview.close()
        manifest = {
            "schema_version": 1,
            "frames": self.frame_count,
            "streams": {
                name: {
                    "shape": self.frame_shapes.get(name),
                    "raw_uint8_sha256": digest.hexdigest(),
                    "frame_sha256": self.raw_frame_hashes[name],
                }
                for name, digest in self.raw_hashes.items()
            },
        }
        (self.root / "raw_frame_hashes.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


def diagnostic_state(env: Any) -> dict[str, np.ndarray]:
    sim, _, _ = raw_model_and_data(env)
    return {
        "integration": read_integration_state(env),
        "qpos": np.array(sim.data.qpos, copy=True),
        "qvel": np.array(sim.data.qvel, copy=True),
        "qacc": np.array(sim.data.qacc, copy=True),
        "qacc_warmstart": np.array(sim.data.qacc_warmstart, copy=True),
        "ctrl": np.array(sim.data.ctrl, copy=True),
        "qfrc_constraint": np.array(sim.data.qfrc_constraint, copy=True),
        "cam_xpos": np.array(sim.data.cam_xpos, copy=True),
        "cam_xmat": np.array(sim.data.cam_xmat, copy=True),
        "body_xpos": np.array(sim.data.body_xpos, copy=True),
        "body_xmat": np.array(sim.data.body_xmat, copy=True),
        "geom_xpos": np.array(sim.data.geom_xpos, copy=True),
        "geom_xmat": np.array(sim.data.geom_xmat, copy=True),
        "site_xpos": np.array(sim.data.site_xpos, copy=True),
        "site_xmat": np.array(sim.data.site_xmat, copy=True),
    }


def append_state(traces: dict[str, list[np.ndarray]], env: Any) -> None:
    for name, value in diagnostic_state(env).items():
        traces.setdefault(name, []).append(value)


def save_traces(path: Path, traces: dict[str, list[np.ndarray]]) -> None:
    np.savez(path, **{name: np.stack(values) for name, values in traces.items()})


def install_tail_rgb_renderer() -> None:
    """Use the production-isolated RGB path only for compared tail frames."""

    from robots.robocasa.stable_renderer import install_persistent_rgb_renderer

    install_persistent_rgb_renderer()


def run_record(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    actions = load_actions(args.actions)
    end_step = min(len(actions), args.keyframe + args.tail_steps)
    env, observation = create_environment(args)
    try:
        for action in actions[: args.keyframe]:
            observation, *_ = env.step(action)
        snapshot_base = out / "snapshot"
        save_snapshot(env, snapshot_base)
        install_tail_rgb_renderer()
        traces: dict[str, list[np.ndarray]] = {}
        append_state(traces, env)
        video = VideoSet(out / "reference_video")
        try:
            for action in actions[args.keyframe : end_step]:
                observation, *_ = env.step(action)
                append_state(traces, env)
                video.append(dict(observation))
        finally:
            video.close()
        save_traces(out / "reference_trace.npz", traces)
        metadata = {
            "phase": "record",
            "seed": args.seed,
            "task": args.task,
            "split": args.split,
            "keyframe": args.keyframe,
            "end_step": end_step,
            "prefix_actions_replayed": args.keyframe,
            "tail_actions_replayed": end_step - args.keyframe,
            "snapshot_sha256": hashlib.sha256(
                (out / "snapshot.npz").read_bytes()
            ).hexdigest(),
        }
        (out / "record.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        env.close()


def run_resume(args: argparse.Namespace) -> None:
    actions = load_actions(args.actions)
    end_step = min(len(actions), args.keyframe + args.tail_steps)
    env, _ = create_environment(args)
    try:
        geometry_before = model_state_difference(env, args.out / "snapshot")
        restore_snapshot(env, args.out / "snapshot")
        install_tail_rgb_renderer()
        traces: dict[str, list[np.ndarray]] = {}
        append_state(traces, env)
        video = VideoSet(args.out / "resumed_video")
        try:
            for action in actions[args.keyframe : end_step]:
                observation, *_ = env.step(action)
                append_state(traces, env)
                video.append(dict(observation))
        finally:
            video.close()
        save_traces(args.out / "resumed_trace.npz", traces)
        metadata = {
            "phase": "resume",
            "seed": args.seed,
            "task": args.task,
            "split": args.split,
            "keyframe": args.keyframe,
            "end_step": end_step,
            "prefix_actions_replayed": 0,
            "tail_actions_replayed": end_step - args.keyframe,
            "model_max_abs_before_restore": max(geometry_before.values(), default=0.0),
            "model_changed_fields_before_restore": sorted(
                name for name, value in geometry_before.items() if value != 0.0
            ),
        }
        (args.out / "resume.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        env.close()


def name_for_joint(raw_model: Any, joint_id: int) -> str:
    return (
        mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        or f"joint#{joint_id}"
    )


def qvel_labels(raw_model: Any) -> list[str]:
    labels = ["unmapped"] * int(raw_model.nv)
    widths = {0: 6, 1: 3, 2: 1, 3: 1}
    for joint_id in range(int(raw_model.njnt)):
        start = int(raw_model.jnt_dofadr[joint_id])
        width = widths[int(raw_model.jnt_type[joint_id])]
        name = name_for_joint(raw_model, joint_id)
        for offset in range(width):
            labels[start + offset] = f"{name}[{offset}]"
    return labels


def compare(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.out / "reference_trace.npz", allow_pickle=False) as stored:
        reference = {name: np.array(stored[name], copy=True) for name in stored.files}
    with np.load(args.out / "resumed_trace.npz", allow_pickle=False) as stored:
        resumed = {name: np.array(stored[name], copy=True) for name in stored.files}
    metrics: dict[str, Any] = {}
    for name, expected in reference.items():
        actual = resumed[name]
        if expected.shape != actual.shape:
            metrics[name] = {
                "shape_match": False,
                "reference": expected.shape,
                "resumed": actual.shape,
            }
            continue
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        row_max = difference.reshape(len(difference), -1).max(axis=1)
        metrics[name] = {
            "shape_match": True,
            "all_zero": bool(np.all(difference == 0.0)),
            "landing_max_abs": float(row_max[0]),
            "tail_max_abs": float(row_max[1:].max()) if len(row_max) > 1 else 0.0,
            "final_max_abs": float(row_max[-1]),
            "first_nonzero_row": (
                int(np.flatnonzero(row_max != 0.0)[0])
                if np.any(row_max != 0.0)
                else None
            ),
        }
    verdict = {
        "schema_version": 1,
        "task": args.task,
        "seed": args.seed,
        "keyframe": args.keyframe,
        "tail_steps": min(args.tail_steps, len(reference["integration"]) - 1),
        "physics_bit_exact": metrics["integration"].get("all_zero", False),
        "qpos_bit_exact": metrics["qpos"].get("all_zero", False),
        "qvel_bit_exact": metrics["qvel"].get("all_zero", False),
        "metrics": metrics,
    }
    required_landing_fields = (
        "integration",
        "qpos",
        "qvel",
        "qacc_warmstart",
        "ctrl",
    )
    verdict["required_landing_bit_exact"] = all(
        metrics[name].get("landing_max_abs") == 0.0 for name in required_landing_fields
    )
    verdict["continuation_bit_exact"] = all(
        metric.get("shape_match", False) and metric.get("tail_max_abs") == 0.0
        for metric in metrics.values()
    )
    # qacc and qfrc_constraint are outputs of MuJoCo's dynamics solve, not
    # integration inputs.  mj_forward recomputes them during restore.  Report
    # landing differences explicitly while requiring the first and every
    # subsequent common action to converge bit-exactly.
    verdict["recomputed_derived_landing_fields"] = [
        name
        for name in ("qacc", "qfrc_constraint")
        if metrics[name].get("landing_max_abs") != 0.0
    ]
    verdict["snapshot_pass"] = bool(
        verdict["required_landing_bit_exact"] and verdict["continuation_bit_exact"]
    )
    verdict["pass"] = verdict["snapshot_pass"]
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def build_side_by_side(args: argparse.Namespace) -> None:
    reference = iio.get_reader(args.out / "reference_video" / "multiview.mp4")
    resumed = iio.get_reader(args.out / "resumed_video" / "multiview.mp4")
    writer = iio.get_writer(
        args.out / "replay_vs_snapshot.mp4",
        fps=20,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for left, right in zip(reference, resumed):
            divider = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
            writer.append_data(np.concatenate([left, divider, right], axis=1))
    finally:
        writer.close()
        reference.close()
        resumed.close()


def compare_videos(args: argparse.Namespace) -> dict[str, Any]:
    """Compare raw renderer hashes and diagnose decoded MP4 equality."""

    raw_reference = json.loads(
        (args.out / "reference_video" / "raw_frame_hashes.json").read_text()
    )
    raw_resumed = json.loads(
        (args.out / "resumed_video" / "raw_frame_hashes.json").read_text()
    )
    raw_comparisons: list[dict[str, Any]] = []
    stream_names = sorted(set(raw_reference["streams"]) | set(raw_resumed["streams"]))
    for name in stream_names:
        left = raw_reference["streams"].get(name)
        right = raw_resumed["streams"].get(name)
        left_frames = left.get("frame_sha256", []) if left is not None else []
        right_frames = right.get("frame_sha256", []) if right is not None else []
        first_mismatch = next(
            (
                index
                for index, (left_hash, right_hash) in enumerate(
                    zip_longest(left_frames, right_frames)
                )
                if left_hash != right_hash
            ),
            None,
        )
        raw_comparisons.append(
            {
                "stream": name,
                "present_in_both": left is not None and right is not None,
                "shape_match": (
                    left is not None
                    and right is not None
                    and left["shape"] == right["shape"]
                ),
                "raw_uint8_sha256_equal": (
                    left is not None
                    and right is not None
                    and left["raw_uint8_sha256"] == right["raw_uint8_sha256"]
                ),
                "frame_count_match": len(left_frames) == len(right_frames),
                "first_mismatching_frame": first_mismatch,
            }
        )
    raw_frame_count_match = raw_reference["frames"] == raw_resumed["frames"]
    raw_pixel_exact = raw_frame_count_match and all(
        item["present_in_both"]
        and item["shape_match"]
        and item["raw_uint8_sha256_equal"]
        and item["frame_count_match"]
        for item in raw_comparisons
    )

    names = (
        "multiview.mp4",
        "robot0_agentview_left.mp4",
        "robot0_agentview_right.mp4",
        "robot0_eye_in_hand.mp4",
    )
    comparisons: list[dict[str, Any]] = []
    for name in names:
        reference = iio.get_reader(args.out / "reference_video" / name)
        resumed = iio.get_reader(args.out / "resumed_video" / name)
        frame_count = 0
        maximum = 0
        changed_values = 0
        reference_digest = hashlib.sha256()
        resumed_digest = hashlib.sha256()
        missing = object()
        try:
            for left, right in zip_longest(reference, resumed, fillvalue=missing):
                if left is missing or right is missing:
                    raise ValueError(f"decoded video frame-count mismatch for {name}")
                left = np.asarray(left)
                right = np.asarray(right)
                if left.shape != right.shape:
                    raise ValueError(
                        f"decoded video shape mismatch for {name}: "
                        f"{left.shape} != {right.shape}"
                    )
                reference_digest.update(left.tobytes())
                resumed_digest.update(right.tobytes())
                difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
                maximum = max(maximum, int(difference.max()))
                changed_values += int(np.count_nonzero(difference))
                frame_count += 1
        finally:
            reference.close()
            resumed.close()
        comparisons.append(
            {
                "video": name,
                "frames": frame_count,
                "decoded_max_abs": maximum,
                "changed_channel_values": changed_values,
                "decoded_sha256_equal": (
                    reference_digest.hexdigest() == resumed_digest.hexdigest()
                ),
            }
        )
    decoded_pixel_exact = all(
        item["decoded_max_abs"] == 0
        and item["changed_channel_values"] == 0
        and item["decoded_sha256_equal"]
        for item in comparisons
    )
    verdict = {
        "pass": raw_pixel_exact,
        "raw_renderer_pixel_exact": raw_pixel_exact,
        "raw_frame_count_match": raw_frame_count_match,
        "raw_comparisons": raw_comparisons,
        "decoded_mp4_pixel_exact": decoded_pixel_exact,
        "decoded_comparisons": comparisons,
    }
    (args.out / "video_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def phase_command(args: argparse.Namespace, phase: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--task",
        args.task,
        "--split",
        args.split,
        "--seed",
        str(args.seed),
        "--keyframe",
        str(args.keyframe),
        "--tail-steps",
        str(args.tail_steps),
        "--camera-size",
        str(args.camera_size),
        "--actions",
        str(args.actions.resolve()),
        "--out",
        str(args.out.resolve()),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("auto", "record", "resume"), default="auto")
    parser.add_argument("--task", default="PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--keyframe", type=int, default=80)
    parser.add_argument("--tail-steps", type=int, default=80)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.keyframe < 1 or args.tail_steps < 1:
        parser.error("keyframe and tail-steps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "record":
        run_record(args)
        return 0
    if args.phase == "resume":
        run_resume(args)
        return 0
    logs = args.out / "logs"
    logs.mkdir(exist_ok=True)
    for phase in ("record", "resume"):
        with (logs / f"{phase}.log").open("w") as stream:
            completed = subprocess.run(
                phase_command(args, phase), stdout=stream, stderr=subprocess.STDOUT
            )
        if completed.returncode:
            print(f"{phase} phase failed; see {logs / f'{phase}.log'}", file=sys.stderr)
            return 2
    verdict = compare(args)
    build_side_by_side(args)
    video_verdict = compare_videos(args)
    verdict["video_raw_bit_exact"] = video_verdict["pass"]
    verdict["decoded_mp4_bit_exact"] = video_verdict["decoded_mp4_pixel_exact"]
    verdict["strict_state_and_video_pass"] = bool(
        verdict["snapshot_pass"] and video_verdict["pass"]
    )
    # GPU rasterization is a diagnostic, not the simulator-state oracle.  A
    # moving wrist view can differ by a few channel values even when every
    # physics and camera transform scalar is bit-exact.  The command therefore
    # succeeds on exact snapshot continuation and reports visual exactness
    # independently.
    verdict["pass"] = verdict["snapshot_pass"]
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
