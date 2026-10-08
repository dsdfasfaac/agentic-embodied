#!/usr/bin/env python3
"""Replay a retained, synchronized ARX RGB-D snapshot without accessing the robot.

The output is offline feasibility evidence, never a live motion/reentry token.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.manipulation.grasp_proposals import (
    LocalGraspService,
    geometry_proposal,
    rigid_pose,
    transfer_grasp_pose,
    parallel_jaw_candidates,
)
from robots.arx.deployment.picktube_grasp_observer import PickTubeGraspObserver
from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
from robots.arx.gateway.grasp_contracts import GraspRecoveryConfig
from robots.arx.gateway.grasp_recovery import GraspRecovery
from robots.arx.gateway.real_config import load_real_hardware_config


def replay_target_history(provider, journal_path, observation, snapshot_sha256):
    """Establish cross-view identity from unique past samples, without future data."""
    journal_path = journal_path.resolve()
    evidence = {
        "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        "past_sample_count": 0,
        "unavailable_sample_count": 0,
    }
    with sqlite3.connect(journal_path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT sequence, payload FROM records WHERE kind=? ORDER BY sequence",
            ("grasp_sensor_evidence",),
        ).fetchall()
    items = [(sequence, json.loads(payload)) for sequence, payload in rows]
    matches = [
        (sequence, item)
        for sequence, item in items
        if item["observation_id"] == observation["observation_id"]
    ]
    if len(matches) != 1:
        raise ValueError("history must contain exactly one selected observation")
    final_sequence, final = matches[0]
    if final["sha256"] != snapshot_sha256 or final["observation"] != observation:
        raise ValueError("history selected observation or sensor SHA differs")
    stamp = observation["hardware"]["state_monotonic_ns"]
    previous_stamp = None
    for sequence, item in items:
        if sequence >= final_sequence:
            break
        current_stamp = item["observation"]["hardware"]["state_monotonic_ns"]
        if current_stamp >= stamp or (
            previous_stamp is not None and current_stamp <= previous_stamp
        ):
            raise ValueError("history contains non-monotonic or future state samples")
        previous_stamp = current_stamp
        path = journal_path.parent / "grasp-sensors" / Path(item["path"]).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("history sensor SHA differs")
        with np.load(path, allow_pickle=False) as snapshot:
            images = {key: snapshot[key].copy() for key in snapshot.files}
        try:
            provider.target_sample(images, item["observation"]["hardware"])
        except ValueError as exc:
            if not str(exc).startswith(("pink tube", "yellow test-tube rack")):
                raise
            evidence["unavailable_sample_count"] += 1
        evidence["past_sample_count"] += 1
    evidence["selected_sequence"] = final_sequence
    return evidence


def probe(args):
    # Initialise numeric dependencies before loading the camera SDK. Fail here
    # with a dependency error instead of halfway through candidate planning.
    import scipy.optimize

    settings = GraspRecoveryConfig.model_validate_json(args.grasp_config.read_text())
    hardware = load_real_hardware_config(args.hardware_config, args.hardware_sha256)
    provider = PickTubeRgbdProvider()
    provider.validate_hardware(hardware)
    value = json.loads(args.observation.read_text())
    observation = value.get("observation", value)
    if "hardware" not in observation:
        raise ValueError("provide the retained grasp_sensor_evidence observation")
    raw = args.snapshot.read_bytes()
    if value.get("sha256") and hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("retained grasp sensor SHA differs")
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        images = {key: snapshot[key].copy() for key in snapshot.files}
    history_evidence = None
    if getattr(args, "history_journal", None):
        history_evidence = replay_target_history(
            provider, args.history_journal, observation, hashlib.sha256(raw).hexdigest()
        )
    observer = PickTubeGraspObserver(provider, cloud_mode=settings.target_cloud_mode)
    cloud = observer.cloud(observation, images)
    state = np.asarray(observation["hardware"]["measured_state"], dtype=float)
    _, orientation, _ = provider.tool_fk.fk(state[7:13])
    if args.engine == "tube_geometry":
        candidates = geometry_proposal(
            cloud,
            orientation,
            surface_offset_m=settings.target_surface_offset_m,
            orientation_search_rad=settings.geometry_orientation_search_rad,
        )
        service_evidence = None
    else:
        endpoint = getattr(settings, args.engine + "_endpoint")
        service = LocalGraspService(
            args.engine,
            endpoint,
            expected_gripper=settings.learned_gripper_id,
            expected_model_sha256=settings.learned_model_sha256,
            sampling_options={
                "num_model_samples": settings.graspgen_samples,
                "horizontal_closing_max": settings.graspgen_horizontal_closing_max,
            },
        )
        if (
            args.engine == "graspgen"
            and settings.graspgen_approach_alignment_min is not None
        ):
            start_p, _, _ = provider.tool_fk.fk(state[7:13])
            target_base = (cloud.camera_to_base @ np.r_[cloud.target_camera_m, 1.0])[:3]
            direction = target_base - start_p
            service.sampling_options.update(
                preferred_approach_camera=(
                    cloud.camera_to_base[:3, :3].T
                    @ (direction / np.linalg.norm(direction))
                ).tolist(),
                approach_alignment_min=settings.graspgen_approach_alignment_min,
            )
        candidates = service.propose(
            cloud, max_candidates=getattr(args, "max_candidates", 8)
        )
        service_evidence = service.last_evidence
        if args.engine == "graspgen" and settings.learned_parallel_jaw_half_turn:
            candidates = parallel_jaw_candidates(
                candidates, max_candidates=getattr(args, "max_candidates", 8)
            )
    reviewer = GraspRecovery(
        settings,
        observer,
        closed_policy=hardware.right_gripper_closed_policy,
        open_policy=hardware.right_gripper_open_policy,
        control_hz=hardware.timing.control_hz,
        joint_bounds=tuple(
            zip(hardware.right.joint_min_rad, hardware.right.joint_max_rad)
        ),
    )
    rows = []
    for index, candidate in enumerate(candidates):
        candidate["transform_base"] = (
            cloud.camera_to_base @ rigid_pose(candidate["transform_camera"])
        ).tolist()
        row = {"candidate_index": index, "candidate": candidate}
        try:
            if args.engine == "tube_geometry":
                goal = reviewer._goal(candidate, args.engine, "pregrasp", state)
            else:
                # Offline geometry inspection must precede physical transfer
                # verification. It cannot grant the live gateway's capability.
                goal = transfer_grasp_pose(
                    candidate["transform_camera"],
                    cloud.camera_to_base,
                    settings.learned_grasp_to_tcp,
                )
                target_base = (
                    cloud.camera_to_base @ np.r_[cloud.target_camera_m, 1.0]
                )[:3]
                row["mapped_engage_tcp_base"] = goal.tolist()
                row["mapped_target_distance_m"] = float(
                    np.linalg.norm(goal[:3, 3] - target_base)
                )
                if (
                    row["mapped_target_distance_m"]
                    > settings.learned_target_distance_max_m
                ):
                    raise ValueError(
                        "learned mapped TCP is outside selected target distance"
                    )
                goal = goal.copy()
                goal[:3, 3] -= settings.pregrasp_distance_m * goal[:3, 0]
            plan, clearance = reviewer._plan(
                state.copy(), state, goal, cloud, "pregrasp", args.max_steps
            )
            row.update(
                feasible=True,
                planned_steps=len(plan),
                minimum_tcp_clearance_m=clearance,
                goal_tcp_base=goal.tolist(),
                final_joint_target_rad=plan[-1, 7:13].tolist(),
            )
        except ValueError as exc:
            row.update(feasible=False, reason=str(exc))
        rows.append(row)
    return {
        "schema_version": "arx.grasp.offline-audit.v1",
        "offline": True,
        "robot_commands_sent": False,
        "live_motion_eligible": False,
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "hardware_config_sha256": args.hardware_sha256,
        "grasp_config_sha256": hashlib.sha256(
            args.grasp_config.read_bytes()
        ).hexdigest(),
        "numeric_versions": {"scipy": scipy.__version__, "numpy": np.__version__},
        "target_evidence": cloud.evidence,
        "history_evidence": history_evidence,
        "engine": args.engine,
        "transfer_physically_verified": settings.learned_gripper_transfer_verified,
        "model_evidence": service_evidence,
        "candidates": rows,
        "feasible_candidate_count": sum(row["feasible"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "snapshot",
        "observation",
        "grasp-config",
        "hardware-config",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hardware-sha256", required=True)
    parser.add_argument(
        "--history-journal",
        type=Path,
        help="Read-only retained past samples for cross-view target identity",
    )
    parser.add_argument(
        "--engine",
        choices=("tube_geometry", "contact_graspnet", "graspgen"),
        default="tube_geometry",
    )
    parser.add_argument(
        "--max-steps", type=int, choices=range(20, 361), default=240, metavar="20..360"
    )
    parser.add_argument(
        "--max-candidates", type=int, choices=range(1, 33), default=8, metavar="1..32"
    )
    args = parser.parse_args()
    failed = False
    try:
        result = probe(args)
    except (ValueError, SystemError, ImportError) as exc:
        failed = True
        result = {
            "schema_version": "arx.grasp.offline-audit.v1",
            "offline": True,
            "robot_commands_sent": False,
            "live_motion_eligible": False,
            "feasible_candidate_count": 0,
            "status": "failed",
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "robot_commands_sent": False,
                "feasible_candidate_count": result["feasible_candidate_count"],
            }
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
