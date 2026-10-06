#!/usr/bin/env python3
"""Replay a retained, synchronized ARX RGB-D snapshot without accessing the robot.

The output is offline feasibility evidence, never a live motion/reentry token.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.manipulation.grasp_proposals import LocalGraspService, geometry_proposal, rigid_pose
from robots.arx.deployment.picktube_grasp_observer import PickTubeGraspObserver
from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
from robots.arx.gateway.grasp_contracts import GraspRecoveryConfig
from robots.arx.gateway.grasp_recovery import GraspRecovery
from robots.arx.gateway.real_config import load_real_hardware_config


def probe(args):
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
    observer = PickTubeGraspObserver(provider)
    cloud = observer.cloud(observation, images)
    state = np.asarray(observation["hardware"]["measured_state"], dtype=float)
    _, orientation, _ = provider.tool_fk.fk(state[7:13])
    if args.engine == "tube_geometry":
        candidates = geometry_proposal(cloud, orientation, surface_offset_m=settings.target_surface_offset_m)
        service_evidence = None
    else:
        endpoint = getattr(settings, args.engine + "_endpoint")
        service = LocalGraspService(args.engine, endpoint,
            expected_gripper=settings.learned_gripper_id, expected_model_sha256=settings.learned_model_sha256)
        candidates = service.propose(cloud, max_candidates=8)
        service_evidence = service.last_evidence
    reviewer = GraspRecovery(settings, observer,
        closed_policy=hardware.right_gripper_closed_policy,
        open_policy=hardware.right_gripper_open_policy, control_hz=hardware.timing.control_hz)
    rows = []
    for candidate in candidates:
        candidate["transform_base"] = (cloud.camera_to_base @ rigid_pose(candidate["transform_camera"])).tolist()
        row = {"candidate": candidate}
        try:
            goal = reviewer._goal(candidate, args.engine, "pregrasp", state)
            plan, clearance = reviewer._plan(state.copy(), state, goal, cloud, "pregrasp", args.max_steps)
            row.update(feasible=True, planned_steps=len(plan), minimum_tcp_clearance_m=clearance,
                       goal_tcp_base=goal.tolist(), final_joint_target_rad=plan[-1, 7:13].tolist())
        except ValueError as exc:
            row.update(feasible=False, reason=str(exc))
        rows.append(row)
    return {"schema_version": "arx.grasp.offline-audit.v1", "offline": True,
            "robot_commands_sent": False, "live_motion_eligible": False,
            "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
            "target_evidence": cloud.evidence, "engine": args.engine,
            "model_evidence": service_evidence, "candidates": rows,
            "feasible_candidate_count": sum(row["feasible"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "observation", "grasp-config", "hardware-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hardware-sha256", required=True)
    parser.add_argument("--engine", choices=("tube_geometry", "contact_graspnet", "graspgen"), default="tube_geometry")
    parser.add_argument("--max-steps", type=int, choices=range(20, 241), default=240, metavar="20..240")
    args = parser.parse_args()
    result = probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "robot_commands_sent": False,
                      "feasible_candidate_count": result["feasible_candidate_count"]}))


if __name__ == "__main__":
    main()
