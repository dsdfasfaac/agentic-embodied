#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Execute the requested live gateway commissioning trace, recording every turn."""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robots.arx.contracts import load_model_contract, load_task_manifest
from robots.arx.gateway.backend import DirectBackend
from robots.arx.gateway.contracts import RuntimeLimits, ToolRequest
from robots.arx.gateway.journal import Journal
from robots.arx.gateway.motion import Calibration, CommandKinematics
from robots.arx.gateway.recording import export_public_recording
from robots.arx.gateway.session_core import ArxSessionCore, BaselineMonitor
from robots.arx.gateway.tools import EefPlanner, GripperPlanner, default_registry
from robots.arx.gateway.zeva import CosmosPredictor, ZevaPlanner
from robots.arx.mujoco_mapping import MujocoMapping


def extract_calibration(scene):
    """Harness-only extraction: the planner receives robot transforms, never XML."""
    root = ET.parse(scene).getroot()
    if root.find("compiler").get("angle") != "radian":
        raise ValueError("calibration extraction requires radians")
    base = root.find("worldbody/body[@name='right_base_link']")

    def vector(element, name, default):
        return [float(x) for x in element.get(name, default).split()]

    links, parent = [], base
    for i in range(1, 7):
        parent = parent.find(f"body[@name='right_link{i}']")
        joint = parent.find(f"joint[@name='right_joint{i}']")
        if joint.get("type") != "hinge" or joint.get("pos", "0 0 0") != "0 0 0":
            raise ValueError("unsupported joint")
        links.append(
            {
                "position": vector(parent, "pos", "0 0 0"),
                "quaternion": vector(parent, "quat", "1 0 0 0"),
                "axis": vector(joint, "axis", "0 0 1"),
                "limits": vector(joint, "range", "0 0"),
            }
        )
    return Calibration(
        base_position=vector(base, "pos", "0 0 0"),
        base_quaternion=vector(base, "quat", "1 0 0 0"),
        links=links,
        tcp_offset=[0.12957, 0.0, 0.0137564],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", default="robots/arx/manifests/task7_model_a.yaml")
    parser.add_argument("--zeva-host", default="127.0.0.1")
    parser.add_argument("--zeva-port", type=int, default=5581)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    # An explicit commissioning budget override, retained as a new task artifact.
    task_value = json.loads((args.scene / "task.yaml").read_text())
    task_value["max_steps"] = args.max_steps
    task_path = args.output / "task.json"
    task_path.write_text(json.dumps(task_value, indent=2))
    task = load_task_manifest(task_path)
    calibration = extract_calibration(args.scene / "scene.xml")
    (args.output / "robot_calibration.json").write_text(
        calibration.model_dump_json(indent=2)
    )
    kinematics = CommandKinematics(calibration)
    backend = DirectBackend(
        scene=args.scene,
        mapping=args.scene / "mapping.json",
        task=task_path,
        seed=args.seed,
    )
    predictor = CosmosPredictor(
        host=args.zeva_host,
        port=args.zeva_port,
        contract=load_model_contract(args.contract),
        task=task,
        timeout_s=300.0,
    )
    core = None
    zeva = ZevaPlanner(
        predictor,
        lambda: core.policy_observation(),
        execution_steps=task.execution_steps,
    )
    registry = default_registry(
        zeva=zeva,
        gripper=GripperPlanner(MujocoMapping.from_json(args.scene / "mapping.json")),
        eef=EefPlanner(kinematics),
    )
    journal = Journal(args.output / "journal.sqlite3")
    core = ArxSessionCore(
        episode_id="commissioning",
        backend=backend,
        registry=registry,
        journal=journal,
        output=args.output,
        limits=RuntimeLimits(
            max_steps=args.max_steps,
            max_decisions=200,
            max_recoveries=1,
            operation_timeout_s=360.0,
            critic_timeout_s=5.0,
            idle_agent_timeout_s=600.0,
            lease_timeout_s=60.0,
            shutdown_timeout_s=10.0,
        ),
        critic=BaselineMonitor(),
    )
    turns = []
    report = {
        "scene": str(args.scene),
        "mode": "commissioning",
        "forward": "world +X",
        "down": "world -Z",
        "max_steps": args.max_steps,
        "stages": [],
        "status": "running",
    }

    def call(stage, tool, arguments):
        before = core.snapshot()
        index = len(turns)
        request = ToolRequest(
            request_id=f"turn-{index}",
            decision_ref=f"decision-{index}",
            observation_id=before["observation"]["observation_id"],
            control_epoch=before["control_epoch"],
            tool=tool,
            arguments=arguments,
            reason=stage,
        )
        journal.register_decision(request, source="runner", evidence=before)
        result = core.execute(request)
        turn = {
            "turn": index,
            "stage": stage,
            "request": request.model_dump(),
            "observation_before": before,
            "result": result,
            "observation_after": core.snapshot(),
            "predicted_tcp": kinematics.fk(core.commit.command[7:13])[0].tolist(),
        }
        turns.append(turn)
        with (args.output / "turns.jsonl").open("a") as f:
            f.write(json.dumps(turn) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(
            json.dumps(
                {
                    "turn": index,
                    "stage": stage,
                    "status": result["status"],
                    "steps": result["executed_steps"],
                    "result": result["result"],
                }
            ),
            flush=True,
        )
        if result["status"] != "completed" or (core.closed and tool != "arx.finish"):
            raise RuntimeError(f"{stage} stopped: {result['status']} {result['error']}")
        return result

    try:
        core.reset()
        call("vla_four_chunks", "arx.zeva", {"max_chunks": 4})
        for stage, opening in [("close_gripper", 0.0), ("open_gripper", 1.0)]:
            for _ in range(3):
                result = call(
                    stage, "arx.set_gripper", {"opening": opening, "max_steps": 60}
                )
                if result["result"]["command_target_reached"]:
                    break
            else:
                raise RuntimeError("gripper did not converge")
        for _ in range(2):
            call("hold_2s", "arx.hold", {"steps": 15})
        home = kinematics.fk(core.commit.command[7:13])[0]
        for stage, target in [
            ("down_10cm", home + np.array([0.0, 0.0, -0.1])),
            ("back_from_down", home),
            ("forward_10cm", home + np.array([0.1, 0.0, 0.0])),
            ("back_from_forward", home),
        ]:
            for _ in range(30):
                current = kinematics.fk(core.commit.command[7:13])[0]
                delta = target - current
                distance = float(np.linalg.norm(delta))
                if distance < 0.0003:
                    report["stages"].append(
                        {
                            "stage": stage,
                            "predicted_tcp": current.tolist(),
                            "target": target.tolist(),
                            "error_m": distance,
                        }
                    )
                    break
                delta *= min(1.0, 0.0099 / distance)
                call(
                    stage,
                    "arx.move_eef",
                    {
                        "delta_xyz_m": delta.tolist(),
                        "frame": "world",
                        "speed_m_s": 0.03,
                    },
                )
            else:
                raise RuntimeError(f"{stage}: command convergence failed")
        call(
            "finish", "arx.finish", {"reason": "Requested commissioning trace complete"}
        )
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        if not core.closed:
            call(
                "finish_after_error",
                "arx.finish",
                {"reason": "Commissioning failure; preserve evidence"},
            )
        raise
    finally:
        core.close()
        report["video_frames"] = export_public_recording(journal, args.output)
        report["turns"] = len(turns)
        report["executed_steps"] = core.step_index
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        journal.close()


if __name__ == "__main__":
    main()
