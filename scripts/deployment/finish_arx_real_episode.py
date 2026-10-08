#!/usr/bin/env python3
"""Home an ended, empty ARX rollout before disabling its controller.

This is a separate, audited motion phase outside the rollout budget. Never
call it for a recoverable critic interruption: the recovery needs the motors.
On failed observations, homing or arrival verification, leave the controller
running. A possible held tube requires operator unloading first.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.deployment.audit_arx_live_observation import audit
from scripts.deployment.stage_arx_picktube_start import stage
from scripts.deployment.replay_arx_picktube_home import replay_home


def finish(*, trial: Path, hardware: Path, hardware_sha: str,
           task: Path, model: Path, output: Path, controller: Path,
           operator_unloaded: bool = False) -> dict:
    # The completed runner result is the terminal boundary, including faults.
    result = json.loads((trial / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "arx.real.post_episode.v1",
              "trial": str(trial), "termination_reason": result["termination_reason"],
              "status": "checking", "controller_disabled": False,
              "operator_confirmed_unloaded": operator_unloaded,
              "homing_is_rollout_step": False}
    try:
        import sqlite3
        with sqlite3.connect(f"file:{trial / 'private/gateway/journal.sqlite3'}?mode=ro", uri=True) as db:
            row = db.execute("SELECT payload FROM records WHERE kind='real_feature_evidence' ORDER BY sequence DESC LIMIT 1").fetchone()
            if row is None:
                raise ValueError("no terminal feature evidence; leave controller enabled")
            features = json.loads(row[0])["features"]
            held = db.execute("SELECT 1 FROM records WHERE kind='task_success' LIMIT 1").fetchone() is not None
        prefix = "privileged.interaction."
        held = held or any(features.get(prefix + name) is True for name in ("gripper_contact", "grasped", "success"))
        if held and not operator_unloaded:
            report["status"] = "requires_unloading"
            return report
        # After explicit unloading, the target may have left the scene. Home
        # still requires fresh synchronized hardware feedback and healthy
        # devices, but target visibility is no longer a prerequisite.
        before = audit(hardware, hardware_sha, task, model,
                       require_features=not operator_unloaded)
        (output / "before.json").write_text(json.dumps(before, indent=2) + "\n")
        fresh = before["features"]
        if (not operator_unloaded and fresh[prefix + "gripper_closed"] and
                fresh["privileged.selected.target_gripper_distance_m"] <= 0.05):
            report["status"] = "requires_unloading"
            return report
        homing = replay_home(hardware, hardware_sha, task,
                             output / "recorded-homing.json", execute=True)
        if homing["status"] != "complete":
            raise ValueError("homing incomplete; leave controller enabled")
        # The recorded path preserved grippers. Only once the arm is home may
        # the already-unloaded grippers be staged to the frozen open start.
        from robots.arx.contracts import load_task_manifest
        from robots.arx.gateway.real_config import load_real_hardware_config
        import numpy as np
        if "final_state" in homing:
            expected = np.asarray(load_task_manifest(task).start_state)
            tolerance = np.asarray(load_real_hardware_config(hardware, hardware_sha).timing.position_tolerance)
            axes = [i for i in range(14) if i not in (6, 13)]
            if np.any(np.abs(np.asarray(homing["final_state"])[axes] - expected[axes]) > tolerance[axes]):
                raise ValueError("arm not at home; gripper staging refused")
        opened = stage(hardware, hardware_sha, task, 150, output / "open-at-home.json", 0.015)
        if opened["status"] != "complete":
            raise ValueError("empty gripper staging incomplete; leave controller enabled")
        after = audit(hardware, hardware_sha, task, model,
                      require_features=not operator_unloaded)
        (output / "after.json").write_text(json.dumps(after, indent=2) + "\n")
        if after["task_start_eligible"] is not True:
            raise ValueError("measured home verification failed; leave controller enabled")
        stopped = subprocess.run(["bash", str(controller), "stop"], capture_output=True, text=True)
        report["controller_stop"] = {"returncode": stopped.returncode,
                                     "stdout": stopped.stdout, "stderr": stopped.stderr}
        if stopped.returncode:
            raise ValueError("controller disable failed")
        report.update(status="homed_and_disabled", controller_disabled=True)
        return report
    except Exception as exc:
        report.update(status="failed_keep_enabled", error=str(exc))
        return report
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial", "hardware-config", "task", "model-contract", "output", "controller"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--hardware-sha256", required=True)
    parser.add_argument("--execute", action="store_true", help="Authorize post-episode homing and disable")
    parser.add_argument("--unloaded", action="store_true",
                        help="Use only after operator confirmation that the held tube has been removed")
    args = parser.parse_args()
    if not args.execute:
        parser.error("homing is robot motion; --execute is required")
    result = finish(trial=args.trial, hardware=args.hardware_config,
                    hardware_sha=args.hardware_sha256, task=args.task,
                    model=args.model_contract, output=args.output, controller=args.controller,
                    operator_unloaded=args.unloaded)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "homed_and_disabled" else 2)


if __name__ == "__main__":
    main()
