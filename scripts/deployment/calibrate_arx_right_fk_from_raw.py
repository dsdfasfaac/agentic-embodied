#!/usr/bin/env python3
"""Fit right controller-EE FK from PickTube joint and end_pos recordings.

The nominal chain supplies link axes/geometry. Forty sorted episodes fit only
the base translation and link-six endpoint offset; the remaining episodes are
held out. The exported calibration is for the controller's end_pos reference,
not an independently measured gripper contact point. No hardware is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.gateway.motion import Calibration, CommandKinematics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(raw_root: Path, nominal_path: Path) -> tuple[Calibration, dict]:
    episodes = sorted(raw_root.glob("*/data.npz"))
    if len(episodes) < 50:
        raise ValueError("at least 50 recorded PickTube episodes are required")
    train, held_out = episodes[:40], episodes[40:]
    nominal = Calibration.model_validate_json(nominal_path.read_text())
    zero = nominal.model_copy(update={
        "base_position": [0.0, 0.0, 0.0], "tcp_offset": [0.0, 0.0, 0.0],
    })
    fk = CommandKinematics(zero)
    rows, values = [], []
    episode_data = {}
    for path in episodes:
        states = np.load(path)["observation.state"]
        if states.ndim != 2 or states.shape[1] != 54 or not np.isfinite(states).all():
            raise ValueError(f"invalid 54D recorded state: {path}")
        episode_data[path] = states
        if path in train:
            for state in states[::10]:
                p, r, _ = fk.fk(state[7:13])
                rows.append(np.hstack((np.eye(3), r)))
                values.append(state[48:51] - p)
    x = np.linalg.lstsq(np.concatenate(rows), np.concatenate(values), rcond=None)[0]
    if not np.isfinite(x).all() or abs(x[1]) > 0.01 or abs(x[4]) > 0.01:
        raise ValueError("fitted right FK base/tool offset is implausible")
    all_joints = np.concatenate([states[:, 7:13] for states in episode_data.values()])
    low = np.maximum(-10.0, np.min(all_joints, axis=0) - 0.05)
    high = np.minimum(10.0, np.max(all_joints, axis=0) + 0.05)
    final = nominal.model_dump(mode="json")
    final["base_position"] = x[:3].tolist()
    final["tcp_offset"] = x[3:].tolist()
    for index, link in enumerate(final["links"]):
        link["limits"] = [float(low[index]), float(high[index])]
    calibration = Calibration.model_validate(final)
    validated_fk = CommandKinematics(calibration)
    groups = {}
    for name, group in (("fit", train), ("held_out", held_out)):
        position_errors, orientation_errors = [], []
        for path in group:
            for state in episode_data[path][::10]:
                p, r, _ = validated_fk.fk(state[7:13])
                position_errors.append(float(np.linalg.norm(p - state[48:51])))
                recorded_r = Rotation.from_euler("xyz", state[51:54]).as_matrix()
                orientation_errors.append(float(np.degrees(
                    Rotation.from_matrix(recorded_r.T @ r).magnitude()
                )))
        groups[name] = {
            "episodes": len(group), "samples": len(position_errors),
            "position_median_m": float(np.median(position_errors)),
            "position_p95_m": float(np.percentile(position_errors, 95)),
            "position_max_m": float(max(position_errors)),
            "orientation_p95_deg": float(np.percentile(orientation_errors, 95)),
            "orientation_max_deg": float(max(orientation_errors)),
        }
    if (groups["held_out"]["position_p95_m"] > 0.005
            or groups["held_out"]["position_max_m"] > 0.01
            or groups["held_out"]["orientation_p95_deg"] > 1.0):
        raise ValueError("held-out right controller FK error exceeds acceptance thresholds")
    report = {
        "schema_version": "arx.right.controller_ee_fk.audit.v1",
        "nominal_chain_sha256": sha256(nominal_path),
        "source_episode_data_sha256": {path.parent.name: sha256(path) for path in episodes},
        "fit": groups["fit"], "held_out": groups["held_out"],
        "fitted_base_translation_m": x[:3].tolist(),
        "fitted_controller_ee_offset_m": x[3:].tolist(),
        "nominal_gripper_tool_offset_m": nominal.tcp_offset,
        "controller_joint_envelope_rad": {
            "min": low.tolist(), "max": high.tolist(),
            "source": "all 50 recorded PickTube episodes plus 0.05 rad margin",
        },
        "scope": "recorded controller end_pos, not live hardware or gripper contact point",
    }
    return calibration, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/home/dodo/chenfu/data/raw/PickTube"))
    parser.add_argument("--nominal-chain", type=Path,
                        default=root / "robots/arx/manifests/real/ac_one_nominal_chain.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    calibration, report = fit(args.raw_root, args.nominal_chain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(calibration.model_dump_json(indent=2) + "\n")
    report["calibration_sha256"] = sha256(args.output)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"calibration": str(args.output),
                      "calibration_sha256": report["calibration_sha256"],
                      "held_out": report["held_out"]}, sort_keys=True))


if __name__ == "__main__":
    main()
