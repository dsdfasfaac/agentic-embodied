#!/usr/bin/env python3
"""Read-only evidence, shadow and pilot comparison for ARX real evolution.

This uses Zetta's CandidateBundle and TemporalCritic. Real comparisons are
explicitly pilot comparisons, not simulator same-seed or held-out promotion.
No robot or model connection is opened by this tool.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from zetta.evolution.critic import TemporalCritic
from zetta.evolution.jsonio import file_sha256
from zetta.evolution.models import CandidateBundle

PREFIX = "privileged.interaction."
DISTANCE = "privileged.selected.target_gripper_distance_m"


def read_trial(path):
    journal = path / "private/gateway/journal.sqlite3"
    with sqlite3.connect(f"file:{journal}?mode=ro", uri=True) as db:
        rows = [(kind, json.loads(payload)) for kind, payload in db.execute(
            "SELECT kind, payload FROM records ORDER BY sequence")]
    counts = {}
    for kind, _ in rows:
        counts[kind] = counts.get(kind, 0) + 1
    observations = {p["observation_id"]: p for k, p in rows if k == "ObservationPublished"}
    series = []
    for kind, payload in rows:
        if kind == "real_feature_evidence":
            observation = observations[payload["observation_id"]]
            series.append({"observation_id": payload["observation_id"],
                           "step_index": observation["step_index"],
                           "arrival_verified": observation["hardware"]["arrival_verified"],
                           "features": payload["features"]})
    if not series:
        raise ValueError("trial has no real feature evidence")
    result = json.loads((path / "result.json").read_text())
    distance = [item["features"][DISTANCE] for item in series]
    closed = [item for item in series if item["features"][PREFIX + "gripper_closed"]]
    summary = {
        "trial": str(path.resolve()), "journal_sha256": file_sha256(journal),
        "result_sha256": file_sha256(path / "result.json"),
        "identities": json.loads((path / "identities.json").read_text()),
        "result": result, "event_counts": counts,
        "initial_budget": json.loads((path / "reset.json").read_text())["budget_remaining"],
        "feature_observations": len(series),
        "distance_m": {"initial": distance[0], "minimum": min(distance), "final": distance[-1]},
        "max_lift_m": max(item["features"][PREFIX + "lift_m"] for item in series),
        "first_closed_step": closed[0]["step_index"] if closed else None,
        "first_closed_distance_m": closed[0]["features"][DISTANCE] if closed else None,
        "predicate_counts": {name: sum(bool(item["features"][PREFIX + name]) for item in series)
                             for name in ("gripper_closed", "gripper_contact", "grasped", "success")},
        "verified_task_success": bool(counts.get("task_success", 0)),
        "initial_state": observations[series[0]["observation_id"]]["hardware"]["measured_state"],
        "initial_camera_calibration_sha256": observations[series[0]["observation_id"]]["hardware"]["camera_calibration_sha256"],
    }
    return summary, series


def shadow(bundle_path, trial):
    bundle = CandidateBundle.from_dict(json.loads(bundle_path.read_text()))
    summary, series = read_trial(trial)
    critic = TemporalCritic(bundle.critic_rules)
    proposals = []
    # Reset initializes evidence; live core assesses only post-action arrivals.
    for item in series:
        if item["step_index"] > 0 and item["arrival_verified"]:
            for proposal in critic.evaluate(item["features"], step_index=item["step_index"]):
                proposals.append({"step_index": item["step_index"],
                                  "observation_id": item["observation_id"], **proposal})
    return {"schema_version": "arx.real.shadow.v1", "bundle_sha256": bundle.sha256,
            "bundle_file_sha256": file_sha256(bundle_path),
            "source_journal_sha256": summary["journal_sha256"],
            "proposals": proposals, "robot_commands_sent": False,
            "scope": "trigger replay only; no intervention outcome inferred"}


def compare(parent_path, candidate_path):
    parent, _ = read_trial(parent_path)
    candidate, _ = read_trial(candidate_path)
    if parent["identities"]["bundle"] == candidate["identities"]["bundle"]:
        raise ValueError("comparison requires a changed candidate")
    state_error = [abs(a - b) for a, b in zip(parent["initial_state"], candidate["initial_state"])]
    # Explicit physical matching tolerances; never pretend byte-identical reset.
    tolerances = [.05] * 14
    tolerances[6] = tolerances[13] = .1
    tolerances[12] = .06
    checks = {
        "same_hardware": parent["identities"]["hardware"] == candidate["identities"]["hardware"],
        "same_backend_implementation": bool(parent["identities"].get("backend_implementation")) and parent["identities"].get("backend_implementation") == candidate["identities"].get("backend_implementation"),
        "same_provider": parent["identities"]["feature_provider"] == candidate["identities"]["feature_provider"],
        "same_catalog": parent["identities"]["catalog"] == candidate["identities"]["catalog"],
        "same_task": parent["identities"].get("task") == candidate["identities"].get("task") and "task" in parent["identities"],
        "same_model_contract": parent["identities"].get("model_contract") == candidate["identities"].get("model_contract") and "model_contract" in parent["identities"],
        "same_runtime_limits": parent["identities"].get("runtime_limits") == candidate["identities"].get("runtime_limits") and "runtime_limits" in parent["identities"],
        "same_camera_calibrations": parent["initial_camera_calibration_sha256"] == candidate["initial_camera_calibration_sha256"],
        "start_state_within_tolerance": all(error <= tolerance for error, tolerance in zip(state_error, tolerances)),
        "initial_distance_within_2cm": abs(parent["distance_m"]["initial"] - candidate["distance_m"]["initial"]) <= .02,
        "same_initial_budget": parent["initial_budget"] == candidate["initial_budget"],
        "no_execution_error": parent["result"]["error"] is None and candidate["result"]["error"] is None,
        "verified_arrivals": parent["event_counts"].get("arrival_unverified", 0) == candidate["event_counts"].get("arrival_unverified", 0) == 0,
        "candidate_intervened": candidate["result"]["counts"]["recoveries"] > 0,
    }
    valid = all(checks.values())
    rescued = valid and candidate["verified_task_success"] and not parent["verified_task_success"]
    # One physical pair cannot authorize promotion or establish generalization.
    decision = "continue_validation" if rescued else "reject" if valid else "inconclusive"
    return {"schema_version": "arx.real.pilot_gate.v1", "checks": checks,
            "reset_matching": "approximate physical matching; not simulator same-seed replay",
            "decision": decision, "promoted": False, "attributed_task_rescue": rescued,
            "minimum_distance_gain_m": parent["distance_m"]["minimum"] - candidate["distance_m"]["minimum"],
            "parent": parent, "candidate": candidate,
            "limitations": ["one physical pair", "no held-out or regression trials",
                             "live image placement matching requires visual review"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--trial", type=Path, required=True)
    replay = sub.add_parser("shadow")
    replay.add_argument("--trial", type=Path, required=True)
    replay.add_argument("--bundle", type=Path, required=True)
    gate = sub.add_parser("compare")
    gate.add_argument("--parent-trial", type=Path, required=True)
    gate.add_argument("--candidate-trial", type=Path, required=True)
    for child in (evidence, replay, gate):
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("evolution evidence output is immutable; use a new path")
    if args.command == "evidence":
        result = read_trial(args.trial)[0]
    elif args.command == "shadow":
        result = shadow(args.bundle, args.trial)
    else:
        result = compare(args.parent_trial, args.candidate_trial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output),
                      "decision": result.get("decision"),
                      "proposals": len(result.get("proposals", []))}, sort_keys=True))


if __name__ == "__main__":
    main()
