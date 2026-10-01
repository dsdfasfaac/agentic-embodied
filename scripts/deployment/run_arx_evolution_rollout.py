#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run one fresh frozen ARX trial; emit result path and an explicit exit code."""

import argparse
import json
import sys
import time
import uuid
import threading
from pathlib import Path

from robots.arx.deployment.contracts import EXIT_CODES, RolloutResult, Trial
from robots.arx.deployment.runner import RolloutRunner
from robots.arx.evolution_result import authoritative_success, convert_result
from robots.arx.trajectory_recorder import record_public_trajectory
from zetta.evolution.jsonio import atomic_write_json, file_sha256


def _campaign_trial(args):
    registry = json.loads(args.scene_registry.read_text())
    from zetta.evolution.store import CampaignStore

    frozen = CampaignStore(args.campaign_root).manifest()
    if frozen.runtime.get("evidence_policy") != args.evidence_policy:
        raise ValueError("campaign evidence policy changed")
    if frozen.runtime.get("vla_host") != args.vla_host or frozen.runtime.get("vla_port") != args.vla_port:
        raise ValueError("Zeva endpoint differs from the frozen campaign")
    frozen_privileged = bool(frozen.runtime.get("privileged", False))
    if bool(getattr(args, "privileged", False)) != frozen_privileged:
        raise ValueError("rollout privileged mode differs from the frozen campaign")
    if registry["task_contract"]["task"] != args.task:
        raise ValueError("campaign task and scene registry disagree")
    heldout = "heldout" in args.logical_id
    entries = registry["heldout"] if heldout else registry["entries"]
    matches = [item for item in entries if int(item["reset_seed"]) == args.seed]
    if len(matches) != 1:
        raise ValueError("scheduled scene/reset seed does not identify one frozen trial")
    entry = matches[0]
    scene = Path(entry["scene"])
    for name, filename in {"model": "model.mjb", "mapping": "mapping.json", "task": "task.yaml", "metadata": "metadata.json", "model_contract": None}.items():
        path = args.model_contract if name == "model_contract" else scene / filename
        if file_sha256(path) != entry["hashes"][name]:
            raise ValueError(f"frozen scene input changed: {name}")
    candidate = None
    agent = None
    if args.bundle_sha256 != "none":
        if args.bundle.is_file():
            from zetta.evolution.jsonio import canonical_sha256, read_json
            if canonical_sha256(read_json(args.bundle)) != args.bundle_sha256:
                raise ValueError("candidate bundle digest mismatch")
            candidate = {
                "package": str(args.bundle.resolve()),
                "package_sha256": args.bundle_sha256,
                "contract_sha256": registry["task_contract"]["digest"],
                "catalog_sha256": frozen.tool_catalog_sha256,
                "bootstrap_sha256": frozen.runtime["deployment_bootstrap_sha256"],
                "critic_runtime_limits": str(args.critic_runtime_limits.resolve()),
            }
            agent = json.loads(args.agent_settings.read_text())
        else:
            from robots.arx.critics.packages import load_candidate, PackageManifest
            package = load_candidate(args.bundle)
            if package.sha256 != args.bundle_sha256:
                raise ValueError("candidate package digest mismatch")
            manifest = PackageManifest.model_validate_json(package.manifest_bytes)
            if (manifest.contract_sha256 != registry["task_contract"]["digest"]
                    or manifest.tool_catalog_sha256 != frozen.tool_catalog_sha256
                    or manifest.deployment_bootstrap_sha256 != frozen.runtime["deployment_bootstrap_sha256"]):
                raise ValueError("candidate package differs from frozen campaign identities")
            candidate = {"package": str(args.bundle.resolve()), "package_sha256": package.sha256,
                         "contract_sha256": manifest.contract_sha256, "catalog_sha256": manifest.tool_catalog_sha256,
                         "bootstrap_sha256": manifest.deployment_bootstrap_sha256,
                         "critic_runtime_limits": str(args.critic_runtime_limits.resolve())}
            agent = json.loads(args.agent_settings.read_text())
    elif args.baseline_mode != "strict_pure_vla":
        raise ValueError("active baseline requires a candidate package")
    return Trial.model_validate({
        "schema_version": "arx.rollout.trial.v1", "trial_id": args.logical_id + f"-a{args.attempt_index:03d}",
        "mode": "candidate" if candidate else "baseline",
                        "environment": {"scene": str(scene), "mapping": str(scene / "mapping.json"),
                        "task": str(scene / "task.yaml"), "model_contract": str(args.model_contract.resolve()), "calibration": str(args.calibration.resolve()) if getattr(args, "calibration", None) else None, "privileged": bool(getattr(args, "privileged", False)), "seed": args.seed},
        "vla": {"host": args.vla_host, "port": args.vla_port, "expected_identity": {}},
        "gateway": {"python": sys.executable, "host": "127.0.0.1", "port": args.gateway_port,
                    "runtime_limits": str(args.runtime_limits.resolve())},
        "candidate": candidate, "agent": agent,
        "runner_limits": json.loads(args.runner_limits.read_text()), "evaluation": "none",
    })


def _campaign_run(args):
    started = time.time()
    trial = _campaign_trial(args)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(5):
            if args.output_dir.is_dir():
                with args.heartbeat_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"time": time.time(), "phase": "rollout"}) + "\n")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        result = RolloutRunner(trial, args.output_dir).run()
    finally:
        stop.set()
        thread.join(timeout=6)
    outcome = authoritative_success(args.output_dir / "private/gateway/journal.sqlite3")
    if result.status == "completed":
        result = result.model_copy(update={"outcome": result.outcome.model_copy(update={"task_success": outcome})})
    try:
        index = record_public_trajectory(args.output_dir) if result.status == "completed" else {}
    except (ValueError, OSError, KeyError) as exc:
        index = {"publication_error": type(exc).__name__}
        result = result.model_copy(update={"status": "infrastructure_error", "termination_reason": "public_evidence_incomplete"})
    episode = convert_result(result, logical_id=args.logical_id, attempt_index=args.attempt_index,
                             generation=args.generation, seed=args.seed, policy_rng=args.policy_rng,
                             started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                             elapsed_s=time.time() - started, artifact_index=index)
    if args.result_file != args.output_dir / "episode_record.json":
        raise ValueError("result file must be inside the attempt directory")
    atomic_write_json(args.result_file, episode.as_dict())
    return 0 if episode.status == "valid" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-config", type=Path)
    parser.add_argument("--output", type=Path)
    for name in ("campaign-root", "logical-id", "task", "baseline-mode", "vla-host", "evidence-policy"):
        parser.add_argument("--" + name)
    for name in ("attempt-index", "generation", "seed", "policy-rng", "vla-port", "gateway-port"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    for name in ("scene-registry", "output-dir", "result-file", "heartbeat-file", "runtime-limits", "runner-limits", "model-contract", "critic-runtime-limits", "agent-settings"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--privileged", action="store_true")
    args = parser.parse_args()
    if args.campaign_root:
        if args.evidence_policy not in {"arx_rgb_public_v1", "arx_privileged_v1"}:
            parser.error("unsupported ARX evidence policy")
        return _campaign_run(args)
    if args.trial_config is None or args.output is None:
        parser.error("legacy mode requires --trial-config and --output")
    output = args.output.resolve()
    if output.exists():
        parser.error("output must be a new attempt directory")
    try:
        trial = Trial.model_validate_json(args.trial_config.read_text())
    except Exception as exc:
        output.mkdir(parents=True)
        output.chmod(0o700)
        result = RolloutResult(
            trial_id="invalid-trial",
            attempt_id=uuid.uuid4().hex,
            mode="unknown",
            status="configuration_error",
            termination_reason="invalid_trial",
            cleanup_status="not_started",
            error={"type": type(exc).__name__, "message": "Trial validation failed"},
        )
        atomic_write_json(output / "result.json", result.model_dump())
    else:
        result = RolloutRunner(trial, output).run()
    print(
        json.dumps(
            {
                "result_path": str(output / "result.json"),
                "attempt_id": result.attempt_id,
            }
        )
    )
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
