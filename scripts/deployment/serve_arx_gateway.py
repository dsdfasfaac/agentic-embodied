#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Serve one frozen ARX episode with baseline or registered critics (OSMesa)."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path

from robots.arx.gateway.contracts import RuntimeLimits


@dataclass
class CoreFactory:
    scene: str
    mapping: str
    task: str
    contract: str
    output: str
    seed: int
    episode_id: str
    zeva_host: str
    zeva_port: int
    limits: dict
    calibration: str | None
    package: str | None = None
    critic_limits: dict | None = None
    privileged: bool = False

    def __call__(self, cancelled, phase_changed):
        from robots.arx.contracts import load_model_contract, load_task_manifest
        from robots.arx.gateway.backend import DirectBackend
        from robots.arx.gateway.journal import Journal
        from robots.arx.gateway.motion import Calibration, CommandKinematics
        from robots.arx.gateway.session_core import ArxSessionCore, BaselineMonitor
        from robots.arx.gateway.tools import (
            EefPlanner,
            GripperPlanner,
            default_registry,
        )
        from robots.arx.gateway.zeva import CosmosPredictor, ZevaPlanner
        from robots.arx.mujoco_mapping import MujocoMapping

        task, contract = (
            load_task_manifest(self.task),
            load_model_contract(self.contract),
        )
        limits = RuntimeLimits.model_validate(self.limits)
        if limits.max_steps > task.max_steps:
            raise ValueError("frozen gateway budget exceeds environment task budget")
        predictor = CosmosPredictor(
            host=self.zeva_host,
            port=self.zeva_port,
            contract=contract,
            task=task,
            timeout_s=limits.operation_timeout_s,
        )
        core = None
        zeva = ZevaPlanner(
            predictor,
            lambda: core.policy_observation(),
            execution_steps=task.execution_steps,
            action_horizon=contract.action_horizon,
        )
        eef = None
        if self.calibration:
            calibration = Calibration.model_validate_json(
                Path(self.calibration).read_text()
            )
            eef = EefPlanner(CommandKinematics(calibration))
        critic = BaselineMonitor()
        bindings = ()
        package_sha256 = "baseline"
        reentry = None
        if self.package:
            from robots.arx.critics import ArxCriticRegistry, WorkerLimits
            from robots.arx.critics.registry import AlwaysIneligibleReentry, RgbFeatureReentry

            if str(self.package).endswith(".json"):
                from zetta.evolution.jsonio import read_json, canonical_sha256
                from zetta.evolution.models import CandidateBundle
                from zetta.evolution.critic import TemporalCritic
                from robots.arx.gateway.contracts import Assessment, Proposal, RecoveryBinding
                bundle = CandidateBundle.from_dict(read_json(Path(self.package)))
                class BundleCritic:
                    sha256 = bundle.sha256
                    def __init__(self): self.temporal = TemporalCritic(bundle.critic_rules)
                    def reset(self, observation, images): self.temporal.reset()
                    def lifecycle(self, event): pass
                    def observe(self, observation, images):
                        events = []
                        for item in self.temporal.evaluate(observation, step_index=observation["step_index"]):
                            rule = next(r for r in bundle.critic_rules if r.rule_id == item["rule_id"])
                            events.append(Proposal(detector_id="bundle", failure_mode=rule.rule_id, rule_id=rule.rule_id, evidence_observation_ids=[observation["observation_id"]], reason_code=rule.rule_id, summary=rule.proposal))
                        return Assessment(critic_id="bundle", observation_id=observation["observation_id"], step_index=observation["step_index"], status="failure" if events else "clear", events=events)
                critic = BundleCritic()
                bindings = tuple(RecoveryBinding(binding_id=r.recovery_id, failure_modes=list(r.trigger_rule_ids), skill_entrypoint="bundle", allowed_tools=[s.tool for s in r.steps], max_recovery_steps=len(r.steps), max_agent_decisions=8, monitor_policy="recovery_local", reentry_policy_id=r.recovery_id) for r in bundle.recovery_rules)
                package_sha256 = bundle.sha256
                class BundleReentry:
                    def inspect(self, args, context):
                        obs = context["observations"][-1]
                        privileged = obs.get("privileged", {})
                        interaction = privileged.get("interaction", {})
                        eligible = "privileged" in obs and not interaction.get("failure", False)
                        return {"schema_version":"arx.reentry.assessment.v1", "recovery_id":context["recovery_id"], "observation_id":args.observation_ids[-1], "policy_id":context["policy_id"], "status":"eligible" if eligible else "ineligible", "checks":[{"check_id":"privileged-clearance", "status":"pass" if eligible else "fail", "evidence_ids":args.observation_ids, "reason_code":"privileged_clear" if eligible else "privileged_failure"}]}
                reentry = BundleReentry()
            else:
                critics = ArxCriticRegistry(
                    limits=WorkerLimits.model_validate(self.critic_limits)
                )
                package = critics.register_package(self.package)
                if (
                    package.critic_manifest.evaluation_timeout_ms
                    > limits.critic_timeout_s * 1000
                ):
                    raise ValueError("critic timeout exceeds gateway barrier deadline")
                critics.preflight()
                critic = critics.freeze()
                bindings = package.bindings
                package_sha256 = package.sha256
                reentry = (
                    RgbFeatureReentry(package, critics.limits)
                    if package.json("reentry/manifest.json")["implementation"] == "rgb_feature_threshold"
                    else AlwaysIneligibleReentry()
                )
        registry = default_registry(
            zeva=zeva,
            eef=eef,
            reentry=reentry,
            gripper=GripperPlanner(
                MujocoMapping.from_json(self.mapping),
                command_offsets=task.control.gripper_command_offsets,
            ),
        )
        backend = DirectBackend(
            scene=self.scene, mapping=self.mapping, task=self.task, seed=self.seed
        )
        core = ArxSessionCore(
            episode_id=self.episode_id,
            backend=backend,
            registry=registry,
            journal=Journal(Path(self.output) / "journal.sqlite3"),
            output=Path(self.output),
            limits=limits,
            critic=critic,
            bindings=bindings,
            package_sha256=package_sha256,
            cancel_requested=cancelled,
            phase_changed=phase_changed,
            privileged=self.privileged,
        )
        return core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "mapping", "task", "contract", "output", "runtime-config"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--package", type=Path, help="Sealed candidate directory; frozen before reset"
    )
    mode.add_argument("--bundle", type=Path, help="Structured CandidateBundle JSON")
    parser.add_argument("--critic-runtime-config", type=Path)
    mode.add_argument(
        "--baseline",
        action="store_true",
        help="Explicit no-intervention monitor",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--privileged", action="store_true")
    parser.add_argument("--zeva-host", default="127.0.0.1")
    parser.add_argument("--zeva-port", default=5581, type=int)
    parser.add_argument(
        "--listen-host", default="127.0.0.1", choices=("127.0.0.1", "::1")
    )
    parser.add_argument("--listen-port", default=8091, type=int)
    args = parser.parse_args()
    limits = RuntimeLimits.model_validate_json(args.runtime_config.read_text())
    if args.package and not args.critic_runtime_config:
        parser.error("--package requires --critic-runtime-config")
    if args.bundle:
        args.package = args.bundle
    args.output.mkdir(parents=True, exist_ok=False)
    os.chmod(args.output, 0o700)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    episode_id = uuid.uuid4().hex
    agent, harness = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    credentials = args.output / "capabilities.json"
    with credentials.open("x") as stream:
        os.chmod(credentials, 0o600)
        json.dump(
            {
                "episode_id": episode_id,
                "agent_capability": agent,
                "harness_capability": harness,
            },
            stream,
        )
    factory = CoreFactory(
        str(args.scene),
        str(args.mapping),
        str(args.task),
        str(args.contract),
        str(args.output),
        args.seed,
        episode_id,
        args.zeva_host,
        args.zeva_port,
        limits.model_dump(),
        str(args.calibration) if args.calibration else None,
        str(args.package) if args.package else None,
        json.loads(args.critic_runtime_config.read_text())
        if args.critic_runtime_config
        else None,
        privileged=args.privileged,
    )
    import uvicorn

    from robots.arx.gateway.service import create_app
    from robots.arx.gateway.worker import EpisodeWorker

    worker = EpisodeWorker(
        factory=factory, output=args.output, episode_id=episode_id, limits=limits
    )
    try:
        uvicorn.run(
            create_app(worker, agent_capability=agent, harness_capability=harness),
            host=args.listen_host,
            port=args.listen_port,
            access_log=False,
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
