#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Serve one ARX real-robot episode through the existing gateway."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.contracts import load_model_contract, load_task_manifest
from robots.arx.gateway.contracts import RuntimeLimits
from robots.arx.gateway.real_config import (
    REAL_JOINT_CHANNELS,
    build_real_backend,
    load_real_hardware_config,
    validate_real_hardware_config,
)


@dataclass
class RealCoreFactory:
    hardware_config: str
    hardware_sha256: str
    task: str
    model_contract: str
    output: str
    episode_id: str
    limits: dict
    zeva_host: str
    zeva_port: int
    kinematics_calibration: str | None
    bundle: str | None = None
    tool_catalog: str | None = None
    real_input_contract: str | None = None
    expected_real_input_sha256: str | None = None
    feature_provider: str | None = None
    expected_feature_provider_sha256: str | None = None
    preflight_only: bool = False

    def __call__(self, cancelled, phase_changed):
        from robots.arx.gateway.journal import Journal
        from robots.arx.gateway.motion import Calibration, CommandKinematics
        from robots.arx.gateway.session_core import ArxSessionCore, BaselineMonitor
        from robots.arx.gateway.tools import (
            EefPlanner, PolicyGripperPlanner, default_registry,
        )
        from robots.arx.gateway.zeva import CosmosPredictor, ZevaPlanner
        from robots.arx.gateway.bundle_runtime import BundleMonitor, RealBundleReentry, RealFeatureProvider
        from robots.arx.deployment.bundle_program import compile_programs
        from robots.arx.deployment.real_input import LiveCapabilities, RealInputContract, preflight_real_bundle, _load_bundle

        config = load_real_hardware_config(
            Path(self.hardware_config), self.hardware_sha256
        )
        task_path, model_path = Path(self.task), Path(self.model_contract)
        validate_real_hardware_config(config, task_path, model_path)
        task, model = load_task_manifest(task_path), load_model_contract(model_path)
        limits = RuntimeLimits.model_validate(self.limits)
        backend = None
        try:
            core = None
            predictor = CosmosPredictor(
                host=self.zeva_host, port=self.zeva_port,
                contract=model, task=task,
                timeout_s=limits.operation_timeout_s,
            )
            zeva = ZevaPlanner(
                predictor, lambda: core.policy_observation(),
                execution_steps=task.execution_steps,
                action_horizon=model.action_horizon,
            )
            eef = None
            if self.kinematics_calibration is not None:
                calibration = Calibration.model_validate_json(
                    Path(self.kinematics_calibration).read_text()
                )
                eef = EefPlanner(CommandKinematics(calibration))
            provider = None
            bundle = None
            programs = {}
            if self.bundle:
                if not all((self.tool_catalog, self.real_input_contract,
                            self.expected_real_input_sha256, self.feature_provider,
                            self.expected_feature_provider_sha256)):
                    raise ValueError("bundle requires frozen catalog, real input, and feature provider")
                provider = RealFeatureProvider(
                    Path(self.feature_provider), self.expected_feature_provider_sha256
                )
                if hasattr(provider.impl, "validate_hardware"):
                    provider.impl.validate_hardware(config)
                bundle, _ = _load_bundle(Path(self.bundle))
                contract = RealInputContract.model_validate_json(Path(self.real_input_contract).read_text())
                programs = compile_programs(
                    bundle, max_tool_calls=contract.max_recovery_tool_calls,
                    max_physical_steps=limits.max_steps,
                )
                if any(call.tool == "arx.move_eef" for program in programs.values() for call in program.calls) and eef is None:
                    raise ValueError("bundle EEF recovery requires reviewed kinematics calibration")
            monitor = BundleMonitor(bundle, provider) if bundle else BaselineMonitor()
            reentry = (RealBundleReentry(
                bundle, provider,
                max_sensor_age_ms=config.timing.max_sensor_age_ms,
                max_sensor_skew_ms=config.timing.max_sensor_skew_ms,
                monitor=monitor,
            ) if bundle else None)
            registry = default_registry(
                zeva=zeva,
                gripper=PolicyGripperPlanner(
                    closed_policy=config.right_gripper_closed_policy,
                    open_policy=config.right_gripper_open_policy,
                    max_policy_step=task.control.max_gripper_step,
                ),
                eef=eef, reentry=reentry,
            )
            if bundle:
                catalog = registry.freeze()
                live = LiveCapabilities(
                    schema_version="arx.real.capabilities.v1",
                    robot_id=f"{config.left.can_port}+{config.right.can_port}",
                    cameras=[{
                        "name": c.name, "device_id": c.serial,
                        "calibration_id": c.calibration_id,
                        "calibration_sha256": c.calibration_sha256,
                        "width": c.width, "height": c.height,
                        "channels": 3, "dtype": "uint8", "color_order": "RGB",
                    } for c in config.cameras],
                    depth_cameras=[c.name.removesuffix("_rgb") + "_depth_mm"
                                   for c in config.cameras if c.depth_enabled],
                    joint_channels=list(REAL_JOINT_CHANNELS),
                    feature_sources=provider.sources,
                    tool_catalog_sha256=catalog["catalog_sha256"],
                )
                report = preflight_real_bundle(
                    bundle_path=Path(self.bundle), task_manifest_path=task_path,
                    model_contract_path=model_path,
                    tool_catalog_path=Path(self.tool_catalog),
                    real_contract_path=Path(self.real_input_contract),
                    live_capabilities=live,
                    expected_real_contract_sha256=self.expected_real_input_sha256,
                )
                if json.loads(Path(self.tool_catalog).read_text()) != catalog:
                    raise ValueError("frozen catalog content differs from live registry")
                if set(programs) != {p["recovery_id"] for p in report["recovery_plans"]}:
                    raise ValueError("compiled recovery plan differs from preflight")
                if self.preflight_only:
                    return report
                Path(self.output, "bundle-preflight.json").write_text(json.dumps(report, sort_keys=True))
            backend = build_real_backend(
                config=config, task_path=task_path, model_path=model_path
            )
            core = ArxSessionCore(
                episode_id=self.episode_id,
                backend=backend,
                registry=registry,
                journal=Journal(Path(self.output) / "journal.sqlite3"),
                output=Path(self.output),
                limits=limits,
                critic=monitor,
                bindings=tuple(program.binding for program in programs.values()),
                programs=programs,
                package_sha256=bundle.sha256 if bundle else "baseline",
                cancel_requested=cancelled,
                phase_changed=phase_changed,
                privileged=False,
            )
            return core
        except BaseException:
            if backend is not None:
                backend.close()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hardware-config", "task", "model-contract"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-hardware-sha256", required=True)
    parser.add_argument("--kinematics-calibration", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--tool-catalog", type=Path)
    parser.add_argument("--real-input-contract", type=Path)
    parser.add_argument("--expected-real-input-sha256")
    parser.add_argument("--feature-provider", type=Path)
    parser.add_argument("--expected-feature-provider-sha256")
    parser.add_argument("--zeva-host", default="127.0.0.1")
    parser.add_argument("--zeva-port", type=int, default=5581)
    parser.add_argument("--listen-host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8091)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()

    config = load_real_hardware_config(
        args.hardware_config, args.expected_hardware_sha256
    )
    validate_real_hardware_config(config, args.task, args.model_contract)
    if args.check_config:
        if args.bundle:
            if args.runtime_config is None:
                parser.error("bundle --check-config requires --runtime-config")
            limits = RuntimeLimits.model_validate_json(args.runtime_config.read_text())
            report = RealCoreFactory(
                str(args.hardware_config), args.expected_hardware_sha256,
                str(args.task), str(args.model_contract), "/tmp", "preflight", limits.model_dump(),
                args.zeva_host, args.zeva_port,
                str(args.kinematics_calibration) if args.kinematics_calibration else None,
                str(args.bundle),
                str(args.tool_catalog) if args.tool_catalog else None,
                str(args.real_input_contract) if args.real_input_contract else None,
                args.expected_real_input_sha256,
                str(args.feature_provider) if args.feature_provider else None,
                args.expected_feature_provider_sha256,
                True,
            )(lambda: False, lambda _: None)
            print(json.dumps(dict(report, hardware_opened=False), sort_keys=True))
            return
        print(json.dumps({
            "valid": True, "schema_version": config.schema_version,
            "arm_transport": config.arm_transport,
            "camera_serials": {camera.name: camera.serial for camera in config.cameras},
            "control_hz": config.timing.control_hz,
            "hardware_opened": False,
        }, sort_keys=True))
        return
    if args.runtime_config is None or args.output is None:
        parser.error("--runtime-config and --output are required to serve hardware")
    task = load_task_manifest(args.task)
    limits = RuntimeLimits.model_validate_json(args.runtime_config.read_text())
    if limits.max_steps > task.max_steps:
        parser.error("gateway max_steps exceeds frozen task limit")
    args.output.mkdir(parents=True, exist_ok=False)
    os.chmod(args.output, 0o700)
    episode_id = uuid.uuid4().hex
    agent, harness = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    credentials = args.output / "capabilities.json"
    with credentials.open("x") as stream:
        os.chmod(credentials, 0o600)
        json.dump({
            "episode_id": episode_id,
            "agent_capability": agent,
            "harness_capability": harness,
        }, stream)
    factory = RealCoreFactory(
        str(args.hardware_config), args.expected_hardware_sha256,
        str(args.task), str(args.model_contract), str(args.output),
        episode_id, limits.model_dump(),
        args.zeva_host, args.zeva_port,
        str(args.kinematics_calibration) if args.kinematics_calibration else None,
        str(args.bundle) if args.bundle else None,
        str(args.tool_catalog) if args.tool_catalog else None,
        str(args.real_input_contract) if args.real_input_contract else None,
        args.expected_real_input_sha256,
        str(args.feature_provider) if args.feature_provider else None,
        args.expected_feature_provider_sha256,
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
            host=args.listen_host, port=args.listen_port, access_log=False,
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
