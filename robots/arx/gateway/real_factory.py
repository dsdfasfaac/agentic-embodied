"""Construct the shared real-robot gateway core for serving and continuation.

Factory construction and preflight do not publish robot commands. Hardware
acquisition begins only when the factory builds a live backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from robots.arx.contracts import load_model_contract, load_task_manifest
from robots.arx.gateway.contracts import RuntimeLimits
from robots.arx.gateway.real_config import (
    REAL_ARX_CURRENT_CHANNELS,
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
    allow_learned_pregrasp_commissioning: bool = False
    grasp_config: str | None = None
    expected_grasp_config_sha256: str | None = None

    def __call__(self, cancelled, phase_changed):
        from robots.arx.gateway.journal import Journal
        from robots.arx.gateway.motion import Calibration, CommandKinematics
        from robots.arx.gateway.session_core import ArxSessionCore, BaselineMonitor
        from robots.arx.gateway.tools import (
            EefPlanner,
            PolicyGripperPlanner,
            default_registry,
        )
        from robots.arx.gateway.zeva import CosmosPredictor, ZevaPlanner
        from robots.arx.gateway.bundle_runtime import (
            BundleMonitor,
            RealBundleReentry,
            RealFeatureProvider,
        )
        from robots.arx.deployment.bundle_program import compile_programs
        from robots.arx.deployment.real_input import (
            LiveCapabilities,
            RealInputContract,
            preflight_real_bundle,
            _load_bundle,
        )

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
                host=self.zeva_host,
                port=self.zeva_port,
                contract=model,
                task=task,
                timeout_s=limits.operation_timeout_s,
            )
            zeva = ZevaPlanner(
                predictor,
                lambda: core.policy_observation(),
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
                if not all(
                    (
                        self.tool_catalog,
                        self.real_input_contract,
                        self.expected_real_input_sha256,
                        self.feature_provider,
                        self.expected_feature_provider_sha256,
                    )
                ):
                    raise ValueError(
                        "bundle requires frozen catalog, real input, and feature provider"
                    )
                provider = RealFeatureProvider(
                    Path(self.feature_provider), self.expected_feature_provider_sha256
                )
                provider.validate_hardware(config)
                bundle, _ = _load_bundle(Path(self.bundle))
                contract = RealInputContract.model_validate_json(
                    Path(self.real_input_contract).read_text()
                )
                programs = compile_programs(
                    bundle,
                    max_tool_calls=contract.max_recovery_tool_calls,
                    max_physical_steps=limits.max_steps,
                    nominal_chunk_steps=task.execution_steps,
                )
                if (
                    any(
                        call.tool == "arx.move_eef"
                        for program in programs.values()
                        for call in program.calls
                    )
                    and eef is None
                ):
                    raise ValueError(
                        "bundle EEF recovery requires reviewed kinematics calibration"
                    )
            monitor = (
                BundleMonitor(
                    bundle,
                    provider,
                    terminal_feature=(
                        "privileged.interaction.success"
                        if task.name == "pickup_test_tube"
                        else None
                    ),
                )
                if bundle
                else BaselineMonitor()
            )
            reentry = (
                RealBundleReentry(
                    bundle,
                    provider,
                    max_sensor_age_ms=config.timing.max_sensor_age_ms,
                    max_sensor_skew_ms=config.timing.max_sensor_skew_ms,
                    monitor=monitor,
                )
                if bundle
                else None
            )
            grasp = None
            if self.grasp_config:
                from zetta.evolution.jsonio import file_sha256
                from robots.arx.gateway.grasp_contracts import GraspRecoveryConfig
                from robots.arx.gateway.grasp_recovery import (
                    GraspRecovery,
                    GraspReentry,
                    TargetVerifiedGripperPlanner,
                )
                from robots.arx.deployment.picktube_grasp_observer import (
                    PickTubeGraspObserver,
                )

                if (
                    file_sha256(Path(self.grasp_config))
                    != self.expected_grasp_config_sha256
                ):
                    raise ValueError("grasp configuration SHA mismatch")
                settings = GraspRecoveryConfig.model_validate_json(
                    Path(self.grasp_config).read_text()
                )
                if (
                    (
                        settings.learned_pregrasp_commissioning
                        or settings.learned_grasp_commissioning
                    )
                    and not settings.learned_gripper_transfer_verified
                    and not self.allow_learned_pregrasp_commissioning
                ):
                    raise ValueError(
                        "unverified learned transfer requires the supervised commissioning harness"
                    )
                settings.validate_execution_phases(
                    call.arguments.get("phase", "pregrasp")
                    for program in programs.values()
                    for call in program.calls
                    if call.tool == "arx.execute_grasp"
                )
                if provider is None or task.name != "pickup_test_tube":
                    raise ValueError(
                        "PickTube grasp tools require the real PickTube observer"
                    )
                grasp = GraspRecovery(
                    settings,
                    PickTubeGraspObserver(
                        provider.impl, cloud_mode=settings.target_cloud_mode
                    ),
                    closed_policy=config.right_gripper_closed_policy,
                    open_policy=config.right_gripper_open_policy,
                    control_hz=config.timing.control_hz,
                    joint_bounds=tuple(
                        zip(config.right.joint_min_rad, config.right.joint_max_rad)
                    ),
                )
                reentry = GraspReentry(reentry, grasp)
            gripper = PolicyGripperPlanner(
                closed_policy=config.right_gripper_closed_policy,
                open_policy=config.right_gripper_open_policy,
                max_policy_step=task.control.max_gripper_step,
            )
            if grasp is not None:
                gripper = TargetVerifiedGripperPlanner(gripper, grasp)
            registry = default_registry(
                zeva=zeva,
                gripper=gripper,
                eef=eef,
                reentry=reentry,
                grasp=grasp,
            )
            if bundle:
                catalog = registry.freeze()
                live = LiveCapabilities(
                    schema_version="arx.real.capabilities.v1",
                    robot_id=f"{config.left.can_port}+{config.right.can_port}",
                    cameras=[
                        {
                            "name": c.name,
                            "device_id": c.serial,
                            "calibration_id": c.calibration_id,
                            "calibration_sha256": c.calibration_sha256,
                            "width": c.width,
                            "height": c.height,
                            "channels": 3,
                            "dtype": "uint8",
                            "color_order": "RGB",
                        }
                        for c in config.cameras
                    ],
                    depth_cameras=[
                        c.name.removesuffix("_rgb") + "_depth_mm"
                        for c in config.cameras
                        if c.depth_enabled
                    ],
                    joint_channels=list(REAL_JOINT_CHANNELS),
                    auxiliary_channels=(
                        list(REAL_ARX_CURRENT_CHANNELS)
                        if config.arm_transport == "arx_ros2"
                        else []
                    ),
                    feature_sources=provider.sources,
                    tool_catalog_sha256=catalog["catalog_sha256"],
                )
                report = preflight_real_bundle(
                    bundle_path=Path(self.bundle),
                    task_manifest_path=task_path,
                    model_contract_path=model_path,
                    tool_catalog_path=Path(self.tool_catalog),
                    real_contract_path=Path(self.real_input_contract),
                    live_capabilities=live,
                    expected_real_contract_sha256=self.expected_real_input_sha256,
                )
                if json.loads(Path(self.tool_catalog).read_text()) != catalog:
                    raise ValueError(
                        "frozen catalog content differs from live registry"
                    )
                if set(programs) != {
                    p["recovery_id"] for p in report["recovery_plans"]
                }:
                    raise ValueError("compiled recovery plan differs from preflight")
                if self.preflight_only:
                    if grasp is not None:
                        report["grasp_motion_enabled"] = grasp.config.motion_enabled
                    return report
                if (
                    grasp is not None
                    and not grasp.config.motion_enabled
                    and any(
                        call.tool == "arx.execute_grasp"
                        for program in programs.values()
                        for call in program.calls
                    )
                ):
                    raise ValueError(
                        "grasp motion is disabled; complete commissioning and freeze an enabled configuration"
                    )
                Path(self.output, "bundle-preflight.json").write_text(
                    json.dumps(report, sort_keys=True)
                )
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
