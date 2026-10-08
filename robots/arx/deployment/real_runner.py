"""One real-robot CandidateBundle attempt using the shared rollout journal/client."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from robots.arx.gateway.contracts import RuntimeLimits
from zetta.evolution.jsonio import file_sha256

from .bundle_program import compile_programs
from .real_input import RealInputContract, _load_bundle
from .runner import RolloutRunner


class RealBundleRunner(RolloutRunner):
    def __init__(
        self,
        *,
        output,
        python,
        hardware_config,
        hardware_sha256,
        task,
        model_contract,
        runtime_config,
        runner_limits,
        bundle,
        catalog,
        real_input_contract,
        real_input_sha256,
        feature_provider,
        feature_provider_sha256,
        kinematics_calibration,
        zeva_host,
        zeva_port,
        listen_host,
        listen_port,
        grasp_config=None,
        grasp_config_sha256=None,
    ):
        candidate = SimpleNamespace(
            package=str(bundle),
            package_sha256="",
            catalog_sha256="",
            contract_sha256=real_input_sha256,
        )
        gateway = SimpleNamespace(
            python=str(python),
            host=listen_host,
            port=listen_port,
            runtime_limits=str(runtime_config),
        )
        trial = SimpleNamespace(
            trial_id="real-" + Path(output).name,
            mode="candidate",
            environment=SimpleNamespace(task=str(task)),
            candidate=candidate,
            agent=None,
            gateway=gateway,
            runner_limits=runner_limits,
            model_dump=lambda: {
                "mode": "real_bundle",
                "bundle": str(bundle),
                "hardware_config": str(hardware_config),
            },
        )
        super().__init__(trial, output)
        self._real_bundle = True
        self._structured_bundle = True
        self.hardware_config, self.hardware_sha256 = (
            Path(hardware_config),
            hardware_sha256,
        )
        self.model_contract, self.runtime_config = Path(model_contract), Path(
            runtime_config
        )
        self.bundle, self.catalog_path = Path(bundle), Path(catalog)
        self.real_input_contract, self.real_input_sha256 = (
            Path(real_input_contract),
            real_input_sha256,
        )
        self.feature_provider, self.feature_provider_sha256 = (
            Path(feature_provider),
            feature_provider_sha256,
        )
        self.kinematics_calibration = (
            Path(kinematics_calibration) if kinematics_calibration else None
        )
        self.zeva_host, self.zeva_port = zeva_host, zeva_port
        self.grasp_config = Path(grasp_config) if grasp_config else None
        self.grasp_config_sha256 = grasp_config_sha256

    def preflight(self):
        from robots.arx.gateway.real_config import (
            load_real_hardware_config,
            validate_real_hardware_config,
        )
        from robots.arx.deployment.real_input import _catalog_tools
        from robots.arx.contracts import load_task_manifest

        for path in (
            self.hardware_config,
            self.model_contract,
            self.runtime_config,
            self.bundle,
            self.catalog_path,
            self.real_input_contract,
            self.feature_provider,
            Path(self.trial.environment.task),
            Path(self.trial.gateway.python),
        ):
            path.resolve(strict=True)
        config = load_real_hardware_config(self.hardware_config, self.hardware_sha256)
        validate_real_hardware_config(
            config, Path(self.trial.environment.task), self.model_contract
        )
        task = load_task_manifest(Path(self.trial.environment.task))
        limits = RuntimeLimits.model_validate_json(self.runtime_config.read_text())
        if limits.max_steps > task.max_steps:
            raise ValueError("gateway max_steps exceeds task limit")
        if self.trial.runner_limits.heartbeat_interval_s * 3 >= limits.lease_timeout_s:
            raise ValueError("heartbeat interval exceeds gateway lease")
        if file_sha256(self.real_input_contract) != self.real_input_sha256:
            raise ValueError("real input contract SHA-256 mismatch")
        contract = RealInputContract.model_validate_json(
            self.real_input_contract.read_text()
        )
        if file_sha256(self.bundle) != contract.candidate_file_sha256:
            raise ValueError("bundle file SHA-256 mismatch")
        if file_sha256(self.feature_provider) != self.feature_provider_sha256:
            raise ValueError("feature provider SHA-256 mismatch")
        bundle, _ = _load_bundle(self.bundle)
        if bundle.sha256 != contract.candidate_sha256:
            raise ValueError("bundle semantic SHA-256 mismatch")
        catalog = json.loads(self.catalog_path.read_text())
        _catalog_tools(catalog, contract.tool_catalog_sha256)
        self.bundle_programs = compile_programs(
            bundle,
            max_tool_calls=contract.max_recovery_tool_calls,
            max_physical_steps=limits.max_steps,
            nominal_chunk_steps=task.execution_steps,
        )
        uses_grasp = any(
            call.tool in {"arx.propose_grasp", "arx.review_grasp", "arx.execute_grasp"}
            for p in self.bundle_programs.values()
            for call in p.calls
        )
        if uses_grasp and self.grasp_config is None:
            raise ValueError("grasp recovery requires a frozen grasp configuration")
        if self.grasp_config:
            from robots.arx.gateway.grasp_contracts import GraspRecoveryConfig

            if file_sha256(self.grasp_config) != self.grasp_config_sha256:
                raise ValueError("grasp configuration SHA mismatch")
            grasp_settings = GraspRecoveryConfig.model_validate_json(
                self.grasp_config.read_text()
            )
            if (
                grasp_settings.learned_pregrasp_commissioning
                and not grasp_settings.learned_gripper_transfer_verified
            ):
                raise ValueError(
                    "unverified learned transfer requires the pregrasp commissioning harness"
                )
            grasp_settings.validate_execution_phases(
                call.arguments.get("phase", "pregrasp")
                for program in self.bundle_programs.values()
                for call in program.calls
                if call.tool == "arx.execute_grasp"
            )
            if uses_grasp and not grasp_settings.motion_enabled:
                raise ValueError("grasp motion is disabled in the frozen configuration")
        if (
            any(
                call.tool == "arx.move_eef"
                for p in self.bundle_programs.values()
                for call in p.calls
            )
            and self.kinematics_calibration is None
        ):
            raise ValueError("EEF recovery requires kinematics calibration")
        if self.kinematics_calibration:
            self.kinematics_calibration.resolve(strict=True)
        self.trial.candidate.package_sha256 = bundle.sha256
        self.trial.candidate.catalog_sha256 = contract.tool_catalog_sha256
        self.gateway_limits = limits
        self._save("private/trial.json", self.trial.model_dump())
        self._save(
            "identities.json",
            {
                "bundle": bundle.sha256,
                "real_input": self.real_input_sha256,
                "hardware": self.hardware_sha256,
                "feature_provider": self.feature_provider_sha256,
                "catalog": contract.tool_catalog_sha256,
                "task": file_sha256(Path(self.trial.environment.task)),
                "model_contract": file_sha256(self.model_contract),
                "runtime_limits": file_sha256(self.runtime_config),
                "backend_implementation": file_sha256(
                    Path(__file__).resolve().parents[1] / "gateway/real_backend.py"
                ),
                "grasp_config": self.grasp_config_sha256,
                "grasp_implementation": (
                    file_sha256(
                        Path(__file__).resolve().parents[1]
                        / "gateway/grasp_recovery.py"
                    )
                    if uses_grasp
                    else None
                ),
                "grasp_sources": (
                    {
                        name: file_sha256(Path(__file__).resolve().parents[3] / name)
                        for name in (
                            "robots/manipulation/grasp_proposals.py",
                            "robots/arx/deployment/picktube_grasp_observer.py",
                            "robots/arx/gateway/grasp_contracts.py",
                        )
                    }
                    if self.grasp_config
                    else None
                ),
            },
        )

    def start(self):
        t = self.trial
        args = [
            t.gateway.python,
            "-m",
            "scripts.deployment.serve_arx_real_gateway",
            "--hardware-config",
            str(self.hardware_config),
            "--expected-hardware-sha256",
            self.hardware_sha256,
            "--task",
            t.environment.task,
            "--model-contract",
            str(self.model_contract),
            "--runtime-config",
            str(self.runtime_config),
            "--output",
            str(self.output / "private/gateway"),
            "--bundle",
            str(self.bundle),
            "--tool-catalog",
            str(self.catalog_path),
            "--real-input-contract",
            str(self.real_input_contract),
            "--expected-real-input-sha256",
            self.real_input_sha256,
            "--feature-provider",
            str(self.feature_provider),
            "--expected-feature-provider-sha256",
            self.feature_provider_sha256,
            "--zeva-host",
            self.zeva_host,
            "--zeva-port",
            str(self.zeva_port),
            "--listen-host",
            t.gateway.host,
            "--listen-port",
            str(t.gateway.port),
        ]
        if self.kinematics_calibration:
            args += ["--kinematics-calibration", str(self.kinematics_calibration)]
        if self.grasp_config:
            args += [
                "--grasp-config",
                str(self.grasp_config),
                "--expected-grasp-config-sha256",
                self.grasp_config_sha256,
            ]
        log = (self.output / "private/gateway-process.log").open("wb")
        child_env = os.environ.copy()
        for key in list(child_env):
            if key.endswith(
                ("API_KEY", "ACCESS_TOKEN", "SECRET_KEY")
            ) or key.startswith("ZETTA_API_"):
                child_env.pop(key, None)
        self.process = subprocess.Popen(
            args,
            cwd=Path(__file__).resolve().parents[3],
            env=child_env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        log.close()
        self._connect_gateway()
