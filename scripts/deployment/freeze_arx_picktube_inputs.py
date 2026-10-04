#!/usr/bin/env python3
"""Freeze the sample PickTube bundle's static real-input/catalog contract.

This does not attest live arm status or approve a hardware configuration.
The real gateway still requires its independent --check-config gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.arx.contracts import load_model_contract, load_task_manifest
from robots.arx.deployment.picktube_rgbd_provider import PickTubeRgbdProvider
from robots.arx.deployment.real_input import (
    RealInputContract, _catalog_tools, _check_recoveries, _check_rules, _load_bundle,
)
from robots.arx.gateway.real_config import REAL_JOINT_CHANNELS
from robots.arx.gateway.tools import default_registry
from zetta.evolution.jsonio import file_sha256


class _Unavailable:
    def prepare(self, *args):
        raise RuntimeError("catalog export does not execute tools")

    def inspect(self, *args):
        raise RuntimeError("catalog export does not execute tools")


def freeze(bundle_path: Path, output_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    task_path = root / "robots/arx/manifests/pickup_test_tube.yaml"
    model_path = root / "robots/arx/manifests/task7_model_a.yaml"
    provider_path = root / "robots/arx/deployment/picktube_rgbd_provider.py"
    calibration_root = root / "robots/arx/manifests/real"
    bundle, _ = _load_bundle(bundle_path)
    task = load_task_manifest(task_path)
    model = load_model_contract(model_path)
    stub = _Unavailable()
    catalog = default_registry(zeva=stub, gripper=stub, eef=stub,
                               reentry=stub).describe()
    cameras = []
    for spec, serial in zip(model.cameras,
                            ("260422272500", "260422271945", "260422275847")):
        calibration = calibration_root / f"dodo_{spec.name}_d405_intrinsics.json"
        cameras.append({
            "name": spec.name, "device_id": serial,
            "calibration_id": spec.calibration_id,
            "calibration_sha256": file_sha256(calibration),
            "width": spec.width, "height": spec.height,
            "channels": spec.channels, "dtype": spec.dtype,
            "color_order": spec.color_order,
        })
    contract = RealInputContract.model_validate({
        "schema_version": "arx.real.input.v1",
        "candidate_sha256": bundle.sha256,
        "candidate_file_sha256": file_sha256(bundle_path),
        "task_manifest_sha256": file_sha256(task_path),
        "task_id": task.task_id, "task_name": task.name,
        "model_contract_sha256": file_sha256(model_path),
        "tool_catalog_sha256": catalog["catalog_sha256"],
        "cameras": cameras, "depth_cameras": ["front_depth_mm"],
        "joint_channels": list(REAL_JOINT_CHANNELS),
        "feature_sources": PickTubeRgbdProvider().feature_sources(),
        "max_critic_history_steps": 16,
        "max_critic_cooldown_steps": 16,
        "max_recovery_tool_calls": 8,
    })
    _catalog_tools(catalog, contract.tool_catalog_sha256)
    used_features = _check_rules(bundle, contract)
    recovery_plans = _check_recoveries(bundle, contract, catalog)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "tool-catalog.json"
    contract_path = output_dir / "real-input-contract.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    contract_path.write_text(contract.model_dump_json(indent=2) + "\n")
    return {
        "bundle": str(bundle_path),
        "bundle_file_sha256": file_sha256(bundle_path),
        "bundle_semantic_sha256": bundle.sha256,
        "catalog": str(catalog_path),
        "catalog_sha256": catalog["catalog_sha256"],
        "real_input_contract": str(contract_path),
        "real_input_contract_sha256": file_sha256(contract_path),
        "feature_provider_sha256": file_sha256(provider_path),
        "used_features": used_features,
        "recovery_plans": recovery_plans,
        "audit_scope": "static declarations only; live joint feedback and EEF kinematics not attested",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--bundle", type=Path, default=root /
                        "robots/arx/manifests/real/sample_picktube_candidate_bundle.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.bundle, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
