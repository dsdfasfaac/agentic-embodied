"""Real input preflight uses the existing structured sample bundle."""

import json
from pathlib import Path

import pytest

from robots.arx.deployment.real_input import (
    LiveCapabilities,
    RealInputContract,
    preflight_real_bundle,
)
from zetta.evolution.jsonio import canonical_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/arx_privileged_test_20260929_061734/campaign"
BUNDLE = CAMPAIGN / "bundle.json"
CATALOG = CAMPAIGN / "tool-catalog.json"
TASK = ROOT / "robots/arx/manifests/pickup_test_tube.yaml"
MODEL = ROOT / "robots/arx/manifests/task7_model_a.yaml"


@pytest.fixture
def frozen_input(tmp_path):
    bundle = json.loads(BUNDLE.read_text())
    catalog = json.loads(CATALOG.read_text())
    model = json.loads(MODEL.read_text())
    cameras = [
        {
            **camera,
            "device_id": "real-" + camera["name"],
            "calibration_sha256": "a" * 64,
        }
        for camera in model["cameras"]
    ]
    joints = ["joint_" + str(i) for i in range(14)]
    features = [
        {
            "name": "privileged.interaction.gripper_closed",
            "provider_id": "test-gripper-observer",
            "provider_sha256": "b" * 64,
            "source_kind": "joint_feedback",
            "source_ids": ["joint_13"],
            "scalar_type": "boolean",
            "units": "boolean",
            "max_age_ms": 100,
        },
        {
            "name": "privileged.selected.target_gripper_distance_m",
            "provider_id": "test-target-distance-observer",
            "provider_sha256": "c" * 64,
            "source_kind": "fused",
            "source_ids": ["front_rgb", "joint_13"],
            "scalar_type": "number",
            "units": "m",
            "max_age_ms": 100,
        },
    ]
    contract = {
        "schema_version": "arx.real.input.v1",
        "candidate_sha256": canonical_sha256(bundle),
        "candidate_file_sha256": file_sha256(BUNDLE),
        "task_manifest_sha256": file_sha256(TASK),
        "task_id": 0,
        "task_name": "pickup_test_tube",
        "model_contract_sha256": file_sha256(MODEL),
        "tool_catalog_sha256": catalog["catalog_sha256"],
        "cameras": cameras,
        "joint_channels": joints,
        "feature_sources": features,
        "max_critic_history_steps": 16,
        "max_critic_cooldown_steps": 16,
        "max_recovery_tool_calls": 8,
    }
    live = {
        "schema_version": "arx.real.capabilities.v1",
        "robot_id": "test-robot",
        "cameras": cameras,
        "joint_channels": joints,
        "feature_sources": features,
        "tool_catalog_sha256": catalog["catalog_sha256"],
    }

    def run(*, contract_update=None, live_update=None, bundle_update=None):
        current_contract = json.loads(json.dumps(contract))
        current_live = json.loads(json.dumps(live))
        if contract_update:
            contract_update(current_contract)
        if live_update:
            live_update(current_live)
        contract_path = tmp_path / "real-input.json"
        bundle_path = BUNDLE
        if bundle_update:
            current_bundle = json.loads(json.dumps(bundle))
            bundle_update(current_bundle)
            bundle_path = tmp_path / "bundle.json"
            bundle_path.write_text(json.dumps(current_bundle))
            current_contract["candidate_sha256"] = canonical_sha256(current_bundle)
            current_contract["candidate_file_sha256"] = file_sha256(bundle_path)
        RealInputContract.model_validate(current_contract)
        contract_path.write_text(json.dumps(current_contract))
        return preflight_real_bundle(
            bundle_path=bundle_path,
            task_manifest_path=TASK,
            model_contract_path=MODEL,
            tool_catalog_path=CATALOG,
            real_contract_path=contract_path,
            live_capabilities=LiveCapabilities.model_validate(current_live),
            expected_real_contract_sha256=file_sha256(contract_path),
        )

    return run


def test_sample_bundle_accepts_attested_real_sources_and_plans_two_eef_calls(frozen_input):
    report = frozen_input()
    assert report["eligible"]
    assert report["recovery_plans"][0]["tool_calls"] == 5
    assert report["recovery_plans"][0]["steps"][1] == {
        "tool": "arx.move_eef",
        "tool_calls": 2,
    }


def test_privileged_feature_without_live_provider_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="does not attest feature"):
        frozen_input(live_update=lambda value: value["feature_sources"].pop())


def test_privileged_feature_without_real_source_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="no matching real camera source"):
        frozen_input(
            contract_update=lambda value: value["feature_sources"][1].update(
                source_kind="camera_rgb", source_ids=["mujoco_target_pose"]
            ),
            live_update=lambda value: value["feature_sources"][1].update(
                source_kind="camera_rgb", source_ids=["mujoco_target_pose"]
            ),
        )


def test_rgbd_critic_source_requires_live_metric_depth(frozen_input):
    def rgbd_feature(value):
        value["feature_sources"][1].update(
            source_kind="rgbd_fused",
            source_ids=["front_rgb", "front_depth_mm", "joint_13"],
        )

    with pytest.raises(ValueError, match="RGBD feature needs RGB, metric depth"):
        frozen_input(contract_update=rgbd_feature, live_update=rgbd_feature)

    def declare_depth(value):
        rgbd_feature(value)
        value["depth_cameras"] = ["front_depth_mm"]

    assert frozen_input(contract_update=declare_depth, live_update=declare_depth)["eligible"]
    with pytest.raises(ValueError, match="live aligned metric depth"):
        frozen_input(contract_update=declare_depth, live_update=rgbd_feature)


def test_camera_calibration_mismatch_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="live camera"):
        frozen_input(
            live_update=lambda value: value["cameras"][0].update(
                calibration_sha256="d" * 64
            )
        )


def test_bundle_sha_mismatch_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="CandidateBundle file SHA"):
        frozen_input(
            contract_update=lambda value: value.update(
                candidate_file_sha256="d" * 64
            )
        )


def test_invalid_recovery_parameter_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="invalid recovery parameters"):
        frozen_input(
            bundle_update=lambda value: value["recovery_rules"][0]["steps"][0][
                "parameters"
            ].update(opening=2.0)
        )


def test_task_identity_mismatch_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="task identity"):
        frozen_input(
            contract_update=lambda value: value.update(task_name="wrong_task")
        )


def test_tool_catalog_mismatch_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="live tool catalog"):
        frozen_input(
            live_update=lambda value: value.update(tool_catalog_sha256="d" * 64)
        )


def test_critic_feature_without_mapping_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="critic feature has no real observation source"):
        frozen_input(
            bundle_update=lambda value: value["critic_rules"][0].update(
                feature="privileged.mujoco.target_pose"
            )
        )


def test_unknown_recovery_tool_is_rejected(frozen_input):
    with pytest.raises(ValueError, match="absent from real catalog"):
        frozen_input(
            bundle_update=lambda value: value["recovery_rules"][0]["steps"][0].update(
                tool="arx.unknown"
            )
        )
