# Copyright (c) 2026 Zetta Contributors
"""Tests for pinned ARX model and task contracts."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from robots.arx.contracts import (
    ARX_CAMERA_NAMES,
    ArxModelContract,
    ArxTaskManifest,
    load_model_contract,
    load_task_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "robots/arx/manifests"


def test_task7_model_contract_is_pinned() -> None:
    contract = load_model_contract(MANIFESTS / "task7_model.yaml")
    assert contract.domain_name == "arx-task7-x5"
    assert contract.domain_id == 17
    assert tuple(camera.name for camera in contract.cameras) == ARX_CAMERA_NAMES
    assert contract.action_dim == contract.state_dim == 14
    assert contract.action_horizon == 32
    assert contract.conditioning_fps == 15.0
    assert contract.action_normalization == "raw"


def test_checkpoint_model_a_contract_is_pinned_separately() -> None:
    contract = load_model_contract(MANIFESTS / "task7_model_a.yaml")
    assert contract.domain_name == "arx_task7"
    assert contract.domain_id == 17
    assert contract.action_horizon == 32
    task = load_task_manifest(MANIFESTS / "pickup_test_tube.yaml")
    assert task.execution_steps == 16


def test_pickup_manifest_matches_arx_runtime_contract() -> None:
    task = load_task_manifest(MANIFESTS / "pickup_test_tube.yaml")
    assert task.task_id == 0
    assert task.instruction == "Pick up test tube with the pink label."
    assert task.execution_steps == 16
    assert task.max_steps == 600
    assert task.control.lock_left_arm is True
    assert task.control.zero_left_model_state is True
    assert task.control.gripper_command_offsets == (0.0, 0.9)
    assert task.start_state == (
        0.0322351456,
        0.0062942505,
        0.0024795532,
        -0.0402460098,
        -0.0268936157,
        -0.0192642212,
        -2.98,
        0.0181198120,
        -0.0013351440,
        0.0032424927,
        -0.0310897827,
        -0.0040054321,
        0.0700006485,
        -3.4,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("domain_id", 8, "domain"),
        ("action_dim", 7, "dimension"),
        ("action_horizon", 16, "horizon"),
        ("conditioning_fps", 20, "conditioning"),
        ("action_normalization", "meanstd", "normalization"),
    ],
)
def test_model_contract_rejects_incompatible_values(
    field: str, value: object, match: str
) -> None:
    valid = load_model_contract(MANIFESTS / "task7_model.yaml")
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(valid, **{field: value})


def test_model_contract_rejects_unknown_key() -> None:
    valid = load_model_contract(MANIFESTS / "task7_model.yaml")
    payload = dataclasses.asdict(valid)
    payload["unreviewed_option"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        ArxModelContract.from_mapping(payload)


def test_task_contract_rejects_bad_state_and_scene_allowlist() -> None:
    valid = load_task_manifest(MANIFESTS / "pickup_test_tube.yaml")
    payload = dataclasses.asdict(valid)
    payload["start_state"] = [0.0] * 13
    with pytest.raises(ValueError, match="14 values"):
        ArxTaskManifest.from_mapping(payload)
    payload = dataclasses.asdict(valid)
    payload["starting_scenes"]["default_scene_id"] = "not-allowed"
    with pytest.raises(ValueError, match="allowlisted"):
        ArxTaskManifest.from_mapping(payload)


def test_manifests_use_dependency_free_json_yaml_subset(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text("schema_version: zetta_arx_task_v1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON-compatible YAML"):
        load_task_manifest(path)
