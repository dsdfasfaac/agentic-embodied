# Copyright (c) 2026 Zetta Contributors
"""Tests for the fail-closed ARX Agentic deployment configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robots.arx.task_adapter import ArxAgenticTaskConfig
from rollout_runtime.config.schema import load_config

_ROOT = Path(__file__).resolve().parents[1]
_MANIFESTS = _ROOT / "robots/arx/manifests"
_MAPPING = _ROOT.parent / "Zeva_arx/assets/ac_one/ac_one_14d_mapping.json"


def _config(tmp_path: Path, **overrides) -> ArxAgenticTaskConfig:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    digest = "a" * 64
    (prepared / "metadata.json").write_text(json.dumps({
        "bundle_digest": digest,
        "robot_source_archive_sha256": "d238e12107afbef5d0eca4cc36d1bf37ffdcef27e27a28a5fd30f35265e53ea0",
    }))
    for name in ("model.mjb", "reset_state.npz", "composition_report.json"):
        (prepared / name).write_bytes(name.encode())
    values = {
        "prepared_scene_bundle": prepared.resolve(),
        "prepared_bundle_digest": digest,
        "mapping_path": _MAPPING.resolve(),
        "task_manifest": (_MANIFESTS / "pickup_test_tube.yaml").resolve(),
        "model_contract": (_MANIFESTS / "task7_model.yaml").resolve(),
        "camera_names": {"front_rgb": "front", "left_rgb": "left", "right_rgb": "right"},
        "output_dir": (tmp_path / "output").resolve(),
    }
    values.update(overrides)
    return ArxAgenticTaskConfig(**values)


@pytest.mark.skipif(not _MAPPING.is_file(), reason="../Zeva_arx checkout required")
def test_runtime_config_pins_env_policy_and_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime = config.runtime_config()
    assert runtime["env_config"]["provider"] == "arx_ac_one"
    assert runtime["env_config"]["clip_actions"] is False
    assert runtime["rollout_worker"]["policy_backend"] == "cosmos3_edge_arx_remote"
    assert runtime["rollout_worker"]["policy_config"]["port"] == 5581
    provenance = config.provenance()
    assert provenance["prepared_bundle_digest"] == "a" * 64
    assert provenance["robot_source_archive_sha256"].startswith("d238")


@pytest.mark.skipif(not _MAPPING.is_file(), reason="../Zeva_arx checkout required")
def test_config_rejects_digest_and_camera_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        _config(tmp_path, prepared_bundle_digest="b" * 64)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="camera_names"):
        _config(other, camera_names={"front_rgb": "same", "left_rgb": "same", "right_rgb": "right"})


def test_arx_runtime_preset_is_structurally_valid() -> None:
    config = load_config("arx_task7_mujoco")
    assert config.env_config["provider"] == "arx_ac_one"
    assert config.rollout_worker.policy_backend == "cosmos3_edge_arx_remote"
    assert config.rollout_worker.scheduler.max_batch_size == 1
