# Copyright (c) 2026 Zetta Contributors
"""Tests for immutable Real2Sim scene bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robots.arx.contracts import ArxStartingScenes
from robots.arx.scene_bundle import Real2SimSceneBundle, select_scene_id


def _write(path: Path, value: bytes | dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _bundle(tmp_path: Path) -> dict:
    run_id = "test_run"
    artifacts = {
        "active_layout": ("agent/active_layout.json", {"schema_version": "1.0", "run_id": run_id, "objects": [{"id": "tube"}]}),
        "assembly_report": ("scene/assembly_report.json", {"schema_version": "1.0", "run_id": run_id, "objects": [{"logical_id": "tube", "root_body": "tube__object"}]}),
        "final_scene": ("output/final_scene.json", {"schema_version": "1.0", "run_id": run_id, "backend": "mujoco", "status": "succeeded", "scene_xml": "scene/scene.xml", "compiled_model": "output/model.mjb", "state": "output/settled_state.npz"}),
        "scene_xml": ("scene/scene.xml", b"<mujoco/>") ,
        "model_mjb": ("output/model.mjb", b"model"),
        "settled_state": ("output/settled_state.npz", b"state"),
    }
    refs = {}
    for name, (relative, value) in artifacts.items():
        refs[name] = {"path": relative, "sha256": _write(tmp_path / relative, value)}
    robot = tmp_path / "external/robot.xml"
    archive = tmp_path / "external/AC one.7z"
    mapping = tmp_path / "external/mapping.json"
    robot_hash = _write(robot, b"robot")
    archive_hash = _write(archive, b"archive")
    mapping_hash = _write(mapping, b"mapping")
    return {
        "schema_version": "zetta_real2sim_scene_bundle_v1",
        "scene_id": "pickup_scene",
        "task_name": "pickup_test_tube",
        "source": {"real2sim_commit": "abc123", "run_id": run_id, "kind": "attempt", "index": 2, "retry": 0, "bundle_root": str(tmp_path)},
        "artifacts": refs,
        "objects": {"required_ids": ["tube"], "target_ids": ["tube"]},
        "composition": {"frame_transform": "identity_v1", "robot_source_archive": str(archive), "robot_source_archive_sha256": archive_hash, "robot_xml": str(robot), "robot_xml_sha256": robot_hash, "mapping": str(mapping), "mapping_sha256": mapping_hash, "cameras": "arx_task7_cameras_v1"},
        "reset": {"source": "settled_state", "settle_after_composition_steps": 0},
    }


def test_bundle_validates_files_and_has_path_independent_digest(tmp_path: Path) -> None:
    payload = _bundle(tmp_path)
    bundle = Real2SimSceneBundle.from_mapping(payload)
    digest = bundle.validate_files()
    assert len(digest) == 64
    assert bundle.validate_files() == digest


def test_bundle_rejects_hash_change(tmp_path: Path) -> None:
    payload = _bundle(tmp_path)
    bundle = Real2SimSceneBundle.from_mapping(payload)
    (tmp_path / "scene/scene.xml").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.validate_files()


def test_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    payload = _bundle(tmp_path)
    payload["artifacts"]["scene_xml"]["path"] = "../scene.xml"
    with pytest.raises(ValueError, match="safe and relative"):
        Real2SimSceneBundle.from_mapping(payload)


def test_bundle_rejects_unsuccessful_real2sim_output(tmp_path: Path) -> None:
    payload = _bundle(tmp_path)
    final_path = tmp_path / "output/final_scene.json"
    final = json.loads(final_path.read_text())
    final["status"] = "failed"
    payload["artifacts"]["final_scene"]["sha256"] = _write(final_path, final)
    bundle = Real2SimSceneBundle.from_mapping(payload)
    with pytest.raises(ValueError, match="successful MuJoCo"):
        bundle.validate_files()


def test_scene_selection_is_allowlisted_and_order_independent() -> None:
    first = ArxStartingScenes("seeded_uniform", "a", ("b", "a"))
    second = ArxStartingScenes("seeded_uniform", "a", ("a", "b"))
    assert select_scene_id(first, seed=17) == select_scene_id(second, seed=17)
    assert select_scene_id(first, seed=17) in {"a", "b"}
    assert select_scene_id(first, seed=17, explicit_scene_id="a") == "a"
    with pytest.raises(ValueError, match="allowlisted"):
        select_scene_id(first, seed=17, explicit_scene_id="other")


def test_explicit_scene_selection_requires_id() -> None:
    scenes = ArxStartingScenes("explicit", "a", ("a",))
    with pytest.raises(ValueError, match="requires scene_id"):
        select_scene_id(scenes, seed=0)
