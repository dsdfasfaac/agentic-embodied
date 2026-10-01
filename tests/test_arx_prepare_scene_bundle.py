# Copyright (c) 2026 Zetta Contributors
"""Tests for deterministic, self-contained Real2Sim + ARX composition."""

from __future__ import annotations

import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from robots.arx.prepare_scene_bundle import compose_scene, preserve_robot_physics
from robots.arx.scene_bundle import Real2SimSceneBundle


def _put(path: Path, data: bytes | dict) -> str:
    value = json.dumps(data).encode() if isinstance(data, dict) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path: Path) -> Real2SimSceneBundle:
    run_id = "run"
    mesh = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    _put(tmp_path / "source/mesh.obj", mesh)
    scene_xml = b"""<mujoco><compiler meshdir='.'/><asset><mesh name='object_mesh' file='mesh.obj'/></asset><worldbody><body name='tube'><freejoint name='tube_free'/><geom type='sphere' size='.01'/></body></worldbody></mujoco>"""
    robot_xml = b"""<mujoco><compiler meshdir='.'/><asset><mesh name='robot_mesh' file='mesh.obj'/></asset><worldbody><body name='base_link'><geom type='sphere' size='.01'/><body><joint name='left_joint1'/><geom type='sphere' size='.01' mass='.1'/></body></body></worldbody><actuator><position name='left_joint1_position' joint='left_joint1'/></actuator></mujoco>"""
    refs: dict[str, dict[str, str]] = {}
    values: dict[str, tuple[str, bytes | dict]] = {
        "active_layout": ("agent/active_layout.json", {"run_id": run_id, "objects": [{"id": "tube"}]}),
        "assembly_report": ("scene/assembly_report.json", {"run_id": run_id, "objects": [{"logical_id": "tube"}]}),
        "final_scene": ("output/final_scene.json", {"schema_version": "1.0", "run_id": run_id, "backend": "mujoco", "status": "succeeded", "scene_xml": "scene.xml", "compiled_model": "model.mjb", "state": "settled_state.npz"}),
        "scene_xml": ("source/scene.xml", scene_xml),
    }
    for name, (relative, value) in values.items():
        refs[name] = {"path": relative, "sha256": _put(tmp_path / relative, value)}
    source_model = mujoco.MjModel.from_xml_path(str(tmp_path / "source/scene.xml"))
    model_path = tmp_path / "output/model.mjb"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(source_model, str(model_path))
    refs["model_mjb"] = {"path": "output/model.mjb", "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    state = io.BytesIO()
    np.savez(state, qpos=np.zeros(source_model.nq), qvel=np.zeros(source_model.nv), time=np.array(0.0))
    refs["settled_state"] = {"path": "output/settled_state.npz", "sha256": _put(tmp_path / "output/settled_state.npz", state.getvalue())}
    robot = tmp_path / "robot/robot.xml"
    archive = tmp_path / "robot/AC one.7z"
    archive_hash = _put(archive, b"authoritative archive")
    _put(tmp_path / "robot/mesh.obj", mesh)
    robot_hash = _put(robot, robot_xml)
    mapping = tmp_path / "robot/mapping.json"
    mapping_hash = _put(mapping, b"{}")
    return Real2SimSceneBundle.from_mapping({
        "schema_version": "zetta_real2sim_scene_bundle_v1", "scene_id": "scene_a", "task_name": "task",
        "source": {"real2sim_commit": "abc", "run_id": run_id, "kind": "attempt", "index": 0, "retry": 0, "bundle_root": str(tmp_path)},
        "artifacts": refs,
        "objects": {"required_ids": ["tube"], "target_ids": ["tube"]},
        "composition": {"frame_transform": "identity_v1", "robot_source_archive": str(archive), "robot_source_archive_sha256": archive_hash, "robot_xml": str(robot), "robot_xml_sha256": robot_hash, "mapping": str(mapping), "mapping_sha256": mapping_hash, "cameras": "rig_v1"},
        "reset": {"source": "settled_state", "settle_after_composition_steps": 0},
    })


def test_composer_rewrites_assets_and_compiles(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    xml = compose_scene(_fixture(tmp_path), output)
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.njnt == 2
    assert model.nu == 1
    text = xml.read_text()
    assert "assets/real2sim/" in text
    assert "assets/arx/" in text
    assert json.loads((output / "metadata.json").read_text())["scene_id"] == "scene_a"


def test_composer_rejects_mutated_source(tmp_path: Path) -> None:
    bundle = _fixture(tmp_path)
    source = bundle.artifact_path(bundle.artifacts.scene_xml)
    data = source.read_text().replace("name='tube'", "name='base_link'")
    source.write_text(data)
    # Hash validation catches source mutation before unsafe composition.
    try:
        compose_scene(bundle, tmp_path / "prepared")
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("mutated source must be rejected")


def test_robot_defaults_survive_extraction_without_affecting_objects():
    robot = ET.fromstring('''<mujoco><option integrator="implicitfast"/>
      <default><joint damping="1.5" armature="0.02"/>
        <default class="finger"><joint damping="8"/></default></default>
      <worldbody><body name="base_link"><body><joint name="arm"/>
        <geom size="0.01" type="sphere" mass="0.1"/>
        <body childclass="finger"><joint name="finger" damping="9"/>
          <geom size="0.01" type="sphere" mass="0.1"/></body>
      </body></body></worldbody></mujoco>''')
    scene = ET.fromstring('''<mujoco><option timestep="0.003" gravity="0 0 -9.81"/>
      <worldbody><body><joint name="object"/><geom size="0.01" type="sphere" mass="0.1"/></body></worldbody></mujoco>''')
    preserve_robot_physics(scene, robot)
    scene.find('worldbody').append(robot.find('./worldbody/body'))
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.timestep == 0.003
    for name, damping, armature in [('object', 0, 0), ('arm', 1.5, .02), ('finger', 9, .02)]:
        i = model.jnt_dofadr[model.joint(name).id]
        assert model.dof_damping[i] == damping
        assert model.dof_armature[i] == armature
