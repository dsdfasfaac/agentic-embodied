# Copyright (c) 2026 Zetta Contributors
"""Offline Real2Sim + ARX MuJoCo scene preparation.

This module deliberately has no import-time MuJoCo dependency.  Preparation is
run in the zetta-mujoco conda environment and produces immutable episode input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from robots.arx.camera_config import ArxCameraConfig, load_camera_config
from robots.arx.scene_bundle import Real2SimSceneBundle, load_scene_bundle

_ROBOT_SECTIONS = ("equality", "contact", "actuator", "sensor", "tendon")


def preserve_robot_physics(scene: ET.Element, robot: ET.Element) -> None:
    """Materialize robot defaults before extracting its body/actuator fragments.

    Defaults must not be merged into the scene's main class: doing so would
    change the reconstructed objects too. Explicit attributes also survive
    later extraction and retain per-joint overrides (e.g. finger damping).
    The robot integrator is a deliberate global override; all other scene
    options, including gravity and timestep, remain scene-owned.
    """
    classes: dict[str, dict[str, dict[str, str]]] = {"main": {}}

    def collect(node: ET.Element, inherited: dict[str, dict[str, str]]) -> None:
        values = copy.deepcopy(inherited)
        for child in node:
            if child.tag != "default":
                values.setdefault(child.tag, {}).update(child.attrib)
        classes[node.get("class", "main")] = values
        for child in node.findall("default"):
            collect(child, values)

    defaults = robot.find("default")
    if defaults is not None:
        collect(defaults, {})

    def materialize(node: ET.Element, inherited_class: str = "main", parent_tag: str = "") -> None:
        if node.tag == "default":
            return
        name = node.get("class", inherited_class)
        if name not in classes:
            raise ValueError(f"unknown robot default class: {name}")
        default_tag = node.tag
        if parent_tag == "equality" and node.tag in {"joint", "tendon"}:
            default_tag = "equality"
        for key, value in classes[name].get(default_tag, {}).items():
            node.attrib.setdefault(key, value)
        node.attrib.pop("class", None)
        child_class = node.attrib.pop("childclass", inherited_class)
        for child in node:
            materialize(child, child_class, node.tag)

    materialize(robot)
    robot_option = robot.find("option")
    if robot_option is not None and robot_option.get("integrator") is not None:
        option = scene.find("option")
        if option is None:
            option = ET.SubElement(scene, "option")
        option.set("integrator", robot_option.get("integrator"))


def _compiler_directory(root: ET.Element, attribute: str, xml_path: Path) -> Path:
    compiler = root.find("compiler")
    value = compiler.get(attribute) if compiler is not None else None
    return (xml_path.parent / value).resolve() if value else xml_path.parent.resolve()


def _copy_asset_files(root: ET.Element, xml_path: Path, output: Path, prefix: str) -> None:
    mesh_root = _compiler_directory(root, "meshdir", xml_path)
    texture_root = _compiler_directory(root, "texturedir", xml_path)
    for element in root.findall("./asset/*"):
        value = element.get("file")
        if not value:
            continue
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = (texture_root if element.tag == "texture" else mesh_root) / source
        source = source.resolve(strict=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        relative = Path("assets") / prefix / f"{digest}_{source.name}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        element.set("file", relative.as_posix())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)


def _names(root: ET.Element) -> set[str]:
    return {value for element in root.iter() if (value := element.get("name"))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _robot_fragment(robot: ET.Element) -> tuple[list[ET.Element], ET.Element, list[ET.Element]]:
    asset = robot.find("asset")
    world = robot.find("worldbody")
    base = world.find("./body[@name='base_link']") if world is not None else None
    if asset is None or base is None:
        raise ValueError("ARX XML must contain asset and worldbody/body[@name='base_link']")
    sections = [section for tag in _ROBOT_SECTIONS if (section := robot.find(tag)) is not None]
    return list(asset), base, sections


def _install_policy_cameras(scene_root: ET.Element, config: ArxCameraConfig) -> None:
    config.require_complete()
    world = scene_root.find("worldbody")
    if world is None:
        raise ValueError("scene XML has no worldbody")
    for camera in config.cameras:
        parent = world if camera.parent_body == "world" else world.find(f".//body[@name='{camera.parent_body}']")
        if parent is None:
            raise ValueError(f"ARX camera parent body is missing: {camera.parent_body}")
        assert camera.pose_xyz_m is not None and camera.pose_quat_wxyz is not None
        ET.SubElement(parent, "camera", {
            "name": camera.semantic_name,
            "pos": " ".join(str(value) for value in camera.pose_xyz_m),
            "quat": " ".join(str(value) for value in camera.pose_quat_wxyz),
            "fovy": str(config.vertical_fovy_deg),
        })
    for camera in config.diagnostic_cameras:
        parent = world if camera.parent_body == "world" else world.find(f".//body[@name='{camera.parent_body}']")
        if parent is None:
            raise ValueError(f"diagnostic camera parent body is missing: {camera.parent_body}")
        ET.SubElement(parent, "camera", {
            "name": camera.name,
            "pos": " ".join(str(value) for value in camera.pose_xyz_m),
            "xyaxes": " ".join(str(value) for value in camera.xyaxes),
            "fovy": str(camera.fovy_deg),
        })


def compose_scene(
    bundle: Real2SimSceneBundle,
    output_dir: str | Path,
    camera_config: str | Path | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    robot_base_quat: tuple[float, float, float, float] | None = None,
    robot_rgba: tuple[float, float, float, float] | None = None,
    robot_material: str | None = None,
) -> Path:
    """Validate and compose XML/assets. Returns the prepared scene XML path."""
    bundle_digest = bundle.validate_files()
    if bundle.composition.frame_transform not in {"identity", "identity_v1"}:
        raise ValueError("only the audited identity frame_transform is currently supported")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_path = bundle.artifact_path(bundle.artifacts.scene_xml)
    robot_path = Path(bundle.composition.robot_xml).expanduser().resolve(strict=True)
    scene_tree = ET.parse(scene_path)
    robot_tree = ET.parse(robot_path)
    scene_root, robot_root = scene_tree.getroot(), robot_tree.getroot()
    preserve_robot_physics(scene_root, robot_root)
    _copy_asset_files(scene_root, scene_path, output, "real2sim")
    _copy_asset_files(robot_root, robot_path, output, "arx")

    robot_assets, robot_base, robot_sections = _robot_fragment(robot_root)
    scene_names = _names(scene_root)
    robot_names = _names(robot_base)
    for item in (*robot_assets, *robot_sections):
        robot_names.update(_names(item))
    collisions = sorted(scene_names & robot_names)
    if collisions:
        raise ValueError(f"Real2Sim and ARX names collide: {collisions}")

    asset = scene_root.find("asset")
    world = scene_root.find("worldbody")
    if asset is None or world is None:
        raise ValueError("Real2Sim XML must contain asset and worldbody")
    for item in robot_assets:
        asset.append(copy.deepcopy(item))
    composed_robot_base = copy.deepcopy(robot_base)
    if robot_base_pos is not None:
        composed_robot_base.set("pos", " ".join(str(value) for value in robot_base_pos))
    if robot_base_quat is not None:
        composed_robot_base.set("quat", " ".join(str(value) for value in robot_base_quat))
    world.append(composed_robot_base)
    if robot_rgba is not None:
        rgba = " ".join(str(value) for value in robot_rgba)
        for material_name in ("ac_one_body", "ac_one_gripper"):
            material = asset.find(f"./material[@name='{material_name}']")
            if material is None:
                raise ValueError(f"robot material is missing: {material_name}")
            material.set("rgba", rgba)
            if robot_material == "dark_silver_metallic":
                material.set("specular", "0.8")
                material.set("shininess", "0.9")
                material.set("reflectance", "0.15")
    for section in robot_sections:
        scene_root.append(copy.deepcopy(section))
    if camera_config is not None:
        _install_policy_cameras(scene_root, load_camera_config(camera_config))
    scene_root.set("model", f"zetta_{bundle.scene_id}")
    output_xml = output / "scene.xml"
    ET.indent(scene_tree, space="  ")
    scene_tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    _build_reset_artifacts(bundle, output_xml, output)
    assembly = _read_json(bundle.artifact_path(bundle.artifacts.assembly_report))
    logical_body_map = {
        str(item["logical_id"]): str(item["root_body"])
        for item in assembly.get("objects", [])
        if item.get("logical_id") and item.get("root_body")
    }
    metadata = {
        "schema_version": "zetta_prepared_scene_v1",
        "scene_id": bundle.scene_id,
        "task_name": bundle.task_name,
        "bundle_digest": bundle_digest,
        "scene_xml": output_xml.name,
        "camera_rig": bundle.composition.cameras,
        "robot_source_archive": bundle.composition.robot_source_archive,
        "robot_source_archive_sha256": bundle.composition.robot_source_archive_sha256,
        "reset_source": bundle.artifacts.settled_state.path,
        "logical_body_map": logical_body_map,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_xml


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _build_reset_artifacts(
    bundle: Real2SimSceneBundle, scene_xml: Path, output: Path
) -> None:
    """Prove prefix compatibility, then transfer the settled scene state."""
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by CLI deployment
        raise RuntimeError("scene preparation requires the zetta-mujoco conda env") from exc

    source_mjb = bundle.artifact_path(bundle.artifacts.model_mjb)
    source_xml = bundle.artifact_path(bundle.artifacts.scene_xml)
    try:
        source_model = mujoco.MjModel.from_binary_path(str(source_mjb))
        source_model_source = "real2sim_mjb"
    except Exception as exc:
        # MJB files are MuJoCo-version-specific. Real2Sim XML is canonical;
        # recompile it with the active Zetta MuJoCo when the binary ABI differs.
        source_model = mujoco.MjModel.from_xml_path(str(source_xml))
        source_model_source = "real2sim_xml_recompiled"
    composed_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    with np.load(bundle.artifact_path(bundle.artifacts.settled_state)) as settled:
        if set(settled.files) != {"qpos", "qvel", "time"}:
            raise ValueError("settled_state must contain exactly qpos, qvel, and time")
        source_qpos = np.asarray(settled["qpos"], dtype=np.float64)
        source_qvel = np.asarray(settled["qvel"], dtype=np.float64)
        source_time = float(settled["time"])
    if source_qpos.shape != (source_model.nq,) or source_qvel.shape != (source_model.nv,):
        raise ValueError("settled_state dimensions do not match the Real2Sim model")
    if composed_model.nq < source_model.nq or composed_model.nv < source_model.nv:
        raise ValueError("composed model cannot contain the complete Real2Sim state")
    prefix_fields = ("jnt_type", "jnt_qposadr", "jnt_dofadr")
    for field in prefix_fields:
        source_value = np.asarray(getattr(source_model, field))
        composed_value = np.asarray(getattr(composed_model, field))[: source_model.njnt]
        if not np.array_equal(source_value, composed_value):
            raise ValueError(f"composition changed Real2Sim joint layout: {field}")

    data = mujoco.MjData(composed_model)
    data.qpos[: source_model.nq] = source_qpos
    data.qvel[: source_model.nv] = source_qvel
    data.time = source_time
    mujoco.mj_forward(composed_model, data)
    np.savez(output / "reset_state.npz", qpos=data.qpos, qvel=data.qvel, time=data.time)
    mujoco.mj_saveModel(composed_model, str(output / "model.mjb"))
    report = {
        "schema_version": "zetta_scene_composition_report_v1",
        "state_mapping": "verified_unchanged_real2sim_prefix",
        "source": {"nq": source_model.nq, "nv": source_model.nv, "njnt": source_model.njnt},
        "composed": {"nq": composed_model.nq, "nv": composed_model.nv, "njnt": composed_model.njnt},
        "source_model": source_model_source,
        "source_mjb_sha256": _sha256(source_mjb),
        "recompiled_for_mujoco": source_model_source != "real2sim_mjb",
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
    }
    (output / "composition_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", dest="bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--robot-base-pos", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--robot-base-quat", nargs=4, type=float, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--robot-rgba", nargs=4, type=float, metavar=("R", "G", "B", "A"))
    parser.add_argument("--robot-material", choices=("dark_silver_metallic",))
    args = parser.parse_args()
    print(compose_scene(
        load_scene_bundle(args.bundle), args.output, args.camera_config,
        tuple(args.robot_base_pos) if args.robot_base_pos else None,
        tuple(args.robot_base_quat) if args.robot_base_quat else None,
        tuple(args.robot_rgba) if args.robot_rgba else None,
        args.robot_material,
    ))


if __name__ == "__main__":
    main()
