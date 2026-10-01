#!/usr/bin/env python3
"""Render front/side diagnostics of an embedded MuJoCo camera pose."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    z_axis = -forward
    up = np.cross(z_axis, right)
    quat_xyzw = Rotation.from_matrix(np.column_stack((right, up, z_axis))).as_quat()
    return np.roll(quat_xyzw, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--camera", default="front_rgb")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.scene / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(args.scene / "reset_state.npz") as reset:
        data.qpos[:] = reset["qpos"]
        data.qvel[:] = reset["qvel"]
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    origin = data.cam_xpos[camera_id].copy()
    rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    tip = origin - 0.28 * rotation[:, 2]

    tree = ET.parse(args.scene / "scene.xml")
    root = tree.getroot()
    scene_root = args.scene.resolve()
    for element in root.findall("./asset/*"):
        filename = element.get("file")
        if filename and not Path(filename).is_absolute():
            element.set("file", str((scene_root / filename).resolve()))
    world = root.find("worldbody")
    assert world is not None
    # Make the shell translucent so an optical center inside it remains visible.
    asset = root.find("asset")
    assert asset is not None
    for material_name in ("ac_one_body", "ac_one_gripper"):
        material = asset.find(f"./material[@name='{material_name}']")
        if material is not None:
            rgba = material.get("rgba", "0.18 0.20 0.23 1").split()
            rgba[-1] = "0.22"
            material.set("rgba", " ".join(rgba))
    ET.SubElement(world, "geom", name="front_camera_origin_marker", type="sphere",
                  pos=" ".join(map(str, origin)), size="0.035", rgba="1 0.02 0.02 1",
                  contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="front_camera_direction_marker", type="capsule",
                  fromto=" ".join(map(str, np.r_[origin, tip])), size="0.012",
                  rgba="1 0.08 0.02 1", contype="0", conaffinity="0")
    views = {
        "front": (np.array([0.0, 1.65, 1.30]), np.array([0.0, -0.2, 0.95])),
        "side": (np.array([1.65, -0.15, 1.25]), np.array([0.0, -0.25, 0.95])),
    }
    for name, (position, target) in views.items():
        ET.SubElement(world, "camera", name=f"diagnostic_{name}",
                      pos=" ".join(map(str, position)),
                      quat=" ".join(map(str, _look_at(position, target))), fovy="48")
    diagnostic_xml = args.output / "camera_pose_diagnostic.xml"
    tree.write(diagnostic_xml, encoding="utf-8", xml_declaration=True)
    diagnostic = mujoco.MjModel.from_xml_path(str(diagnostic_xml))
    diagnostic_data = mujoco.MjData(diagnostic)
    diagnostic_data.qpos[:] = data.qpos
    diagnostic_data.qvel[:] = data.qvel
    mujoco.mj_forward(diagnostic, diagnostic_data)
    renderer = mujoco.Renderer(diagnostic, height=480, width=640)
    for name in views:
        renderer.update_scene(diagnostic_data, camera=f"diagnostic_{name}")
        iio.imwrite(args.output / f"front_camera_pose_{name}.png", renderer.render())
    renderer.close()
    (args.output / "pose.txt").write_text(
        f"world_position_xyz={origin.tolist()}\nworld_forward={(-rotation[:, 2]).tolist()}\n"
    )


if __name__ == "__main__":
    main()
