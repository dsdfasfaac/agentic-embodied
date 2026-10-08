"""Measured tube selection, pinned CAD bounds and exact-pose joint path guards."""

from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from robots.manipulation.target_clouds import upright_tube_mask
from robots.arx.gateway.gripper_geometry import ArxGripperGeometry
from robots.arx.deployment.picktube_rgbd_provider import (
    PickTubeRgbdProvider,
    FRONT_INTRINSICS_SHA256,
    URDF_LIMITS_SHA256,
)


def test_tube_column_keeps_only_seed_connected_measured_points():
    rgb = np.full((24, 20, 3), 120, dtype=np.uint8)
    yy, xx = np.indices(rgb.shape[:2])
    points = np.stack(((xx - 8) * 0.005, np.zeros_like(xx), yy * 0.003), axis=-1)
    seed = np.zeros(rgb.shape[:2], bool)
    seed[7:10, 7:10] = True
    rgb[20:] = [240, 240, 10]
    points[18, 6:11] = np.nan  # stereo hole: cannot fill or bridge it
    mask = upright_tube_mask(rgb, points, np.array([0.0, 0.0, 0.024]), seed)
    assert np.all(np.isfinite(points[mask]))
    assert not mask[18:].any()  # disconnected upper surface remains unobserved
    assert not mask[:, 15:].any()
    with pytest.raises(ValueError, match="surface extent"):
        upright_tube_mask(rgb, points, np.array([0.0, 0.0, 0.024]), seed, above_m=0.005)


def test_pinned_gripper_hulls_detect_body_and_preserve_clear_scene():
    path = (
        Path(__file__).resolve().parents[1]
        / "robots/arx/manifests/real/ac_one_gripper_hulls.json"
    )
    geometry = ArxGripperGeometry(path, hashlib.sha256(path.read_bytes()).hexdigest())
    pose = np.eye(4)
    pose[:3, 3] = geometry.tcp_offset
    points = np.array([[0.035, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert geometry.occupied_mask(points, pose).tolist() == [True, False]
    from scipy.spatial import cKDTree

    with pytest.raises(ValueError, match="CAD sweep"):
        geometry.check(cKDTree(points), pose)
    geometry.check(cKDTree(points[1:]), pose)
    with pytest.raises(ValueError, match="SHA"):
        ArxGripperGeometry(path, "0" * 64)


def test_component_hulls_keep_body_but_do_not_fill_inter_part_void():
    root = Path(__file__).resolve().parents[1] / "robots/arx/manifests/real"
    geometries = [ArxGripperGeometry(root / name, hashlib.sha256((root/name).read_bytes()).hexdigest())
                  for name in ("ac_one_gripper_hulls.json", "ac_one_gripper_component_hulls.json")]
    pose = np.eye(4); pose[:3,3] = geometries[0].tcp_offset
    points = np.array([[.035,0,0], [.01988532424,.03071239464,-.01409020257], [1,1,1]])
    assert geometries[0].occupied_mask(points,pose).tolist() == [True,True,False]
    assert geometries[1].occupied_mask(points,pose).tolist() == [True,False,False]


def hardware_for_bounds(provider, bounds, sha=None):
    root = Path(__file__).resolve().parents[1]
    front = SimpleNamespace(
        name="front_rgb",
        serial="260422272500",
        depth_enabled=True,
        capture_width=640,
        capture_height=480,
        width=320,
        height=240,
        calibration_sha256=FRONT_INTRINSICS_SHA256,
        calibration_file=root
        / "robots/arx/manifests/real/dodo_front_rgb_d405_intrinsics.json",
    )
    return SimpleNamespace(
        cameras=[front],
        right_gripper_closed_policy=0.0,
        right_gripper_open_policy=-3.4,
        right_joint_limit_profile_sha256=sha,
        right=SimpleNamespace(
            joint_min_rad=[x[0] for x in bounds], joint_max_rad=[x[1] for x in bounds]
        ),
    )


def test_expanded_motion_bounds_require_pinned_source_and_stay_inside_cad():
    provider = PickTubeRgbdProvider()
    bounds = [link.limits.copy() for link in provider.controller_fk.calibration.links]
    bounds[3][1] = 1.27
    bounds[5] = [-1.5, 1.5]
    with pytest.raises(ValueError, match="FK envelope"):
        provider.validate_hardware(hardware_for_bounds(provider, bounds))
    provider.validate_hardware(
        hardware_for_bounds(provider, bounds, URDF_LIMITS_SHA256)
    )
    assert provider.tool_fk.calibration.links[5].limits == [-1.5, 1.5]
    bounds[5] = [-4.0, 4.0]
    with pytest.raises(ValueError, match="pinned AC one"):
        provider.validate_hardware(
            hardware_for_bounds(provider, bounds, URDF_LIMITS_SHA256)
        )


def test_small_visible_seed_requires_rack_and_twelve_pixels():
    provider = PickTubeRgbdProvider()
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[70:105, 110:220] = [240, 240, 10]
    rgb[41:47, 179:181] = [230, 70, 150]
    assert provider._pink_component(rgb).sum() == 12
    rgb[:] = 0
    rgb[41:47, 179:181] = [230, 70, 150]
    with pytest.raises(ValueError, match="rack"):
        PickTubeRgbdProvider()._pink_component(rgb)


def test_model_pose_conditions_select_without_modifying_learned_poses():
    from scripts.deployment.serve_arx_graspgen_local import pose_condition_mask
    poses=np.repeat(np.eye(4)[None],3,axis=0)
    poses[1,:3,:3]=[[0,0,1],[1,0,0],[0,1,0]]
    poses[2,:3,:3]=[[0,0,1],[0,-1,0],[1,0,0]]
    original=poses.copy()
    keep=pose_condition_mask(poses,{"up_camera":[0,0,1],"horizontal_closing_max":.35,
        "preferred_approach_camera":[1,0,0],"approach_alignment_min":.7})
    assert keep.tolist()==[False,True,False]
    assert np.array_equal(poses,original)


def test_expanded_cad_pose_must_also_fit_installed_sdk_limits():
    provider = PickTubeRgbdProvider()
    bounds = [link.limits.copy() for link in provider.controller_fk.calibration.links]
    bounds[3] = [-1.6, 1.6]  # Inside CAD but beyond the installed SDK's ±1.29.
    with pytest.raises(ValueError, match='installed SDK limits'):
        provider.validate_hardware(hardware_for_bounds(provider, bounds, URDF_LIMITS_SHA256))


def test_side_grasp_filter_rejects_vertical_approach_without_rewriting_pose():
    from scripts.deployment.serve_arx_graspgen_local import pose_condition_mask
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    original = poses.copy()
    mask = pose_condition_mask(poses, {'up_camera': [0, 0, 1],
        'horizontal_closing_max': .2, 'horizontal_approach_max': .15})
    assert mask.tolist() == [False, True]
    np.testing.assert_array_equal(poses, original)


def test_diversity_retains_lower_score_reachable_alternative_and_halfturn_identity():
    from scripts.deployment.serve_arx_graspgen_local import diverse_pose_indices
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[1, 0, 3] = .001  # Near-identical high score.
    poses[2, :3, :3] = np.diag([-1., -1., 1.])  # Same parallel jaws, exchanged.
    poses[3, 0, 3] = .02  # A distinct, lower-score pose must stay available.
    original = poses.copy()
    assert diverse_pose_indices(poses, np.array([.9, .8, .7, .6]), 2).tolist() == [0, 3]
    np.testing.assert_array_equal(poses, original)


def test_metric_column_accepts_disconnected_measured_glass_but_not_rack_or_neighbor():
    rgb = np.full((24, 20, 3), 120, dtype=np.uint8)
    yy, xx = np.indices(rgb.shape[:2])
    points = np.stack(((xx-8)*.005, np.zeros_like(xx), yy*.003), axis=-1)
    seed = np.zeros(rgb.shape[:2], bool);seed[7:10, 7:10] = True
    points[18, 6:11] = np.nan
    rgb[22:] = [240, 240, 10]
    mask = upright_tube_mask(rgb, points, np.array([0., 0., .024]), seed,
                             radius_m=.018, allow_disconnected=True)
    assert mask[20, 8] and not mask[18, 6:11].any()
    assert not mask[22:].any() and not mask[:, 15:].any()
    assert np.isfinite(points[mask]).all()
