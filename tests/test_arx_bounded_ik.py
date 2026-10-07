"""Regression for the measured dodo start that rejected the Newton-only path."""
from pathlib import Path

import numpy as np
import pytest

from robots.arx.gateway.motion import Calibration, CommandKinematics


def kinematics():
    root = Path(__file__).resolve().parents[1] / "robots/arx/manifests/real"
    controller = Calibration.model_validate_json((root / "dodo_right_controller_ee_fk.json").read_text())
    nominal = Calibration.model_validate_json((root / "ac_one_nominal_chain.json").read_text())
    return CommandKinematics(controller.model_copy(update={"tcp_offset": nominal.tcp_offset}))


def test_measured_start_reaches_pregrasp_without_expanding_joint_bounds():
    kin = kinematics()
    q = np.array([.0261306763, .0242233276, .0040054321, .0070571899, .0112533569, .0177383423])
    start, orientation, _ = kin.fk(q)
    goal = np.array([.4196333526, .2716827993, -.0354140649]) - .03 * orientation[:, 0]
    previous = q.copy()
    for fraction in np.linspace(0., 1., 189)[1:]:
        q = kin.solve(q, start + fraction * (goal - start), orientation)
        assert np.max(np.abs(q - previous)) <= .035
        for joint, link in zip(q, kin.calibration.links):
            assert link.limits[0] <= joint <= link.limits[1]
        previous = q.copy()
    position, actual_rotation, _ = kin.fk(q)
    assert np.linalg.norm(position - goal) < .00015
    assert np.linalg.norm(actual_rotation - orientation) < .00425


def test_invalid_measured_start_is_rejected_without_clipping():
    kin = kinematics()
    q = np.zeros(6)
    q[0] = kin.calibration.links[0].limits[0] - .01
    with pytest.raises(ValueError, match="static joint limit"):
        kin.solve(q, np.zeros(3), np.eye(3))


def test_small_pitch_candidate_keeps_full_path_inside_the_same_limits():
    from scipy.spatial.transform import Rotation
    kin = kinematics()
    q = np.array([.0261306763, .0223159790, .0040054321, .0070571899, .0112533569, .0177383423])
    start, orientation, _ = kin.fk(q)
    delta = np.array([0., .05, 0.])
    final_rotation = Rotation.from_rotvec(delta).as_matrix() @ orientation
    goal = np.array([.4196333526, .2716827993, -.0354140649]) - .03 * final_rotation[:, 0]
    for fraction in np.linspace(0., 1., 189)[1:]:
        previous = q.copy()
        q = kin.solve(q, start + fraction * (goal - start),
                      Rotation.from_rotvec(fraction * delta).as_matrix() @ orientation)
        assert np.max(np.abs(q - previous)) <= .035
    position, actual_rotation, _ = kin.fk(q)
    assert np.linalg.norm(position - goal) < .00015
    assert np.linalg.norm(actual_rotation - final_rotation) < .00425
