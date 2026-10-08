import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robots.arx.gateway.motion import Calibration, CommandKinematics
from scripts.deployment.replay_arx_picktube_home import load_recording, plan_home, ROOT


def fk():
    v = json.loads((ROOT / 'robots/arx/manifests/real/dodo_right_controller_ee_fk.json').read_text())
    h = json.loads((ROOT / 'docs/experiments/arx-graspgen-live-20261008/hardware-sdk-bounded.json').read_text())
    for i, link in enumerate(v['links']):
        link['limits'] = [h['right']['joint_min_rad'][i], h['right']['joint_max_rad'][i]]
    v['tcp_offset'] = json.loads((ROOT / 'robots/arx/manifests/real/ac_one_nominal_chain.json').read_text())['tcp_offset']
    return CommandKinematics(Calibration.model_validate(v))


def goal():
    v = json.loads((ROOT / 'robots/arx/manifests/pickup_test_tube.yaml').read_text())
    q = np.array(v['start_state']);q[[6, 13]] += v['control']['gripper_command_offsets']
    return q


def test_replays_lift_then_return_in_order_without_recorded_grip():
    recording = load_recording()
    current = goal(); current[7:13] = recording[1][0];current[13] = -2.45
    targets, plan = plan_home(current, goal(), fk(), recording)
    assert plan['join_source_frame'] == 165
    frames = [i for i in plan['source_frames'] if i is not None]
    assert frames == sorted(frames) and frames[-1] == 300
    np.testing.assert_array_equal(targets[:, :7], np.tile(current[:7], (len(targets), 1)))
    np.testing.assert_array_equal(targets[:, 13], np.full(len(targets), -2.45))
    assert np.max(np.abs(np.diff(np.vstack([current, targets]), axis=0)[:, 7:13])) <= .2 / 15 + 1e-9
    ps = np.array([fk().fk(q[7:13])[0] for q in targets])
    assert ps[:, 2].max() > ps[0, 2] + .1
    np.testing.assert_allclose(targets[-1, 7:13], goal()[7:13])


def test_near_home_does_not_go_back_to_rack():
    current = goal(); current[10] += .02
    targets, plan = plan_home(current, goal(), fk(), load_recording())
    assert plan['already_near_home']
    assert set(plan['phases']) == {'home_alignment'}


def test_reordered_or_tampered_recording_rejected(tmp_path):
    v = copy.deepcopy(load_recording()[0]);v['timestamps_s'][2] = v['timestamps_s'][0]
    p = tmp_path / 'invalid.json';p.write_text(json.dumps(v))
    with pytest.raises(ValueError, match='reordered'):
        load_recording(p, hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='SHA'):
        load_recording(p)


def test_invalid_feedback_cannot_be_projected_into_limits():
    current = goal(); current[11] = 2.
    with pytest.raises(ValueError, match='limit'):
        plan_home(current, goal(), fk(), load_recording())


def test_learned_pregrasp_enters_by_vertical_escape_before_recorded_return():
    current = goal()
    current[7:13] = [.63878, 2.08877, 1.03094, .91154, .42897, .19932]
    kinematics = fk()
    initial_p = kinematics.fk(current[7:13])[0]
    targets, plan = plan_home(current, goal(), kinematics, load_recording())
    vertical = [q for q, phase in zip(targets, plan['phases']) if phase == 'vertical_escape']
    assert vertical and plan['join_source_frame'] == 186
    xyz = np.array([kinematics.fk(q[7:13])[0] for q in vertical])
    np.testing.assert_allclose(xyz[:, :2], np.tile(initial_p[:2], (len(xyz), 1)), atol=.0002)
    assert np.all(np.diff(xyz[:, 2]) >= -.0002)
    assert xyz[-1, 2] >= initial_p[2] + .06 - .0002
    bridge = [q for q, phase in zip(targets, plan['phases']) if phase == 'recorded_entry']
    assert min(kinematics.fk(q[7:13])[0][2] for q in bridge) >= xyz[-1, 2] - .002
