import numpy as np
from scripts.deployment.stage_arx_picktube_start import _tracking_correction


def test_stalled_joint_correction_is_bounded_and_stops_at_tracking_envelope():
    before = np.zeros(14); before[12] = .452
    measured = before.copy(); target = before.copy(); target[12] -= .035
    goal = before.copy(); goal[12] = .07
    tolerance = np.full(14, .06)
    for _ in range(10):
        corrected = _tracking_correction(target, measured, before, goal, [12], tolerance, .035)
        assert np.max(np.abs(corrected - target)) <= .005 + 1e-12
        assert measured[12] - corrected[12] <= .055 + 1e-12
        assert np.array_equal(corrected[:12], target[:12])
        target = corrected
    assert target[12] == .452 - .055


def test_progressing_or_arrived_joint_does_not_get_extra_lead():
    before = np.zeros(14); before[12] = .452
    goal = before.copy(); goal[12] = .07
    target = before.copy(); target[12] -= .035
    measured = before.copy(); measured[12] -= .002
    tolerance = np.full(14, .06)
    assert np.array_equal(_tracking_correction(target, measured, before, goal, [12], tolerance, .035), target)
    measured[12] = .10
    assert np.array_equal(_tracking_correction(target, measured, before, goal, [12], tolerance, .035), target)
