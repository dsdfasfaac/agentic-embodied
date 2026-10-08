"""Read-only joint paths to an exact learned pose; never replace its orientation."""

import math
import numpy as np
from scipy.spatial.transform import Rotation


def joint_paths(
    kinematics,
    start,
    goal,
    *,
    hz,
    speed_m_s,
    angular_speed_rad_s,
    max_joint_step_rad,
    max_steps,
    final_joint_hint=None,
):
    bounds = np.asarray([link.limits for link in kinematics.calibration.links])
    seeds = [start]
    middle = bounds.mean(axis=1)
    # Numerical starting points, not commanded waypoints.
    seeds += [middle, 0.75 * middle + 0.25 * start, 0.25 * middle + 0.75 * start]
    for fraction in (0.35, 0.65):
        seed = middle.copy()
        seed[1:3] = bounds[1:3, 0] + fraction * np.diff(bounds[1:3], axis=1).ravel()
        seeds.append(seed)
    if final_joint_hint is not None:
        seeds = [np.asarray(final_joint_hint)]
    solutions = []
    for seed in seeds:
        try:
            solved = kinematics.solve(seed, goal[:3, 3], goal[:3, :3])
        except ValueError:
            continue
        if not any(np.linalg.norm(solved - old) < 0.01 for old in solutions):
            solutions.append(solved)
    if not solutions:
        raise ValueError("no bounded final IK solution for learned pose")
    for final in sorted(solutions, key=lambda q: np.linalg.norm(q - start)):
        count = max(
            1, math.ceil(np.max(np.abs(final - start)) / min(0.015, max_joint_step_rad))
        )
        for _ in range(6):
            if count + 30 > max_steps:
                break
            path = start + (np.arange(1, count + 1) / count)[:, None] * (final - start)
            poses = [kinematics.fk(q)[:2] for q in np.vstack((start, path))]
            distance = max(
                np.linalg.norm(b[0] - a[0]) for a, b in zip(poses, poses[1:])
            )
            angle = max(
                np.linalg.norm(Rotation.from_matrix(b[1] @ a[1].T).as_rotvec())
                for a, b in zip(poses, poses[1:])
            )
            ratio = max(
                distance / (speed_m_s / hz), angle / (angular_speed_rad_s / hz), 1.0
            )
            if ratio <= 1.0 + 1e-6:
                yield path
                break
            count = max(count + 1, math.ceil(count * ratio * 1.01))
