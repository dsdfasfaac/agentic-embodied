# Copyright (c) 2026 Zetta Contributors
"""Command-space kinematics using a robot-only, reviewed calibration artifact."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field, model_validator

from .contracts import StrictModel, Vector


class Link(StrictModel):
    position: Vector
    quaternion: list[float] = Field(min_length=4, max_length=4)
    axis: Vector
    limits: list[float] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid(self):
        if abs(np.linalg.norm(self.axis) - 1) > 1e-6:
            raise ValueError("joint axis must have unit length")
        if abs(np.linalg.norm(self.quaternion) - 1) > 1e-6:
            raise ValueError("quaternion must have unit length")
        if self.limits[0] >= self.limits[1]:
            raise ValueError("invalid joint limits")
        return self


class Calibration(StrictModel):
    schema_version: str = "arx.robot.calibration.v1"
    base_position: Vector
    base_quaternion: list[float] = Field(min_length=4, max_length=4)
    links: list[Link] = Field(min_length=6, max_length=6)
    tcp_offset: Vector

    @model_validator(mode="after")
    def valid(self):
        if (
            self.schema_version != "arx.robot.calibration.v1"
            or abs(np.linalg.norm(self.base_quaternion) - 1) > 1e-6
        ):
            raise ValueError("invalid calibration")
        return self


def rotation(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = np.asarray(v) / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def quaternion(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class CommandKinematics:
    def __init__(self, calibration: Calibration):
        self.calibration = Calibration.model_validate(calibration.model_dump())

    def fk(self, joints):
        c = self.calibration
        p, r = np.asarray(c.base_position).copy(), quaternion(c.base_quaternion)
        axes, origins = [], []
        for q, link in zip(joints, c.links):
            if not link.limits[0] <= q <= link.limits[1]:
                raise ValueError("command exceeds static joint limit")
            p = p + r @ link.position
            r = r @ quaternion(link.quaternion)
            axes.append(r @ link.axis)
            origins.append(p.copy())
            r = r @ rotation(np.asarray(link.axis) * q)
        tcp = p + r @ c.tcp_offset
        jac = np.vstack(
            (
                np.array([np.cross(a, tcp - o) for a, o in zip(axes, origins)]).T,
                np.array(axes).T,
            )
        )
        return tcp, r, jac

    def solve(self, joints, target_p, target_r):
        initial = joints.copy().astype(float)
        self.fk(initial)  # Invalid measured starts must never be projected into bounds.
        # Solve within the reviewed joint envelope, including near singular
        # starts. Every Cartesian caller separately checks joint increments.
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        bounds = np.asarray([link.limits for link in self.calibration.links])
        def residual(value):
            p, r, _ = self.fk(value)
            return np.r_[p - target_p, .12 * Rotation.from_matrix(r @ target_r.T).as_rotvec()]
        def jacobian(value):
            _, r, jac = self.fk(value)
            v = Rotation.from_matrix(r @ target_r.T).as_rotvec()
            angle = np.linalg.norm(v)
            x, y, z = v
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            coefficient = 1. / 12. if angle < 1e-5 else (
                1. - angle / (2. * np.tan(angle / 2.))) / angle ** 2
            inverse_left = np.eye(3) - .5 * skew + coefficient * skew @ skew
            return np.vstack((jac[:3], .12 * inverse_left @ jac[3:]))
        solved = least_squares(residual, initial, jac=jacobian, bounds=(bounds[:, 0], bounds[:, 1]),
                               max_nfev=200, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        p, r, _ = self.fk(solved.x)
        if (np.linalg.norm(p - target_p) >= .00015
                or np.linalg.norm(Rotation.from_matrix(r @ target_r.T).as_rotvec()) >= .003):
            raise ValueError("bounded IK did not converge")
        return solved.x

    def plan(self, command, args, *, settling_steps=15):
        d, rv = np.asarray(args.delta_xyz_m), np.asarray(args.delta_rotvec_rad)
        q = command[7:13].copy()
        p, r, _ = self.fk(q)
        if args.frame == "tool":
            d, rv = r @ d, r @ rv
        n = max(
            1,
            math.ceil(np.linalg.norm(d) / min(0.001, args.speed_m_s / 15)),
            math.ceil(np.linalg.norm(rv) / 0.01),
        )
        if n + settling_steps > 60:
            raise ValueError("EEF plan exceeds 60 actions")
        targets = []
        for i in range(1, n + 1):
            alpha = i / n
            q = self.solve(q, p + alpha * d, rotation(alpha * rv) @ r)
            target = command.copy()
            target[7:13] = q
            targets.append(target)
        targets.extend([targets[-1].copy() for _ in range(settling_steps)])
        return np.asarray(targets, dtype=np.float32)
