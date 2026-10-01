"""Bounded Cartesian tools using static robot geometry and command history only.

No simulator object/contact/state access. FK predicts commanded TCP, not measured
TCP. The caller must inspect RGB after each primitive. This is not a collision
planner and never certifies that a path is safe.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotvec(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = skew(v / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def quatmat(q):
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def vector(x, n):
    x = np.asarray(x, dtype=float)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError(f"expected {n} finite numbers")
    return x


class CommandKinematics:
    def __init__(self, calibration):
        self.calibration = calibration
        self.chain = calibration['chain']
        self.tcp = vector(calibration['tcp_offset_link6_m'], 3)

    @classmethod
    def from_scene_xml(cls, path):
        # Static robot transforms only. No object geometry or simulation state.
        root = ET.parse(path).getroot()
        chain = []
        parent = root.find("worldbody/body[@name='right_base_link']")
        if parent is None:
            raise ValueError('expected fixed right_base_link directly in worldbody')
        chain.append({'position': list(map(float, parent.get('pos','0 0 0').split())),
                      'quaternion': list(map(float, parent.get('quat','1 0 0 0').split())),
                      'axis': None})
        for i in range(1, 7):
            parent = parent.find(f"body[@name='right_link{i}']")
            if parent is None:
                raise ValueError('unexpected robot chain')
            j = parent.find(f"joint[@name='right_joint{i}']")
            if j is None or j.get('type') != 'hinge' or j.get('pos','0 0 0') != '0 0 0':
                raise ValueError('only origin-centered revolute chain supported')
            chain.append({'position': list(map(float,parent.get('pos','0 0 0').split())),
                          'quaternion': list(map(float,parent.get('quat','1 0 0 0').split())),
                          'axis': list(map(float,j.get('axis').split())),
                          'range': list(map(float,j.get('range').split()))})
        return cls({'chain':chain, 'tcp_offset_link6_m':[0.12957,0.,0.0137564],
                    'tcp_definition':'fixed midpoint of distal finger pads at nominal closure',
                    'source':'static robot geometry; no runtime simulator readback'})

    def fk(self, q):
        q = vector(q, 6)
        p, r = np.zeros(3), np.eye(3)
        axes, origins = [], []
        for i, link in enumerate(self.chain):
            p = p + r @ np.asarray(link['position'])
            r = r @ quatmat(link['quaternion'])
            if link['axis'] is not None:
                axis = vector(link['axis'],3)
                axes.append(r @ axis)
                origins.append(p.copy())
                r = r @ rotvec(axis * q[i-1])
        tcp = p + r @ self.tcp
        jac = np.vstack([np.array([np.cross(a,tcp-o) for a,o in zip(axes,origins)]).T,
                         np.array(axes).T])
        return tcp, r, jac

    def solve(self, q, target_p, target_r):
        q = vector(q,6).copy()
        for _ in range(160):
            p,r,j = self.fk(q)
            er = .5 * sum((np.cross(r[:,i],target_r[:,i]) for i in range(3)))
            ep = target_p - p
            if np.linalg.norm(ep) < 0.00015 and np.linalg.norm(er) < 0.003:
                return q
            # Orientation has units radians, scaled to metres for damping.
            jw = j.copy(); jw[3:] *= .12
            e = np.r_[ep, .12*er]
            dq = jw.T @ np.linalg.solve(jw@jw.T + 0.00001*np.eye(6), e)
            q += np.clip(dq,-.04,.04)
            for i,link in enumerate(self.chain[1:]):
                if not link['range'][0] <= q[i] <= link['range'][1]:
                    raise ValueError('IK joint limit')
        raise ValueError('IK did not converge; no command executed')

    def move_eef(self, command, *, delta_xyz_m, delta_rotvec_rad=(0,0,0),
                 frame='world', speed_m_s=.015):
        """Plan <=10 mm / <=0.10 rad relative move, with <=1 mm waypoints.

        world and tool frames supported; rotation is held when omitted. The
        returned 14D targets preserve the other arm and current gripper command.
        """
        command = vector(command,14)
        d, rv = vector(delta_xyz_m,3), vector(delta_rotvec_rad,3)
        if frame not in ('world','tool'):
            raise ValueError('frame must be world or tool')
        if np.linalg.norm(d)> .01000001 or np.linalg.norm(rv)>.10000001:
            raise ValueError('move exceeds 10 mm or 0.10 rad per Agent decision')
        if not 0 < speed_m_s <= .03:
            raise ValueError('speed must be in (0, 0.03] m/s')
        q = command[7:13].copy()
        p,r,_ = self.fk(q)
        if frame == 'tool':
            d, rv = r@d, r@rv
        n = max(1,int(np.ceil(np.linalg.norm(d)/min(.001,speed_m_s/15))),
                int(np.ceil(np.linalg.norm(rv)/.01)))
        targets=[]
        for i in range(1,n+1):
            alpha=i/n
            q = self.solve(q,p+alpha*d,rotvec(alpha*rv)@r)
            target=command.copy(); target[7:13]=q
            targets.append(target)
        return targets


def set_gripper(command, *, opening):
    """opening 0=closed, 1=open; preserves both arms, returns hardware target."""
    if not np.isfinite(opening) or not 0 <= opening <= 1:
        raise ValueError('opening must be between 0 and 1')
    target=vector(command,14).copy(); target[13]=-3.4*opening
    return target
