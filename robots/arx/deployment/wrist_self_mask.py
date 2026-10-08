"""Measured wrist-camera self pixels, scoped to a calibrated empty jaw opening.

This supplements CAD self filtering. It does not shrink the swept CAD hull,
complete missing depth, or mark scene objects as harmless from colour alone.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def protected_colour(rgb):
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rack = ((hsv[...,0] >= 18) & (hsv[...,0] <= 30)
            & (hsv[...,1] >= 80) & (hsv[...,2] >= 20))
    label = (hsv[...,1] >= 150) & (hsv[...,2] >= 40)
    return rack | label


class WristSelfMask:
    def __init__(self, path, expected_sha256):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("wrist self reference SHA differs")
        self.value = json.loads(raw)
        v = self.value
        if v.get("schema_version") != "arx.wrist.empty-self.v1":
            raise ValueError("unsupported wrist self reference")
        self.sha256 = expected_sha256
        self.samples = np.asarray(v["pixels_xy_depth_mm"], dtype=float)
        if (self.samples.ndim != 2 or self.samples.shape[1] != 3
                or len(self.samples) < 32 or not np.isfinite(self.samples).all()
                or np.any(self.samples[:, :2] != np.floor(self.samples[:, :2]))
                or not np.all((self.samples[:, 2] >= 70) & (self.samples[:, 2] <= 200))
                or not 0 < v["depth_tolerance_mm"] <= 5
                or not 0 < v["gripper_feedback_tolerance"] <= .08
                or v.get("empty_home_observed") is not True):
            raise ValueError("invalid measured wrist self reference")

    def mask(self, rgb, depth, hardware, provider, target_mask):
        v = self.value
        if (list(depth.shape) != v["image_shape"] or rgb.shape != (*depth.shape, 3)
                or target_mask.shape != depth.shape
                or v["camera_serial"] != hardware["camera_health"]["right_rgb"]["device_id"]
                or v["intrinsics_sha256"] != hardware["camera_calibration_sha256"]["right_rgb"]
                or v["wrist_mount_sha256"] != provider.wrist_mount_sha):
            raise ValueError("wrist self reference camera or mount differs")
        result = np.zeros(depth.shape, bool)
        # Outside the calibrated opening, fall back to CAD; never extrapolate.
        if abs(hardware["measured_state"][13] - v["right_gripper_feedback"]) > v["gripper_feedback_tolerance"]:
            return result
        xy = self.samples[:, :2].astype(int)
        if np.any(xy < 0) or np.any(xy[:, 0] >= depth.shape[1]) or np.any(xy[:, 1] >= depth.shape[0]):
            raise ValueError("wrist self reference pixel out of bounds")
        x, y = xy.T
        # Rack, labels and neighbours stay obstacles even when coincident with
        # a reference pixel. Very dark pixels still require matching metric depth.
        coloured = protected_colour(rgb)
        from scipy.spatial import cKDTree
        if not hasattr(self, "tree"):
            points = np.array([provider._deproject_intrinsics(float(px),float(py),d/1000,
                               provider.wrist_intrinsics) for px,py,d in self.samples])
            self.tree = cKDTree(points)
        # Stereo holes change across scenes. Compare CURRENT measured points
        # with retained measured self surfaces in metric camera space. Never
        # invent a depth value or admit a point from silhouette alone.
        py, px = np.nonzero((depth >= 70) & (depth <= 200) & ~coloured & ~target_mask)
        if len(px):
            current = np.array([provider._deproject_intrinsics(float(qx),float(qy),
                float(depth[qy,qx])/1000,provider.wrist_intrinsics) for qx,qy in zip(px,py)])
            matched = self.tree.query(current)[0] <= v["depth_tolerance_mm"]/1000
            result[py[matched],px[matched]] = True
        return result
