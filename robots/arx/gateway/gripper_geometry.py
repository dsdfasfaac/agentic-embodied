"""Pinned AC one CAD hull checks against observed metric scene points."""

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree


class ArxGripperGeometry:
    def __init__(self, path, expected_sha256):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("gripper geometry SHA differs")
        value = json.loads(raw)
        if value["schema_version"] != "arx.gripper.cad.v1":
            raise ValueError("unsupported gripper geometry")
        self.sha256 = expected_sha256
        self.tcp_offset = np.asarray(value["tcp_link6_m"])
        self.parts = value["parts"]
        self.slider = value["finger_slider_range_m"]

    def _parts_at(self, tcp_pose):
        r = tcp_pose[:3, :3]
        origin = tcp_pose[:3, 3] - r @ self.tcp_offset
        for parent in self.parts:
            for component_index, component in enumerate(parent.get("convex_components", [parent])):
                part = {**parent, **component}
                equations = np.asarray(part["convex_halfspaces"]).copy()
                axis = np.asarray(part["slider_axis_link6"])
                # Conservative union across the entire possible slider opening.
                shifts = (equations[:, :3] @ axis)[:, None] * np.asarray(self.slider)
                equations[:, 3] -= shifts.max(axis=1)
                bounds = np.asarray(part["bounds_local_m"])
                low = bounds[0] + np.minimum(axis * self.slider[0], axis * self.slider[1])
                high = bounds[1] + np.maximum(axis * self.slider[0], axis * self.slider[1])
                offset = np.asarray(part["origin_link6_m"])
                centre = (low + high) / 2
                link = part["link"]
                if "convex_components" in parent:
                    link += f":component-{component_index}"
                yield link, equations, r, origin + r @ offset, centre, np.linalg.norm(high - low) / 2
    def occupied_mask(self, points, pose, *, margin_m=0.005):
        result = np.zeros(len(points), dtype=bool)
        tree = cKDTree(points)
        for _, eq, r, origin, centre, radius in self._parts_at(pose):
            indices = np.asarray(
                tree.query_ball_point(origin + r @ centre, radius + margin_m), dtype=int
            )
            if len(indices):
                local = (points[indices] - origin) @ r
                result[indices] |= np.all(
                    local @ eq[:, :3].T + eq[:, 3] <= margin_m, axis=1
                )
        return result

    def check(self, tree, pose, *, margin_m=0.005):
        for link, eq, r, origin, centre, radius in self._parts_at(pose):
            indices = tree.query_ball_point(origin + r @ centre, radius + margin_m)
            if indices:
                local = (tree.data[indices] - origin) @ r
                inside = np.all(local @ eq[:, :3].T + eq[:, 3] <= margin_m, axis=1)
                if np.any(inside):
                    point = tree.data[np.asarray(indices)[np.flatnonzero(inside)[0]]]
                    raise ValueError(
                        "ARX gripper CAD sweep intersects observed scene: " + link
                        + "; point_base_m=" + json.dumps(point.tolist())
                        + "; tcp_base_m=" + json.dumps(pose[:3, 3].tolist())
                    )
