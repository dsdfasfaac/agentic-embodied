#!/usr/bin/env python3
"""Offline rebuild of CAD hulls, preserving voids between mesh components.

Requires trimesh/scipy in the build environment. Runtime only loads the JSON.
The source STL SHAs must equal the existing pinned AC one geometry manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def build(source, meshes, output):
    import trimesh

    value = json.loads(source.read_text())
    rows, parts = [], []
    for part in value["parts"]:
        path = meshes / (part["link"] + ".STL")
        if hashlib.sha256(path.read_bytes()).hexdigest() != part["mesh_sha256"]:
            raise ValueError("source CAD mesh SHA differs: " + part["link"])
        mesh = trimesh.load_mesh(path)
        if not mesh.is_watertight:
            raise ValueError("source CAD mesh is not watertight")
        components = mesh.split()
        record = {k:v for k,v in part.items() if k not in ("convex_halfspaces","bounds_local_m")}
        record["convex_components"] = []
        volume = 0.
        for component in components:
            if not component.is_watertight:
                raise ValueError("source component is not watertight")
            hull = component.convex_hull
            equations = np.c_[hull.face_normals, -np.einsum("ij,ij->i",hull.face_normals,hull.triangles[:,0])]
            equations = np.unique(np.round(equations,9),axis=0)
            if not np.all(component.vertices @ equations[:,:3].T + equations[:,3] <= 1e-8):
                raise ValueError("component hull lost a CAD vertex")
            record["convex_components"].append({"convex_halfspaces":equations.tolist(),
                                               "bounds_local_m":component.bounds.tolist()})
            volume += hull.volume
        parts.append(record)
        rows.append({"link":part["link"], "component_count":len(components),
                     "mesh_volume_m3":mesh.volume, "old_hull_volume_m3":mesh.convex_hull.volume,
                     "component_hull_volume_m3":volume, "all_source_vertices_enclosed":True})
    value.update(parts=parts,geometry_type="per_connected_mesh_component_convex_hull",coverage_audit=rows)
    value["limitations"] = [
        "Each watertight CAD connected component has its own conservative convex hull; all source vertices covered.",
        "Entire 0..44 mm slider interval retained. This does not calibrate actual jaw width.",
        "Full arm, hidden obstacles and differences between CAD and actual rubber pads are not certified."]
    output.write_text(json.dumps(value,indent=2)+"\n")
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-hulls",type=Path,required=True)
    p.add_argument("--meshes",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    print(build(a.source_hulls,a.meshes,a.output))
