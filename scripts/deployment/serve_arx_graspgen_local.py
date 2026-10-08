#!/usr/bin/env python3
"""Load GraspGen weights on dodo and expose loopback-only proposal inference.

Uses the official NVlabs GraspGen API; no robot/ROS/SDK module is imported.
Run in an isolated model environment, independently of the VLA environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


def pose_condition_mask(poses, constraints):
    """Select learned poses without altering their positions or rotations."""
    keep = np.ones(len(poses), dtype=bool)
    if "up_camera" in constraints:
        keep &= np.abs(poses[:, :3, 0] @ np.asarray(constraints["up_camera"])) <= constraints["horizontal_closing_max"]
    if "preferred_approach_camera" in constraints:
        keep &= poses[:, :3, 2] @ np.asarray(constraints["preferred_approach_camera"]) >= constraints["approach_alignment_min"]
    return keep


class Model:
    def __init__(self, config_path, seed=42):
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info
        import torch

        self.api = GraspGenSampler
        config = load_grasp_cfg(str(config_path))
        self.gripper = str(config.data.gripper_name)
        info = get_gripper_info(self.gripper)
        self.gripper_tcp = np.asarray(
            info.transform_from_base_link_to_tool_tcp
        ).tolist()
        self.gripper_depth = float(info.depth)
        self.seed, self.request_index = seed, 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        entries = {
            "config": Path(config_path),
            "generator": Path(config.eval.checkpoint),
            "discriminator": Path(config.discriminator.checkpoint),
        }
        self.files = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in entries.items()
        }
        self.model_sha = hashlib.sha256(
            json.dumps(self.files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.sampler = GraspGenSampler(config)
        self.lock = threading.Lock()

    def health(self):
        return {
            "ready": True,
            "gripper_id": self.gripper,
            "model_sha256": self.model_sha,
            "checkpoint_sha256": self.files,
            "backend": "nvlabs-graspgen-local",
            "proposal_only": True,
            "base_seed": self.seed,
            "grasp_approach_axis": "+Z",
            "grasp_closing_axis": "X",
            "model_gripper_depth_m": self.gripper_depth,
            "transform_grasp_from_model_tcp": self.gripper_tcp,
        }

    def generate(self, payload):
        points = np.asarray(payload["point_cloud"], dtype=np.float32)
        count = int(payload.get("topk_num_grasps", 8))
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not 32 <= len(points) <= 65536
            or not np.isfinite(points).all()
            or not 1 <= count <= 32
            or payload.get("frame") != "object_centered_opencv_camera_xyz_m"
        ):
            raise ValueError(
                "expected 32..65536 metric object-centred points and 1..32 candidates"
            )
        if payload.get("filter_collisions"):
            raise ValueError(
                "model server has no full scene; request collision review in the robot gateway"
            )
        samples = int(payload.get("num_model_samples", max(64, count * 4)))
        if not 64 <= samples <= 1024:
            raise ValueError("model sampling budget must be in 64..1024")
        constraints = {}
        if payload.get("horizontal_closing_max") is not None:
            maximum = float(payload["horizontal_closing_max"])
            up = np.asarray(payload["up_camera"], dtype=float)
            if (
                not 0 <= maximum <= 0.5
                or up.shape != (3,)
                or not np.isfinite(up).all()
                or abs(np.linalg.norm(up) - 1) > 1e-5
            ):
                raise ValueError("invalid closing-axis condition")
            constraints.update(horizontal_closing_max=maximum, up_camera=up.tolist())
        if payload.get("preferred_approach_camera") is not None:
            direction = np.asarray(payload["preferred_approach_camera"], dtype=float)
            minimum = float(payload["approach_alignment_min"])
            if (
                not 0.5 <= minimum <= 0.99
                or direction.shape != (3,)
                or not np.isfinite(direction).all()
                or abs(np.linalg.norm(direction) - 1) > 1e-5
            ):
                raise ValueError("invalid approach-axis condition")
            constraints.update(
                preferred_approach_camera=direction.tolist(),
                approach_alignment_min=minimum,
            )
        with self.lock:
            import torch

            request_index = self.request_index
            seed = (self.seed + request_index) % (2**32)
            self.request_index += 1
            torch.manual_seed(seed)
            np.random.seed(seed)
            started = time.monotonic()
            grasps, scores = self.api.run_inference(
                points,
                self.sampler,
                num_grasps=samples,
                topk_num_grasps=samples,
                min_grasps=1,
                max_tries=1,
                remove_outliers=False,
            )
        poses = grasps.detach().cpu().numpy()
        keep = pose_condition_mask(poses, constraints)
        indices = np.flatnonzero(keep)[:count]
        raw_count = len(poses)
        grasps, scores = grasps[indices], scores[indices]
        return {
            "ok": True,
            "grasps": [
                {
                    "transform_model": pose.detach().cpu().numpy().tolist(),
                    "score": float(score),
                }
                for pose, score in zip(grasps, scores)
            ],
            "evidence": {
                **self.health(),
                "collision_filter_applied": False,
                "target_only_cloud": True,
                "input_points": len(points),
                "request_index": request_index,
                "seed": seed,
                "num_model_samples": samples,
                "raw_candidate_count": raw_count,
                "conditioned_candidate_count": int(keep.sum()),
                "pose_conditions": constraints,
                "inference_s": time.monotonic() - started,
                "input_cloud_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            },
            "environment_advanced": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18093)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in 0..2**32-1")
    model = Model(args.config, seed=args.seed)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def write(self, code, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            (
                self.write(200, model.health())
                if self.path == "/health"
                else self.write(404, {"error": "not found"})
            )

        def do_POST(self):
            if self.path != "/generate":
                return self.write(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("invalid request length")
                payload = json.loads(self.rfile.read(length))
                self.write(200, model.generate(payload))
            except (ValueError, KeyError, TypeError) as exc:
                self.write(400, {"ok": False, "error": str(exc)})
            except Exception:
                import traceback

                traceback.print_exc()
                self.write(500, {"ok": False, "error": "grasp inference failed"})

    print(json.dumps(model.health(), sort_keys=True), flush=True)

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if args.host == "::1" else socket.AF_INET

    Server((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
