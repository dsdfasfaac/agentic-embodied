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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


class Model:
    def __init__(self, config_path):
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

        self.api = GraspGenSampler
        config = load_grasp_cfg(str(config_path))
        self.gripper = str(config.data.gripper_name)
        entries = {"config": Path(config_path), "generator": Path(config.eval.checkpoint),
                   "discriminator": Path(config.discriminator.checkpoint)}
        self.files = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in entries.items()}
        self.model_sha = hashlib.sha256(json.dumps(self.files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.sampler = GraspGenSampler(config)
        self.lock = threading.Lock()

    def health(self):
        return {"ready": True, "gripper_id": self.gripper,
                "model_sha256": self.model_sha, "checkpoint_sha256": self.files,
                "backend": "nvlabs-graspgen-local", "proposal_only": True}

    def generate(self, payload):
        points = np.asarray(payload["point_cloud"], dtype=np.float32)
        count = int(payload.get("topk_num_grasps", 8))
        if (points.ndim != 2 or points.shape[1] != 3 or not 32 <= len(points) <= 65536
                or not np.isfinite(points).all() or not 1 <= count <= 32
                or payload.get("frame") != "object_centered_opencv_camera_xyz_m"):
            raise ValueError("expected 32..65536 metric object-centred points and 1..32 candidates")
        if payload.get("filter_collisions"):
            raise ValueError("model server has no full scene; request collision review in the robot gateway")
        with self.lock:
            grasps, scores = self.api.run_inference(points, self.sampler,
                num_grasps=max(64, count * 4), topk_num_grasps=count,
                min_grasps=1, max_tries=1, remove_outliers=False)
        return {"ok": True, "grasps": [{"transform_model": pose.detach().cpu().numpy().tolist(),
                                        "score": float(score)} for pose, score in zip(grasps, scores)],
                "evidence": {**self.health(), "collision_filter_applied": False,
                             "target_only_cloud": True, "input_points": len(points)},
                "environment_advanced": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18093)
    args = parser.parse_args()
    model = Model(args.config)

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
            self.write(200, model.health()) if self.path == "/health" else self.write(404, {"error": "not found"})

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
