#!/usr/bin/env python3
"""Run independent ARX rollouts and collect per-seed videos and trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scene = "runs/arx_pickup_test_tube/pickup_test_tube_initial_scene"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=scene)
    parser.add_argument("--mapping", default=f"{scene}/mapping.json")
    parser.add_argument("--task", default=f"{scene}/task.yaml")
    parser.add_argument("--contract", default="robots/arx/manifests/task7_model_a.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5581)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-selection", type=int, default=17,
                        help="seed for reproducibly choosing distinct rollout seeds")
    parser.add_argument("--seeds", type=int, nargs="+", help="explicit rollout seeds")
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("runs/arx_pickip_test_tube_rollout"))
    parser.add_argument("--offline-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.num_seeds <= 1_000_000 or args.chunks < 1:
        parser.error("num-seeds must be in [1, 1000000] and chunks must be positive")
    seeds = args.seeds if args.seeds is not None else random.Random(args.seed_selection).sample(range(1_000_000), args.num_seeds)
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**31 for seed in seeds):
        parser.error("seeds must be distinct integers in [0, 2**31)")
    output = args.output if args.output.is_absolute() else root / args.output
    batch = output / datetime.now(timezone.utc).strftime("batch_%Y%m%dT%H%M%S_%fZ")
    batch.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.setdefault("MUJOCO_GL", "osmesa")
    environment.setdefault("XDG_CACHE_HOME", "/tmp/zetta-arx-cache")
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    summary = {"seeds": seeds, "chunks": args.chunks, "arguments": vars(args), "rollouts": []}

    def save_summary() -> None:
        temporary = batch / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        temporary.replace(batch / "summary.json")

    save_summary()
    print(f"Saving {len(seeds)} rollouts to {batch}", flush=True)
    for index, seed in enumerate(seeds, 1):
        destination = batch / f"seed_{seed}"
        command = [sys.executable, str(root / "scripts/deployment/run_arx_camera_chunk.py")]
        for key in ("scene", "mapping", "task", "contract", "host", "port", "chunks"):
            command.extend([f"--{key}", str(getattr(args, key))])
        command.extend(["--seed", str(seed), "--output", str(destination)])
        command.extend(["--inference-seed", str(seed)])
        if args.offline_test:
            command.append("--offline-test")
        print(f"[{index}/{len(seeds)}] seed={seed}; log: {batch / f'seed_{seed}.log'}", flush=True)
        with (batch / f"seed_{seed}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = {"seed": seed, "returncode": result.returncode, "output": str(destination)}
        if (destination / "audit.json").exists():
            record.update(json.loads((destination / "audit.json").read_text()))
        summary["rollouts"].append(record)
        save_summary()
        print(f"seed={seed}: returncode={result.returncode}, success={record.get('success')}, reason={record.get('terminal_reason')}", flush=True)
    print(f"Summary: {batch / 'summary.json'}", flush=True)
    if any(item["returncode"] != 0 for item in summary["rollouts"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
