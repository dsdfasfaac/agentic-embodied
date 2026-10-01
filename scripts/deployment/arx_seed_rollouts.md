Run from the repository with the MuJoCo Python environment:

```bash
MUJOCO_GL=osmesa XDG_CACHE_HOME=/tmp/zetta-arx-cache \
  /home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  scripts/deployment/run_arx_seed_rollouts.py
```

Defaults select 10 distinct seeds reproducibly using `--seed-selection 17`,
and execute at most 32 inference chunks per seed. Each chunk executes at most
the task's `execution_steps` (currently 16). Each rollout ends on evaluator
success, safety failure, task time limit, or the chunk limit. Errors are logged
and the batch continues to the next seed; the batch exits nonzero if any process
failed. Use `--seeds 17 18 19` for explicit seeds, or change `--seed-selection`
to choose a different set. The existing backend must be listening on port 5581.

Each invocation creates a new `batch_<UTC timestamp>` directory under
`runs/arx_pickip_test_tube_rollout`, with `summary.json`, process logs, and a
directory for each seed. No previous batch is overwritten.

Per-seed outputs:

- `front_rgb.mp4`, `left_rgb.mp4`, `right_rgb.mp4`, `three_view.mp4`: reset
  frame followed by every policy-step frame, at 15 fps.
- `raw_actions.npy`: all predicted horizons, including unexecuted tails.
- `executed_actions.npy`: filtered command targets, not measured positions.
- `trajectory.npz`: aligned numeric arrays described below.
- `audit.json`: success, terminal reason, chunk/step counts and evaluator result.
- `diagnostic_trace.jsonl`: step and controller diagnostics.

For N executed policy actions, trajectory arrays are:

| Key | Shape | Meaning |
| --- | --- | --- |
| `vla_action_targets` | N × 14 | Raw VLA targets actually submitted to the environment |
| `processed_action_targets` | N × 14 | Targets after safety filters and task locks |
| `actual_tracked_actions` | N × 14 | Measured state after each action (absolute positions) |
| `robot_states` | (N+1) × 14 | Initial state and all post-action states |
| `chunk_index`, `action_index` | N | Zero-based source prediction coordinates |
| `time` | N+1 | Simulation timestamps in seconds |
| `qpos`, `qvel`, `ctrl` | (N+1) × model dimension | Full simulator positions, velocities, actuator controls |

Action i maps `robot_states[i]` to `robot_states[i+1]`; video frame i+1
shows its measured result. The 14-channel order is left arm joints 1–6,
left gripper, right arm joints 1–6, right gripper. Grippers use the existing
hardware-domain conversion and clipping; `qpos` preserves simulator readback.
These are policy-rate recordings, not every physics substep. Numeric arrays and
videos are finalized when the rollout completes; a crashed rollout may only
have diagnostics and its process log.

The environment restores a fixed scene for every seed. The batch runner also
passes that seed as `--inference-seed`, which sends `options.seed` to the backend
on every chunk. This overrides the launcher's fixed seed 42 without restarting
the server. Each rollout reuses its selected inference seed across chunks,
matching the server's deterministic-seed behavior. This varies policy sampling,
not object placement; it does not guarantee bitwise GPU reproducibility.

```python
import numpy as np
trajectory = np.load("runs/arx_pickip_test_tube_rollout/batch_.../seed_17/trajectory.npz")
tracking_error = trajectory["actual_tracked_actions"] - trajectory["processed_action_targets"]
```

To collect all ten scenes on multiple GPUs, start an inference server on each
GPU in separate terminals. Give each server its own port and output directory:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=5581 ZEVA_OUTPUT_DIR=/tmp/arx-server-0 \
  bash scripts/deployment/start_zeva_arx_task7_server.sh
CUDA_VISIBLE_DEVICES=1 PORT=5582 ZEVA_OUTPUT_DIR=/tmp/arx-server-1 \
  bash scripts/deployment/start_zeva_arx_task7_server.sh
```

Once both servers are ready, run with the MuJoCo Python environment:

```bash
python scripts/deployment/run_arx_10_scenes.py --collect --gpus 0 1 \
  --ports 5581 5582
```

Scenes are prepared sequentially, then collected concurrently with one scene
batch per GPU/server. Each worker takes the next available scene after finishing
its current batch; seeds within a scene remain sequential. Ports default to
`--port` (5581) plus the GPU's position in `--gpus`. GPU IDs are passed directly
as `CUDA_VISIBLE_DEVICES` to rollout subprocesses; the separately launched
servers must use the corresponding devices. All servers use `--host`.
Without `--gpus`, collection remains sequential. Per-scene output paths and
seed selection are unchanged, and failures are reported after all scenes have
been attempted. `--offline-test` exercises collection without inference servers.
