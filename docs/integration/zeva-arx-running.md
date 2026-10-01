# Running Zeva with the ARX MuJoCo scene

For the proposed opt-in agent/critic recovery integration, see
[the Zetta supervision integration plan](zeva-arx-agentic-plan.md). Its new
flags are planned; the commands below describe the current implemented runner.

This is the shortest validated runbook for the current `PickUpTestTube`
integration. Run every command from:

```bash
cd /data4/zhengyikai/Agentic-Embodied
```

The Cosmos server and MuJoCo client use separate Python environments. Start
them in two terminals. The server uses CUDA; the client uses OSMesa because
EGL produces corrupted RGB buffers on the current host.

## 1. Start the Cosmos3-Edge Task7 server

In terminal 1:

```bash
export COSMOS_ARX_BUNDLE=/data4/zhengyikai/cosmos3-edge-arx5-inference
export COSMOS_ARX_PYTHON="$COSMOS_ARX_BUNDLE/.venv/bin/python"
export ZEVA_CHECKPOINT=/data4/zhengyikai/ckpt1/model
export ZEVA_CONFIG=/data4/zhengyikai/ckpt1/config/config.yaml
export EDGE_MODEL_PATH=/data4/zhengyikai/ckpt1/model
export WAN_VAE_PATH="$COSMOS_ARX_BUNDLE/models/Wan2.2_VAE.pth"
export HOST=127.0.0.1
export PORT=5581

./scripts/deployment/start_zeva_arx_task7_server.sh
```

Wait until the server is listening before starting the client. The launcher
checks the checkpoint, config, model, VAE, Python environment, and NVIDIA GPU.
Its defaults now point to the same accepted `ckpt1` artifacts, so on this host
the minimal equivalent is:

```bash
./scripts/deployment/start_zeva_arx_task7_server.sh
```

## 2. Run a one-chunk live smoke test

In terminal 2:

```bash
MUJOCO_GL=osmesa \
XDG_CACHE_HOME=/tmp/zetta-arx-cache \
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  scripts/deployment/run_arx_camera_chunk.py \
  --scene runs/arx_pickup_test_tube/dark_silver_estimated_top_scene \
  --mapping /data4/zhengyikai/Zeva_arx/assets/ac_one/ac_one_14d_mapping.json \
  --task robots/arx/manifests/pickup_test_tube.yaml \
  --contract robots/arx/manifests/task7_model_a.yaml \
  --host 127.0.0.1 \
  --port 5581 \
  --seed 17 \
  --chunks 1 \
  --output runs/arx_pickup_test_tube/live_zeva_seed17_osmesa_smoke
```

Zeva returns 32 absolute 14D targets. The current task manifest executes the
first 16, then stops because `--chunks 1` was requested.

The output directory must not already exist. Use a new output name for every
run.

## 3. Run four closed-loop chunks

After the smoke test succeeds:

```bash
MUJOCO_GL=osmesa \
XDG_CACHE_HOME=/tmp/zetta-arx-cache \
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  scripts/deployment/run_arx_camera_chunk.py \
  --scene runs/arx_pickup_test_tube/dark_silver_estimated_top_scene \
  --mapping /data4/zhengyikai/Zeva_arx/assets/ac_one/ac_one_14d_mapping.json \
  --task robots/arx/manifests/pickup_test_tube.yaml \
  --contract robots/arx/manifests/task7_model_a.yaml \
  --host 127.0.0.1 \
  --port 5581 \
  --seed 17 \
  --chunks 4 \
  --output runs/arx_pickup_test_tube/live_zeva_seed17_osmesa_4chunks_rerun
```

This is receding-horizon execution. After each 16 executed actions, the client
uses the latest simulator state and three camera views to request a new 32-step
prediction. Four chunks therefore execute at most 64 actions, unless the task
terminates earlier.

Do not use an unbounded chunk count while validating action scale. The current
filters limit motion per step, but they do not impose an authoritative absolute
ARX joint envelope.

## 4. Run without the live model

This validates scene loading, camera rendering, action processing, MuJoCo
stepping, evaluation, and video generation without requiring the server:

```bash
MUJOCO_GL=osmesa \
XDG_CACHE_HOME=/tmp/zetta-arx-cache \
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  scripts/deployment/run_arx_camera_chunk.py \
  --scene runs/arx_pickup_test_tube/dark_silver_estimated_top_scene \
  --mapping /data4/zhengyikai/Zeva_arx/assets/ac_one/ac_one_14d_mapping.json \
  --task robots/arx/manifests/pickup_test_tube.yaml \
  --contract robots/arx/manifests/task7_model_a.yaml \
  --seed 17 \
  --chunks 1 \
  --offline-test \
  --output runs/arx_pickup_test_tube/offline_smoke_osmesa
```

## 5. Inspect the result

The main visual result is:

```text
<output>/three_view.mp4
```

Each run also contains:

- `front_rgb.mp4`, `left_rgb.mp4`, and `right_rgb.mp4`;
- `*_frame0.png`, the pre-video-encoding camera frames;
- `raw_actions.npy`, all 32-step chunks returned by Zeva;
- `executed_actions.npy`, the filtered actions actually sent to MuJoCo;
- `audit.json`, run dimensions, timing, hashes, and server metadata;
- `diagnostic_trace.jsonl`, durable per-stage execution diagnostics;
- `native_fault.log`, Python stacks for supported native fatal signals.

Quick checks:

```bash
cat runs/arx_pickup_test_tube/live_zeva_seed17_osmesa_smoke/audit.json
tail -30 runs/arx_pickup_test_tube/live_zeva_seed17_osmesa_smoke/diagnostic_trace.jsonl
```

## Rendering backend

Use `MUJOCO_GL=osmesa` for the current host. EGL previously initialized and
rendered successfully, but later returned corrupted raw RGB frames and aborted
inside right-camera rendering. This happened before video encoding. OSMesa has
been validated with all three cameras and changes only the rendering backend;
it does not change the state, action, task, or Zeva protocol semantics.

EGL may be re-enabled only after a clean three-camera probe produces valid
`*_frame0.png` files. It is faster, but correctness takes precedence during
integration validation.

## Current action semantics

- Input state: 14D `[left 7D, right 7D]`; the task zeros the left 7D model
  state.
- Output: `32×14` raw absolute joint-position targets, not deltas.
- Execution: first 16 predictions per chunk, configured by
  `robots/arx/manifests/pickup_test_tube.yaml`.
- Control: task locks, low-pass filtering, per-step limits, jump checks, and
  four 60 Hz commands per policy action.
- Grippers: hardware-semantic values are mapped to MuJoCo finger positions;
  finite simulated readback is projected into the representable hardware
  interval.
