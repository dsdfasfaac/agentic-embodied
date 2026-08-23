# Exact mid-episode snapshots for RoboCasa

`robots.robocasa.snapshot` saves enough MuJoCo, RoboSuite, and controller state
to continue a RoboCasa episode from a keyframe without replaying its action
prefix. The intended uses include counterfactual rollouts, critic branches,
failure recovery, and deterministic trajectory regression.

## What a snapshot contains

A snapshot has two files: a compressed, pickle-free `NPZ` payload and a JSON
manifest. It contains:

- MuJoCo `mjSTATE_INTEGRATION`, including time, all object/robot/base/torso
  positions and velocities, actuator state, warm-start accelerations,
  controls, applied forces, equality activation, mocap, userdata, and plugin
  state;
- mutable `MjModel` values used by RoboCasa scene, dynamics, material,
  lighting, camera, equality, tendon, and actuator randomization;
- capture-time derived `MjData` caches such as body/geom/site/camera
  transforms, accelerations, constraint forces, and sensor data;
- numeric Python state owned by the Gym wrapper, task environment, composite
  and part controllers, grippers, recent-value buffers, and observables.

Large immutable texture and mesh payloads are fingerprinted rather than copied
into every snapshot. Restore rejects a model with different topology, names,
dimensions, or static asset bytes.

For the verified PickPlaceCounterToCabinet scenes, a compressed schema-v2
snapshot was approximately 0.5 MB plus a 0.3 MB JSON manifest. The static
texture/mesh payload for the same model was hundreds of megabytes and was not
duplicated.

## API

```python
from robots.robocasa.snapshot import restore_snapshot, save_snapshot

# Save keyframe.npz and keyframe.json.
save_snapshot(env, "/path/to/keyframe")

# Construct and reset the same RoboCasa model before restoring it.
fresh_env.reset(seed=episode_seed)
restore_snapshot(fresh_env, "/path/to/keyframe")

observation, reward, terminated, truncated, info = fresh_env.step(next_action)
```

Restore intentionally follows this order:

1. validate the model and static asset fingerprint;
2. restore mutable `MjModel` values;
3. apply `mjSTATE_INTEGRATION`;
4. run `mj_forward`;
5. re-pin warm-start and capture-time derived `MjData` caches;
6. restore Python/controller/observable state;
7. refresh GPU texture/mesh resources and discard cached RGB renderers.

Calling a force-updated observation method between restore and the first
action can mutate observable timers. The regression script therefore compares
the unadvanced landing state directly and starts both videos after the first
common tail action.

## Cross-process regression

Run the verifier from the repository root in a RoboCasa environment:

```bash
PYTHONPATH=/path/to/robosuite:/path/to/robocasa:$PWD \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
python scripts/experiments/robocasa_snapshot_resume_check.py \
  --task PickPlaceCounterToCabinet \
  --split target \
  --seed 101 \
  --keyframe 96 \
  --tail-steps 112 \
  --actions /path/to/trajectory/actions.jsonl \
  --out /path/to/results/seed_101_k96
```

The record subprocess replays the prefix, saves a snapshot, and executes the
tail. A fresh resume subprocess resets the environment, restores the snapshot
without replaying the prefix, and executes the same tail actions. Outputs
include numerical traces, state and video verdicts, per-camera videos, and a
side-by-side `replay_vs_snapshot.mp4`.

Snapshot success requires bit-exact landing and continuation state. Raw RGB
and decoded MP4 equality are reported independently because GPU rasterization
can produce small wrist-camera differences even when every physics, geometry,
and camera-transform scalar is bit-exact.

## Verified target trajectories

The implementation was evaluated on successful target-split
`PickPlaceCounterToCabinet` trajectories. Snapshots were taken immediately
after the object was lifted, and the original tail was executed through task
success:

| Seed | Keyframe | Tail steps | End event | State result |
|---:|---:|---:|---|---|
| 101 | 96 | 112 | success at step 208 | bit-exact |
| 112 | 72 | 93 | success at step 165 | bit-exact |
| 133 | 56 | 109 | success at step 165 | bit-exact |
| 146 | 82 | 139 | success at step 221 | bit-exact |

Across these cases, the compared state included all 2,067 integration values,
101 positions, 98 velocities, accelerations, warm-start state, controls,
constraint forces, and body/geom/site/camera transforms. Every landing and
tail comparison had a maximum absolute error of zero.
