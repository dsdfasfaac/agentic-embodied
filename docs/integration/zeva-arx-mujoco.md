# Zeva/Cosmos3-Edge ARX X5 integration plan

The ordered code changes, environment commands, and verification gates are in
[`zeva-arx-implementation.md`](zeva-arx-implementation.md).

Status: final implementation plan based on `../Zeva_arx`, reviewed 2026-09-03.

Checkpoint update (2026-09-04): `../ckpt` is a separate Model A export. Its
contract is raw 14-D, 32-action output, 15 Hz, JSON prompt, `concat_view`, and
`replan_execute_steps=16`. Use `robots/arx/manifests/task7_model_a.yaml` and
execute the first 16 actions before replanning. Do not substitute the legacy
`arx_task7`/ID 17 profile. The checkpoint server implementation is deferred;
see the implementation status note below.

Implementation note (2026-09-03): the authoritative robot archive is
`../ARX_Model/AC one/URDF/AC one.7z`, SHA256
`d238e12107afbef5d0eca4cc36d1bf37ffdcef27e27a28a5fd30f35265e53ea0`.
That digest exactly matches `../Zeva_arx/assets/ac_one/SOURCE.json`; the sibling
MJCF is therefore used as the executable MuJoCo conversion of this archive.
The Task7 model server is supplied by
`../cosmos3-edge-arx5-inference/third_party/cosmos-framework`; this bundle can
replace `Zeva_arx` as the inference/runtime reference because it also contains
the protocol, ARX mapping, real2sim runner, and hardware runtime. Zetta launches
its Task7 server through `scripts/deployment/start_zeva_arx_task7_server.sh`.
The bundle's default launcher targets the older normalized ARX5 model and must
not be used for this checkpoint. Remaining deployment inputs are the Wan2.2 VAE,
a compatible Cosmos Python/CUDA environment, a working GPU driver, and final
checkpoint hashes.

## Decision

Connect Zetta's MuJoCo environment to the existing ARX Cosmos3-Edge server over
its synchronous ZMQ protocol. Keep simulator and inference in separate Python
environments and processes:

```text
Zetta planner -> Runtime Gateway
  -> MuJoCo EnvWorker -> ARX X5 Gymnasium environment
  -> RolloutWorker -> Cosmos3EdgeArxPolicyCore
                     -> ZMQ tcp://127.0.0.1:5581
                     -> Cosmos3-Edge ARX checkpoint
```

Do not install Cosmos in the Zetta/MuJoCo environment. `../Zeva_arx` already
defines a stable process boundary, while Cosmos and Zetta have independent
Python/CUDA/PyTorch/MuJoCo/NumPy constraints. Zetta needs only `pyzmq`,
`msgpack`, and NumPy to call the server.

The first target is **ARX Task7 s4000** on port 5581: domain
`arx-task7-x5` (ID 17), JSON prompt, server-side `concat_view`, raw 14-D
actions without stats, 15 Hz conditioning, and a 32-action response. Do not
substitute the older six-task chemistry model on port 5580; it has a different
domain, view path, and mean/std action normalization.

## Pinned sources

Preserve these contracts from `../Zeva_arx`:

- `COSMOS3_EDGE_ARX5_INFERENCE.md`: deployment and model contract;
- `cosmos3_edge/protocol.py`: ZMQ request/response encoding;
- `cosmos3_edge/control.py`: model-state preparation, locks, filtering, limits;
- `cosmos3_edge/config.py` and `cosmos3_edge/tasks/`: seven task definitions;
- `cosmos3_edge/real2sim_pipeline.py`: 14-D MuJoCo mapping and validation;
- `assets/ac_one/ac_one_14d.xml` and `ac_one_14d_mapping.json`.

Reuse or vendor the dependency-light protocol, control, task, and mapping code
with license notices. Do not depend on an ambient sibling `PYTHONPATH`. Record
the source commit and SHA-256 of checkpoint, inference config, MJCF, mapping,
task manifest, and copied adapters in each run artifact.

## Exact model interface

The server uses ZMQ REQ/REP. Msgpack carries NumPy arrays as `.npy` bytes.
Startup calls `ping` and `get_modality_config`; inference sends:

```python
{
  "endpoint": "get_action",
  "data": {
    "schema_version": 1,
    "request_id": "<uuid>",
    "observation": {
      "observation.images.front_rgb": front[None, None],
      "observation.images.left_rgb": left[None, None],
      "observation.images.right_rgb": right[None, None],
      "observation.state": state[None],
      "observation.timestamps_ns": np.asarray(
          [[front_ns, left_ns, right_ns, state_ns]], dtype=np.int64),
    },
    "prompt": [instruction],
    "options": {"executed_action_steps": 32},
  },
}
```

| Input | Contract |
|---|---|
| Cameras | `front_rgb`, `left_rgb`, `right_rgb`, fixed order |
| Images | each contiguous `uint8 [240,320,3]`, RGB |
| Composition | none client-side; Task7 server makes its trained 360x320 mosaic |
| State | finite contiguous `float32 [14]`, raw absolute joint positions |
| Instruction | non-empty canonical task instruction |
| Timestamps | four nanosecond values shaped `[1,4]` |

Initially reproduce the current client behavior of stamping all four modalities
at request time. True acquisition timestamps require a coordinated server and
regression-test update.

State/action order is:

```text
0..5 left_joint1..6; 6 left_gripper;
7..12 right_joint11..16; 13 right_gripper
```

Gripper state is canonicalized to `[-2*pi,0)` before inference. A task may zero
an inactive arm or override a gripper in **model state only**. Physical locks
are applied separately to outputs. Hardware grippers use `[-3.4,0.0]` open to
closed. Each scalar drives two MuJoCo finger joints/actuators, mapped to
`[0.044,0.0]` metres by the explicit inverse affine mapping. Never infer the
mapping from XML order.

The response is `(action_dict, metadata)`. `action_dict["action"]` is finite
`float32 [32,14]`, optionally `[1,32,14]` before removing batch dimension.
Validate the entire block even if a future task executes a shorter prefix.

At connection, require continuous raw positions and validate when reported:
14 action channels, 32 steps, camera shape `[240,320,3]`, resolution `"480"`,
domain name/ID, raw normalization, and 15 Hz. Add missing identity fields to
server modality/readiness metadata before acceptance. Never guess or apply the
port-5580 normalization client-side.

## Tasks and terminal conditions

The initial registry is the seven configs in `../Zeva_arx/cosmos3_edge/tasks`:
`pickup_test_tube`, `place_beaker`, `pour_water`, `prepare_salt_solution`,
`titration`, `weigh`, and `extraction` (IDs 0-6). Each pins instruction,
14-D start state, executed prefix, maximum steps, locks, model-state overrides,
filter coefficients, step/jump limits, gripper offsets, and publishable arms.
All current configs execute all 32 returned actions.

Create immutable local `zetta_arx_task_v1` manifests derived from `TaskConfig`:

```yaml
schema_version: zetta_arx_task_v1
task_id: 0
name: pickup_test_tube
instruction: Pick up test tube with the pink label.
start_state: [<14 pinned values>]
execution_steps: 32
max_steps: 600
control: {<locks, overrides, filters, limits, offsets>}
success: {evaluator: pickup_test_tube_v1, parameters: {...}}
failure: {evaluators: [safety_v1, unrecoverable_v1]}
```

The planner selects a registered task but cannot alter its start state, safety
settings, or success rule. Put the canonical language in
`ResetSpec.instruction`; pass task name and randomized initialization through
`ResetSpec.options`.

Each task also references one or more immutable starting-scene bundles exported
by `../Real2Sim`. The bundle and selection interface is specified in the
implementation runbook. Zetta never resolves a mutable "latest attempt".

Every task needs an independent simulator evaluator in
`robots/arx/tasks/<task>.py`:

```python
info = {
  "is_success": success,
  "termination_reason": "success" | "safety" | "unrecoverable" | None,
  "task_progress": progress,
  "task_id": task_name,
}
terminated = success or terminal_failure
truncated = control_step >= task.max_steps
```

Use `success_mode: info_key`, `success_info_key: is_success`. Success must also
set `terminated=True`, or Zetta's current chunk loop continues. Runtime is the
authority; model metadata, reward, appearance, and planner judgement are not.
Implement `pickup_test_tube` first and freeze its object names, reset
distribution, lift/contact/hold thresholds, and failure rules before adding the
other six.

## ARX MuJoCo environment

`../Zeva_arx` supplies a chunk preview/safety gate, not a complete Zetta env: it
resets from measured state, processes one chunk, and has no general success
contract. Build `robots/arx/environment.py` around the same MJCF,
`MujocoMapping`, and `ActionProcessor` semantics.

The Gym action space is raw policy `Box([14])`. One Gym step is one 15 Hz target:

1. validate `[14]` and apply task locks;
2. apply arm/gripper low-pass filters and per-step limits;
3. enforce jump thresholds;
4. map grippers to paired MuJoCo fingers;
5. interpolate arm targets at 60 Hz with the same quintic curve, updating
   grippers at the policy tick;
6. advance MuJoCo for exactly 1/15 second.

Task command offsets belong only at the real hardware-output boundary; they are
not model actions and should not distort simulated policy state unless a pinned
calibration test shows the simulator models that hardware offset.

Do not reset between chunks. Subsequent policy requests use simulated measured
joints and newly rendered views. This differs from the real-robot preview,
which exits after one chunk if live observations did not advance. Set generic
Runtime `clip_actions: false`; ARX validation/filtering owns action semantics.

Extend the MuJoCo session with backward-compatible `render_views()` and config:

```yaml
camera_names:
  main: front_rgb
  wrist: left_rgb
  extra: [right_rgb]
```

Construct the common observation as front=`main_image`, left=`wrist_image`,
right=`extra_view_images[0]`, with semantic names, 14-D measured/model state,
and four timestamps in `extras`. The policy resolves by semantic name, resizes
each to 320x240 RGB, and never makes the mosaic. Pin simulated camera intrinsics,
extrinsics, crop, channel order, and resolution against the real cameras.
Privileged evaluator state must not enter policy state or requests.

PlaceBeaker's RGB-D reconstruction aligns a live beaker into preview simulation.
In a native simulated testbed, initialize/evaluate MuJoCo object state directly.

## Zetta policy backend

Add `rollout_runtime/backends/cosmos3_edge_arx_policy.py` with a strict config:

```python
host = "127.0.0.1"; port = 5581; timeout_s = 300.0
action_dim = 14; served_action_steps = 32; conditioning_fps = 15.0
domain_name = "arx-task7-x5"; domain_id = 17
checkpoint_sha256 = "<required>"
modality_contract_sha256 = "<required>"
```

`Cosmos3EdgeArxPolicyCore` implements the existing persistent policy lifecycle:

- `load`: create one client and verify modality/model identity;
- `infer_batch`: serve requests serially because ZMQ REQ and server are sync;
- `update_weights`: reject hot swaps; restart the worker for a new checkpoint;
- `close`: close socket/context.

For each request: decode three views, read named measured state, apply
`prepare_model_state(task)`, choose the override or canonical instruction, call
`predict`, validate `[32,14]`, and return the task execution prefix. The env,
not policy backend, applies physical locks/filtering. On timeout/protocol error,
discard and reconnect the REQ socket before another request.

Register backend `cosmos3_edge_arx_remote`, family `cosmos3_edge_arx`, and
policy ID `cosmos3_edge_arx_task7`. Add no WebSocket fallback, Panda transforms,
implicit padding/reordering, or normalization.

## Runtime preset

Add `rollout_runtime/config/presets/arx_task7_mujoco.yaml`:

```yaml
env_family: mujoco
env_config:
  provider: gymnasium
  env_id: robots.arx.environment:ZettaArxManipulation-v0
  env_kwargs:
    scene_path: /absolute/path/to/arx_testbed.xml
    mapping_path: /absolute/path/to/ac_one_14d_mapping.json
    task_manifest: /absolute/path/to/pickup_test_tube.yaml
  observation_mode: rgb_state
  render_mode: rgb_array
  render_backend: egl
  camera_names: {main: front_rgb, wrist: left_rgb, extra: [right_rgb]}
  image_width: 320
  image_height: 240
  action_dim: 14
  chunk_size: 32
  clip_actions: false
  max_episode_steps: 600
  process_isolation: true
  instruction: Pick up test tube with the pink label.
  success_mode: info_key
  success_info_key: is_success

rollout_worker:
  policy_id: cosmos3_edge_arx_task7
  policy_family: cosmos3_edge_arx
  policy_backend: cosmos3_edge_arx_remote
  device: remote
  dtype: bfloat16
  policy_config:
    host: 127.0.0.1
    port: 5581
    timeout_s: 300
    action_dim: 14
    served_action_steps: 32
    conditioning_fps: 15
    domain_name: arx-task7-x5
    domain_id: 17
    checkpoint_sha256: <fill when available>
    modality_contract_sha256: <fill from ready metadata>
```

Keep model horizon and executed prefix separate even though both are currently
32. Runtime may stop early on environment termination.

## Zeva memory scope

The inspected ARX path is a Cosmos3-Edge action-policy client/server. It exposes
no BIT/PIM/CTE request fields, attempt finalization endpoint, or memory response
metadata. Do not claim deployment-time Zeva memory is active merely because the
project/model is called Zeva.

Integrate the supplied WAM path first. When an ARX Zeva memory API exists, add a
separately versioned phase consuming actions confirmed as executed, attempt and
task-instance IDs, boundary observations, and Runtime terminal outcomes. That
requires post-chunk and episode-finalization hooks; pre-step `infer_batch`
cannot know which action prefix actually executed.

## File sequence

1. `robots/arx/`: task registry/manifests, control, mapping, Gym env, first task
   evaluator.
2. `rollout_runtime/backends/mujoco_session.py`: multi-view rendering across the
   spawned proxy.
3. `rollout_runtime/backends/mujoco_env.py`: named views/state and reset options.
4. `rollout_runtime/backends/cosmos3_edge_arx_policy.py`: strict ZMQ bridge.
5. Backend/config registration and `arx_task7_mujoco.yaml`.
6. `robots/arx/toolkit.py` and prompts: observe/run/finish with Runtime verdict.
7. Contract, controller, environment, and optional checkpoint-backed tests.

## Acceptance gates

1. **Identity:** hash all artifacts; readiness says Task7/domain 17/raw/15 Hz,
   three 240x320 views, and 32x14; both environments pass `pip check`.
2. **Protocol:** run `../Zeva_arx/tests/task7_server_smoke.py`; golden-test the
   serialized Zetta request and forced-timeout reconnect.
3. **Controller parity:** all seven task transforms match `../Zeva_arx`; mapping
   round-trips and paired fingers pass; identical initial state/chunk produces
   identical processed targets to `Real2SimPipeline.simulate`.
4. **Environment:** deterministic reset/frames; continuous state across chunks;
   one step is 1/15 s with 60 Hz interpolation; privileged state is absent from
   requests; success, safety, and time limit yield correct terminal flags.
5. **End to end:** inference-only, one chunk, then a full fixed-seed
   `pickup_test_tube` episode. Audit requested/returned/processed/executed counts
   and archive hashes, metadata, seed, terminal verdict, and video. Repeat with
   deterministic server seed and compare request/action hashes.
6. **Sim/real parity:** compare safe recorded trajectories for order, units,
   gripper direction, end-effector motion, camera crop/orientation, timing, and
   locks. Simulation success never authorizes real motion; retain the sibling
   repo's explicit real-execution gates and hardware safety supervision.

## Checkpoint arrival

1. Export checkpoint with the Task7 deployment config and hash all shards/index.
2. Launch `start_cosmos3_edge_task7_server.sh` on 5581.
3. Archive ready/modality metadata and hash the canonical metadata.
4. Run the Task7 server smoke test.
5. Fill `checkpoint_sha256` and `modality_contract_sha256` in the preset.
6. Run acceptance gates 2-5 in order.

No design choice remains checkpoint-dependent. Only identity, server
compatibility, resource consumption, and numerical behavior remain to measure.
