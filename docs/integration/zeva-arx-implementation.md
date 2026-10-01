# Zeva/Cosmos3-Edge ARX X5 step-by-step implementation

This runbook turns `zeva-arx-mujoco.md` into an ordered implementation and
verification sequence. It targets the Task7 Model A service in
`../cosmos3-edge-arx5-inference`, checkpoint `../ckpt1`, and a simulated ARX
AC one/X5 embodiment in a custom MuJoCo scene.

## Current implementation summary (2026-09-06)

Implemented on branch `dev-yikai`:

- strict Task7 model/task contracts (`arx_task7`, domain ID 17, three RGB
  cameras, raw 14-D state/action, 32-step horizon, 15 Hz);
- immutable Real2Sim starting-scene bundle validation, deterministic selection,
  ARX MJCF composition, reset-state preservation, and provenance hashes;
- ARX action mapping, paired-finger conversion, 60 Hz MuJoCo command stepping,
  and 16-action execution followed by replanning;
- pickup success, hold/contact, workspace, drop, timeout, and safety evaluators;
- ZMQ/MsgPack Cosmos client, modality handshake, finite-shape checks, socket
  recovery, and the persistent `cosmos3_edge_arx_remote` Runtime backend;
- `arx_ac_one` Runtime environment provider and front/left/right observation
  bridge;
- Zetta Agentic task adapter/toolkit, observation-before-motion guard, and
  Runtime-authoritative terminal result;
- checkpoint-specific Runtime preset and deployment launcher;
- Task7 Cosmos server made inference-only: it no longer imports LeRobot,
  PyArrow, dataset adapters, the offline evaluator, or the training dataloader;
- compatibility for self-contained HF exports: the processor is loaded from
  `ckpt1/model`, `WAN_VAE_PATH` replaces the training-host VAE path, and absent
  optional semantic-memory fields default to the disabled legacy behavior.

`../ckpt1` is the accepted checkpoint. Every model artifact verifies against
its recorded SHA-256; only `checksums/SHA256SUMS` fails its own self-referential
entry. The server has been confirmed by the operator to start successfully.
The synthetic checkpoint smoke test returned a finite `[32,14]` chunk in
13.049 s with state-echo error 0.00338 (limit 0.05). The server's generic model
service initially labelled the identity/no-stats path `minmax`; the Task7
wrapper now reports its actual external convention as `raw`. Restart the server
after this patch before running Zetta's strict modality handshake. The focused
Zetta ARX suite has passed locally; scene-backed rollout gates listed below
remain to be executed and archived.

### Current Real2Sim starting scene

The first scene is `/data4/zhengyikai/Real2Sim/runs/tubes_2_mujoco_modified`.
It is a successful MuJoCo Real2Sim run containing `tube_stand_01` and three
upright tubes (`tube_01`, `tube_02`, `tube_03`). The `PickUpTestTube` manifest
selects scene ID `tubes_2_mujoco_modified` and targets `tube_01` (the
leftmost/pink-labelled tube). The run is immutable and must be wrapped in a Zetta
scene-bundle manifest hashing its layout, assembly report, final scene, XML,
compiled model, and settled state before ARX composition.

## Agentic semantic audit (2026-09-03)

The ARX integration follows Zetta's existing reBot Agentic structure:

- `robots.arx` exposes the required `get_env_spec()` and `get_toolkit()` hooks,
  so `zetta --env arx` resolves through the normal environment registry;
- the planner sees only contract inspection, three-view observation, one
  high-level Zeva execution skill, and `finish`; it never receives a direct
  action, simulator-step, task-state, or evaluator-configuration tool;
- the execution skill rejects motion until an observation was returned and is
  one-shot for the episode;
- every action is obtained and applied through `RuntimeGateway.policy_step`;
  the adapter does not call MuJoCo `step` or the ZMQ client directly;
- terminal truth in `agentic-summary.json` is copied from `StepResult.success`,
  `terminated`, and `truncated`, never inferred from images or policy metadata;
- initialization and shutdown use the standard EnvSpec/Toolkit lifecycle.

One pre-existing Zetta-wide limitation remains: `ApiAgentLoop`, the Codex
recorder, and the Claude Code recorder recognize a `finish` call from its input
arguments before (or independently of) the toolkit result. Consequently, the
ARX toolkit's guarded `finish` handler rejects premature success for direct
tool/MCP callers, but cannot retroactively prevent every planner recorder from
storing the attempted success arguments. This integration does not change that
global planner behavior. Downstream acceptance must use the Runtime-owned
`agentic-summary.json` verdict, not the planner's `finish.status` alone. A
repository-wide fix should make every planner promote finish only when the
actual tool result contains `_finish: true`.

The work is divided into independently testable changes. Steps 1-9 do not need
the final checkpoint. Steps 10-12 require a running Cosmos3-Edge server.

## Definition of done

The integration is complete when Zetta can reset a registered ARX task, produce
three correctly named RGB observations plus a 14-D model state, request a
32x14 action chunk from the Task7 server, execute the manifest-approved 16-step
prefix at the correct simulated rate, stop on authoritative success/failure/time limit, and
write a reproducible episode artifact containing all contract hashes.

Deployment-time BIT/PIM memory is not part of this milestone because the
current ARX server exposes no memory lifecycle protocol.

## 1. Freeze external contracts and fixtures

### Code and artifacts

Create:

```text
robots/arx/
  __init__.py
  contracts.py
  manifests/
    pickup_test_tube.yaml
tests/fixtures/arx/
  modality_task7.json
  request_input.npz
  expected_request.msgpack
  action_chunk.npy
```

`robots/arx/contracts.py` should define strict dataclasses:

- `ArxModelContract`: domain, camera order/shapes, state/action dimensions,
  model horizon, policy frequency, and normalization mode;
- `ArxTaskManifest`: instruction, start state, execution prefix, maximum steps,
  locks, model-state overrides, filters, limits, offsets, and evaluator config;
- `ArxCameraSpec`: semantic name, size, RGB layout, and calibration ID.

Reject unknown YAML keys and validate every dimension/range. Pin these Task7
values:

```text
domain_name = arx-task7-x5
domain_id = 17
camera_names = front_rgb,left_rgb,right_rgb
camera_shape = 240,320,3
state_dim = action_dim = 14
model_horizon = 32
conditioning_fps = 15
normalization = raw
```

Copy task constants from `../Zeva_arx/cosmos3_edge/tasks`, preserving their
license. Start with `pickup_test_tube`; import the other six only after the
first end-to-end task passes.

Generate fixtures from recorded or synthetic safe inputs using the existing
`CosmosEdgeClient` encoder. `expected_request.msgpack` must exclude volatile
UUID/timestamp values or canonicalize them before comparison.

### Tests

Add `tests/test_arx_contracts.py`:

- accept the pinned Task7/model and pickup manifest;
- reject unknown fields, wrong camera order, dimensions, domain, normalization,
  non-finite start state, invalid horizons, and unsafe threshold values;
- verify the committed manifest matches the selected `../Zeva_arx` task config.

Gate: contract tests pass without MuJoCo, CUDA, ROS, or the checkpoint.

## 2. Define the Real2Sim starting-scene interface

Every task has one or more approved starting scenes produced by `../Real2Sim`.
Zetta consumes an immutable bundle, never a run's current attempt and never a
mutable `runs/<name>` directory by itself.

### Producer artifacts

A successful attempt or accepted randomized variant provides:

```text
agent/active_layout.json
scene/scene.xml
scene/assembly_report.json
output/final_scene.json
output/model.mjb
output/settled_state.npz
```

Require `final_scene.json` schema `1.0`, backend `mujoco`, status `succeeded`,
and valid scene/model/state paths. The active layout supplies stable logical
object IDs; the assembly report maps them to root bodies. The settled state is
for the **scene-only** compiled model, not a robot-composed model.

Attempt zero is the run root; later attempts are `attempts/attempt_NNN`.
Randomized variants are self-contained roots under
`randomization/variants/variant_NNNNNN[_try_NN]`. Both are allowed only through
an exact registered root and content digest.

### Consumer manifest and selection

Create `robots/arx/scene_bundle.py` and
`robots/arx/schemas/real2sim_scene_bundle_v1.schema.json`:

```yaml
schema_version: zetta_real2sim_scene_bundle_v1
scene_id: pickup_tubes_attempt_002
task_name: pickup_test_tube
source:
  real2sim_commit: <required>
  run_id: tubes_1_dr_test_d8aaff92
  kind: attempt                 # attempt | randomized_variant
  index: 2
  retry: 0
  bundle_root: /absolute/immutable/path/to/attempt_002
artifacts:
  active_layout: {path: agent/active_layout.json, sha256: <required>}
  assembly_report: {path: scene/assembly_report.json, sha256: <required>}
  final_scene: {path: output/final_scene.json, sha256: <required>}
  scene_xml: {path: scene/scene.xml, sha256: <required>}
  model_mjb: {path: output/model.mjb, sha256: <required>}
  settled_state: {path: output/settled_state.npz, sha256: <required>}
objects:
  required_ids: [tube_stand_1, tube_1, tube_2, tube_3]
  target_ids: [tube_2]
composition:
  frame_transform: real2sim_world_to_arx_world_v1
  robot_source_archive: /data4/zhengyikai/ARX_Model/AC one/URDF/AC one.7z
  robot_source_archive_sha256: d238e12107afbef5d0eca4cc36d1bf37ffdcef27e27a28a5fd30f35265e53ea0
  robot_xml: /absolute/path/to/ac_one_14d.xml
  robot_xml_sha256: <required>
  mapping: /absolute/path/to/ac_one_14d_mapping.json
  mapping_sha256: <required>
  cameras: arx_task7_cameras_v1
reset:
  source: settled_state
  settle_after_composition_steps: 0
```

The authoritative robot source is `../ARX_Model/AC one/URDF/AC one.7z`; its
pinned hash matches `../Zeva_arx/assets/ac_one/SOURCE.json`. The Zeva MJCF is
the executable MuJoCo conversion of that URDF archive, with documented proxy
collision geometry.

For the verified `../ckpt1` Model A export, use `task7_model_a.yaml`: the
server returns 32 actions but the Runtime executes 16 and then replans. The
checkpoint's domain string is `arx_task7` and this deployment pins numeric
domain ID `17` (the ARX Task7 service identity). The first server modality
probe must verify that the running official Cosmos server reports the same
name/ID pair before production use.

Artifact paths are normalized relative POSIX paths. Reject absolute artifact
paths, `..`, escaping symlinks, missing/hash-mismatched files, duplicate scene
IDs, wrong tasks, and unknown fields. Robot/mapping paths are explicit external
deployment inputs and independently hashed.

Each task manifest adds:

```yaml
starting_scenes:
  selection: seeded_uniform       # fixed | seeded_uniform | explicit
  default_scene_id: pickup_tubes_attempt_002
  allowed_scene_ids:
    - pickup_tubes_attempt_002
    - pickup_tubes_variant_000004
```

`fixed` uses the default; `seeded_uniform` sorts IDs and selects deterministically
from the episode seed; `explicit` requires an allowed
`ResetSpec.options["scene_id"]`. Never accept a reset-time filesystem path.
Record selected scene ID and bundle digest in every observation/artifact.

### Offline preparation and composition

Implement:

```bash
python -m robots.arx.prepare_scene_bundle \
  --bundle-manifest /path/to/pickup_tubes_attempt_002.yaml \
  --output /immutable/cache/<bundle-digest>
```

It must:

1. verify producer metadata and hashes;
2. load scene XML and validate model/state dimensions;
3. cross-check required logical IDs and assembly-report body mappings;
4. compose ARX, three cameras, and evaluator sites/geoms with namespaced names;
5. apply a pinned coordinate-frame transform;
6. compile and build a source-to-composed state map by names;
7. restore settled scene state and initialize ARX from the task start state;
8. run `mj_forward`, optional pinned settling, and evaluator preflight;
9. write composed XML/MJB, reset state, state map, composition report, and a
   content-addressed output manifest.

Prefer named object/joint restoration from `final_scene.json`. If v1 lacks
enough names, prefix copying is permitted only when preparation proves and
records that the source `nq/nv` layout is an unchanged prefix. Runtime itself
must never assume prefix compatibility. The source `model.mjb` is a verification
artifact, not the final model containing robot/cameras/evaluators. Cache keys
include every source and composition hash. Never edit `../Real2Sim`.

At reset the environment loads only the prepared bundle, copies its reset
`qpos/qvel` and actuator state, and applies task-approved options. Initially,
use accepted Real2Sim randomized variants as distinct scene IDs rather than
adding an untracked second randomizer in Zetta.

### Tests

Add `tests/test_arx_real2sim_bundle.py`,
`tests/test_arx_scene_composition.py`, and a minimal fixture attempt. Test:

- attempt and randomized-variant validation;
- rejection of unfinished/wrong backend/schema, mutable selection, traversal,
  escaping symlinks, missing artifacts, and hash changes;
- run ID, logical ID, body mapping, and final-scene cross-checks;
- source XML/MJB/settled-state dimension validation;
- deterministic composition and restoration;
- robot state cannot overwrite objects and objects cannot overwrite robot;
- camera/evaluator name-collision rejection;
- two scenes for one task restore different intended object states;
- seeded selection is stable and registry-order independent;
- unregistered IDs/direct paths are rejected;
- preparation and episodes leave source bytes unchanged.

Gate: two prepared starting scenes for the first task select deterministically
and produce stable, correctly restored initial observations.

## 3. Port the ARX action and MuJoCo mapping logic

### Code

Create:

```text
robots/arx/control.py
robots/arx/mujoco_mapping.py
```

Port the dependency-light behavior from:

```text
../Zeva_arx/cosmos3_edge/control.py
../Zeva_arx/cosmos3_edge/real2sim_pipeline.py
```

Keep the following functions/classes API-compatible where practical:

- `as_action`
- `canonicalize_gripper`
- `prepare_model_state`
- `apply_task_locks`
- `ActionProcessor`
- `interpolate_commands`
- `MujocoMapping`

The mapping must use the explicit 14-channel order. Channels 6 and 13 must each
map to two finger joints and two position actuators. Do not copy ROS publishers,
viewer code, or real-execution authorization into the simulator environment.

Expose two separate operations:

```python
model_state = prepare_model_state(measured_state, task)
processed_action = processor.process(raw_model_action)
```

Do not apply real-hardware `gripper_command_offsets` to model actions or MuJoCo
unless a calibration-specific simulator manifest explicitly requests them.

### Tests

Add `tests/test_arx_control.py` and `tests/test_arx_mujoco_mapping.py`:

- differential-test functions against `../Zeva_arx` for all seven task configs;
- test gripper wrap at `-2*pi`, `0`, and multiple rotations;
- test locks and model-state overrides independently;
- test filter and step-limit state across a full 32-action chunk;
- reject NaN/Inf, incorrect shape, unsafe jumps, and out-of-range grippers;
- test hardware-to-finger and inverse mapping endpoints/midpoints;
- reject a mapping with only one finger or duplicated joint/actuator names.

Gate: identical initial state and raw chunk produce numerically identical
processed targets to the sibling implementation.

## 4. Implement the ARX Gymnasium environment

### Code

Create:

```text
robots/arx/environment.py
robots/arx/tasks/__init__.py
robots/arx/tasks/pickup_test_tube.py
```

Register `ZettaArxManipulation-v0`. Constructor arguments should be explicit:

```python
prepared_scene_bundle: str
mapping_path: str
task_manifest: str
camera_names: dict[str, str]
policy_hz: float = 15.0
command_hz: float = 60.0
```

On construction:

1. load and verify the prepared scene bundle, mapping, and task manifest;
2. resolve every required joint, actuator, body, geom, and camera by name;
3. validate actuator/joint types, limits, and paired fingers;
4. create `Box(shape=(14,))` in raw policy coordinates;
5. create a stable named observation schema.

On `reset(seed, options)`:

1. reject unknown options;
2. select an allowed prepared scene and restore its reset state;
3. apply task start state and only bundle-approved reset options;
4. reset the `ActionProcessor` from the actual simulated measured state;
5. clear terminal counters and evaluator history;
6. return named state and an initial evaluator result.

On each `step(raw_action)`:

1. validate the raw 14-D action;
2. process locks, filtering, per-step limits, and jump guards;
3. convert gripper channels to paired finger targets;
4. produce four 60 Hz subcommands using the ARX quintic interpolation;
5. advance MuJoCo for exactly 1/15 second total;
6. compute measured 14-D state from joint positions;
7. evaluate safety, task progress, and task success;
8. return Gymnasium's five-tuple.

Use a timestep/frame-skip combination whose simulated duration is exact within
floating-point tolerance; validate it at startup. State must continue between
policy chunks.

Implement the pickup evaluator from named simulator quantities. Its manifest
must pin at least the target object, allowed gripper, lift threshold, required
contacts, hold steps, workspace bounds, and unrecoverable-drop rule. Do not
reuse the viewer's `BEAKER LIFTED` diagnostic as a generic success signal.

### Tests

Add `tests/test_arx_env.py`, marked `mujoco` where necessary:

- deterministic reset for equal seed and divergence for selected different seeds;
- correct fixed, seeded, and explicit starting-scene selection;
- action/state width and stable named observation layout;
- exact simulated time per action and four interpolation substeps;
- continuity across two 32-action chunks;
- locks remain fixed at the episode initialization state;
- paired fingers move together with the correct inverse convention;
- malformed/unsafe action fails before physics advances;
- success requires the pinned hold duration and returns `terminated=True`;
- safety/unrecoverable failures terminate with distinct reasons;
- maximum steps returns `truncated=True`, not success;
- close is idempotent and releases renderer/native resources.

Gate: a deterministic scripted safe trajectory can reset, execute multiple
chunks, and terminate with an environment-owned verdict.

## 5. Add multi-camera support to the MuJoCo Runtime

### Code

Modify `rollout_runtime/backends/mujoco_env.py`:

- add optional validated `camera_names` with `main`, `wrist`, and `extra` keys;
- require unique non-empty camera names for RGB modes;
- include it in `session_config()`;
- encode named views into `Observation.main_image`, `wrist_image`, and
  `extra_view_images`;
- preserve `extras["camera_names"]`, named raw state, and modality timestamps;
- preserve the current single-`camera_name` path for existing presets.

Modify `rollout_runtime/backends/mujoco_session.py`:

- add `render_views() -> dict[str, np.ndarray]` to the local session;
- add a `render_views` command to the spawned RPC worker/proxy;
- validate each frame as contiguous `uint8 HWC RGB` at configured resolution;
- render all views on the owning session thread;
- retain `render()` as a compatibility wrapper for one camera.

The environment should expose named policy state through a small optional
method such as `policy_state()` or a structured raw observation. Avoid parsing
meaning from the generic flattened state vector.

### Tests

Extend `tests/test_mujoco_agentic_integration.py` or add
`tests/test_mujoco_multiview.py`:

- existing single-camera tests remain unchanged;
- local and spawned sessions return the same three semantic views;
- camera order is stable and duplicate/missing cameras are rejected;
- RGBA is reduced to RGB; wrong shape/dtype fails closed;
- encoded observations round-trip pixel-identically;
- process isolation owns and releases all render contexts.

Gate: all existing MuJoCo tests plus the new three-camera tests pass.

## 6. Implement the ZMQ protocol client locally

### Code

Create `robots/arx/cosmos_edge_client.py`, based on
`../Zeva_arx/cosmos3_edge/protocol.py`. It owns:

- NumPy/msgpack codec;
- `ping`, `get_modality_config`, and `get_action` requests;
- strict request/response validation;
- finite `[32,14]` response normalization;
- socket recreation after timeout, malformed reply, or REQ state failure.

Add server identity validation against `ArxModelContract`, not just the current
substring check for `continuous raw positions`. Compute a digest from canonical
modality metadata. Do not retry an ambiguous completed inference automatically;
surface failure and reconnect for the next operation.

### Tests

Add `tests/test_arx_cosmos_edge_client.py` with an in-process fake ZMQ REP server:

- validate exact endpoint and request nesting;
- compare canonical payload with the fixture from Step 1;
- accept `[32,14]` and `[1,32,14]` responses;
- reject wrong domain/modality, normalized model, dimensions, NaN/Inf, and
  malformed metadata;
- force timeout and prove the following request uses a fresh socket;
- prove concurrent calls are serialized or rejected explicitly.

Gate: protocol tests pass without CUDA or Cosmos.

## 7. Add the persistent Zetta policy backend

### Code

Create `rollout_runtime/backends/cosmos3_edge_arx_policy.py` containing:

- `COSMOS3_EDGE_ARX_POLICY_FAMILY = "cosmos3_edge_arx"`;
- `Cosmos3EdgeArxPolicyConfig`;
- `Cosmos3EdgeArxPolicyCore` implementing `load`, `infer_batch`,
  `update_weights`, and `close`.

For each `InferenceRequest`:

1. resolve images by `Observation.extras["camera_names"]`;
2. decode front/left/right and resize each to 320x240 RGB;
3. read the named measured 14-D state;
4. load the episode's immutable task manifest;
5. apply only `prepare_model_state` before inference;
6. select `instruction_override` or observation instruction;
7. submit one synchronous ZMQ request;
8. validate the complete response and return the task execution prefix.

Process a batch serially. The environment applies output locks/filtering. Reject
hot weight updates. Include inference round-trip time, server metadata digest,
and raw/returned horizon in action metadata without treating them as success.

Modify `rollout_runtime/backends/__init__.py` to register
`cosmos3_edge_arx_remote`. Add the policy family to parameter/batching rules in
`rollout_runtime/core/policy_inference.py` with no model-specific mixable keys.
Update schema documentation and backend allowlists.

### Tests

Add `tests/test_arx_policy_backend.py`:

- load/close lifecycle and identity rejection;
- observation-to-server request conversion;
- task instruction override behavior;
- three-camera semantic order independent of storage order;
- state override without physical action locking;
- serial multi-request behavior and per-request failure isolation;
- returned execution prefix and metadata;
- unsupported hot update rejection.

Gate: fake-server policy tests and all existing policy backend tests pass.

## 8. Wire task reset and chunk termination

### Code

Allow the ARX env to receive a registered string task name and initialization
options through `ResetSpec.options`. Keep generic MuJoCo's integer `task_id`
restriction unchanged unless a typed multi-task API is added separately.

Verify `MujocoEnvCore._chunk_step_one` stops on either terminal flag and reports
the actual executed prefix. Add task ID, terminal reason, progress, processed
action diagnostics, and artifact hashes to per-step/chunk info in JSON-safe
form.

If the policy backend truncates the 32-action response to `execution_steps`,
ensure `PolicyRequest.actions_per_chunk` agrees. Reject conflicting requested
horizons instead of silently taking a different prefix.

### Tests

Add/extend tests for:

- instruction and task options survive reset;
- success on action 5 of 32 executes exactly five actions;
- timeout/safety failure preserves side-effect and executed-count reporting;
- a second step after termination is rejected;
- configured maximum counts physical actions, not inference calls.

Gate: early termination is observable and auditable through the Gateway result.

## 9. Add preset, toolkit, prompts, and artifacts

### Code

Create:

```text
rollout_runtime/config/presets/arx_task7_mujoco.yaml
robots/arx/task_adapter.py
robots/arx/toolkit.py
robots/arx/prompt_bundle.py
```

The preset follows `zeva-arx-mujoco.md`; checkpoint and modality digests are
required environment/deployment substitutions, never permissive empty defaults.

Expose planner tools:

- `observe_arx_scene`: read-only observation;
- `run_arx_policy`: run the bounded episode through Gateway policy steps;
- standard `finish`.

The adapter records:

```text
run-config.json
model-contract.json
task-manifest.json
scene-bundle-manifest.json
scene-composition-report.json
episode-summary.json
actions-returned.npy
actions-executed.npy
episode.mp4
```

The summary contains source/artifact hashes, seeds, server modality metadata,
inference timings, action counts, termination reason, progress, and Runtime
success. Prompts instruct the planner to accept only Runtime success.

### Tests

Add `tests/test_arx_agentic_integration.py` with fake env/policy and fake ZMQ:

- only intended tools are visible;
- observation precedes motion;
- all motion passes through Gateway `policy_step`;
- planner cannot override manifest safety/evaluator fields;
- artifact action counts and terminal verdict match Runtime results;
- failed Runtime verdict cannot be reported as success.

Gate: a CPU-only fake end-to-end agentic episode passes in CI.

## 10. Set up the two runtime environments

### Zetta/MuJoCo environment

Use a dedicated conda environment; do not reuse the Cosmos environment:

```bash
cd /data4/zhengyikai/Agentic-Embodied
conda create -n zetta-mujoco python=3.10 -y
conda activate zetta-mujoco
python -m pip install --upgrade pip
python -m pip install -e ".[test,ray,mujoco]"
python -m pip install pyzmq
python -m pip check
```

The current development checkout already has a working `zetta-mujoco` conda
environment. Re-run the install commands after dependency metadata changes;
do not create a repo-local venv.

Before EGL tests:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Keep scenes/checkpoints outside Git. Prepare and hash each immutable Real2Sim
bundle before launch; Runtime consumes only the prepared registry.

### Cosmos inference environment and server

Use the dedicated Python 3.13 environment in the self-contained inference
bundle. Do not install `cosmos-framework[train]`: its cloud/training dependency
set is unnecessary and has an `aioboto3`/modern `boto3` resolution conflict.

```bash
cd /data4/zhengyikai/cosmos3-edge-arx5-inference
conda create -y -p "$PWD/.venv" python=3.13 pip

COSMOS_PYTHON="$PWD/.venv/bin/python"
"$COSMOS_PYTHON" -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchcodec==0.10.0+cu130
"$COSMOS_PYTHON" -m pip install \
  --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
  -r requirements-runtime.txt
"$COSMOS_PYTHON" -m pip install \
  --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
  natten==0.21.6.dev6+cu130.torch210.gb300 \
  flash-attn==2.7.4.post1+cu130.torch210
"$COSMOS_PYTHON" -m pip install --no-deps -e third_party/cosmos-framework
"$COSMOS_PYTHON" -m pip check
```

Required inference artifacts:

```text
/data4/zhengyikai/ckpt1/model
/data4/zhengyikai/ckpt1/config/config.yaml
/data4/zhengyikai/cosmos3-edge-arx5-inference/models/Wan2.2_VAE.pth
```

Start the server through Zetta's Task7 launcher, not the bundle's legacy
`start_cosmos3_edge_arx5_server.sh`:

```bash
cd /data4/zhengyikai/Agentic-Embodied
export COSMOS_ARX_BUNDLE=/data4/zhengyikai/cosmos3-edge-arx5-inference
export COSMOS_ARX_PYTHON="$COSMOS_ARX_BUNDLE/.venv/bin/python"
export ZEVA_CHECKPOINT=/data4/zhengyikai/ckpt1/model
export ZEVA_CONFIG=/data4/zhengyikai/ckpt1/config/config.yaml
export EDGE_MODEL_PATH=/data4/zhengyikai/ckpt1/model
export WAN_VAE_PATH="$COSMOS_ARX_BUNDLE/models/Wan2.2_VAE.pth"
export CUDA_VISIBLE_DEVICES=0
export HOST=127.0.0.1
export PORT=5581
./scripts/deployment/start_zeva_arx_task7_server.sh
```

The launcher sets `COSMOS_TRAINING=0`. Preserve this: otherwise Cosmos imports
training/cloud backends and creates false dependency requirements. Server logs
are written below `../cosmos3-edge-arx5-inference/runtime/cosmos3_edge_arx_task7`.
If inference and rendering share one GPU, measure peak memory and retain enough
headroom; otherwise assign MuJoCo rendering and Cosmos to separate GPUs.

### Smoke checks

```bash
cd /data4/zhengyikai/cosmos3-edge-arx5-inference
PYTHONPATH="$PWD" \
  /home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  /data4/zhengyikai/Zeva_arx/tests/task7_server_smoke.py \
  --host 127.0.0.1 --port 5581
```

Then run the Zetta client contract probe before constructing MuJoCo.

## 11. Checkpoint-backed integration tests

Mark these `remote` so normal CI does not require CUDA:

1. **Server identity:** connect, archive modality metadata, and compare digest.
2. **Inference-only:** use the golden three images/state; require finite 32x14.
3. **One chunk:** reset MuJoCo, infer once, execute at most 32 actions, and save
   before/after frames plus raw/processed/executed actions.
4. **Continuous rollout:** perform at least two policy calls and prove the
   second request contains the advanced state and new renders.
5. **Full pickup episode:** stop only on env success/failure/time limit.
6. **Determinism:** with server deterministic seed and fixed env seed, compare
   canonical request and returned-action hashes across two fresh runs.
7. **Failure injection:** kill/restart server between operations; verify no
   environment step occurs for failed inference and reconnection succeeds.

Do not require task success for the first checkpoint smoke gate; require
protocol correctness, finite safe actions, correct timing, and bounded episode
termination. Task success becomes a separate model-quality acceptance result.

## 12. Evaluation and promotion

### Current readiness status

The server, checkpoint, VAE, CUDA environment, and Zetta client boundary are
now available. Server startup and configuration/path resolution are confirmed.
The remaining work is empirical integration validation rather than missing
core implementation.

### Still missing or untested

1. Archive a successful Task7 smoke request and modality response/digest.
2. Run one real checkpoint request and verify finite `[32,14]` raw actions,
   state-echo tolerance, latency, and deterministic repetition.
3. ~~Prepare at least one production Real2Sim scene bundle~~ **Done for the
   current validation scene**: `/tmp/tubes_2_arx_prepared` was composed from
   `Real2Sim/runs/tubes_2_mujoco_modified`; the MJB ABI mismatch was handled by
   recompiling from the source XML under MuJoCo 3.3.1. The report verifies the
   Real2Sim state prefix and composed dimensions (`nq=44`, `nv=40`). Production
   copies should still be placed in a durable artifact directory. Remaining:
   final logical body IDs, reset state, composition report, and calibrated
   camera names.
4. Validate the simulated front/left/right poses against the real Task7 camera
   convention and confirm the resulting 360x320 server mosaic visually.
5. **Environment gate is not camera-valid yet.** The current Real2Sim XML only
   contains diagnostic cameras (`overview`, `probe_front`, `probe_top`,
   `probe_close`). These must not be passed to Zeva as policy observations. The
   Zeva contract means `front_rgb` is the fixed top/scene camera and
   `left_rgb`/`right_rgb` are the left/right ARX wrist cameras. The ARX bundle
   currently provides only front-camera calibration metadata
   (`front_d405_rgbd_calibration.json`); it does not provide wrist calibration
   or MuJoCo camera bodies. Add calibrated top and wrist cameras to the
   composed scene (or provide an explicit camera-attachment config) before
   rendering policy inputs. The earlier EGL reset test with probe cameras is
   therefore only a renderer/physics sanity check, not a Zeva validation.
   Run one MuJoCo chunk: execute exactly the configured first 16 actions at
   15 Hz, record before/after observations, and prove no action clipping or
   channel permutation occurred.
6. Run two consecutive inference chunks and prove the second request uses the
   advanced simulator state and newly rendered images.
7. Run full fixed-seed pickup episodes and characterize success, safety-stop,
   timeout, and evaluator false-positive/negative rates.
8. Inject server timeout, malformed output, disconnect, and restart; verify a
   failed inference never advances MuJoCo and the next session reconnects.
9. Exercise the full Zetta planner/tool loop and compare its terminal record to
   Runtime-owned `agentic-summary.json`.
10. Run the broader repository test suite in dependency-capable environments;
    current evidence covers the focused ARX/Runtime tests, not all optional
    ROS, Torch, and simulator integrations.
11. Validate real ARX joint order, gripper ranges/sign, start pose, cameras,
    command timing, and hardware safety gates before any physical execution.

### Next steps

Execute these gates in order:

1. server smoke and modality archive;
2. deterministic inference-only request;
3. prepare/calibrate one `PickUpTestTube` Real2Sim scene;
4. one-chunk then two-chunk MuJoCo rollout;
5. full direct Runtime episode and failure injection;
6. full Agentic episode with semantic-result audit;
7. multi-scene fixed-seed evaluation campaign;
8. only after simulation acceptance, separately review and authorize the real
   ARX deployment path.

Use a versioned seed set and scene/task manifest. For every episode record:

- reset seed and initial-state digest;
- selected Real2Sim scene ID, source digest, and composition digest;
- all three input-view hashes per inference;
- model state, raw actions, processed actions, and actual executed prefix;
- inference and simulated-control timing;
- progress, contacts/safety events, terminal flags, and reason;
- video and complete artifact identities.

Promotion stages:

1. **Contract-ready:** Steps 1-9 and CPU/fake tests pass.
2. **Checkpoint-ready:** Task7 smoke and inference-only tests pass.
3. **Simulation-ready:** one-chunk and continuous-rollout gates pass.
4. **Task-ready:** pickup evaluator and fixed-seed campaign pass agreed metrics.
5. **Agentic-ready:** planner/tool episode produces the same Runtime verdict and
   audited artifacts as direct evaluation.

Real-robot execution is outside these promotion stages. MuJoCo success does not
authorize hardware motion; preserve the ARX runtime's explicit human gates and
hardware safety supervision.

### Camera configuration placeholder

`robots/arx/manifests/arx_task7_cameras.yaml` records the Zeva camera contract,
the shared front-camera intrinsics source, and the intended wrist parent links.
The wrist poses are intentionally `null` pending measurement relative to each
arm's joint 6. Runtime policy execution must reject this configuration until
both wrist poses (and the top-camera pose) are filled in and calibrated.

## Recommended commit order

Keep changes reviewable in this order:

1. ARX contracts, task manifest, and fixtures;
2. Real2Sim bundle schema/preparation and composition tests;
3. action processing and mapping with differential tests;
4. Gymnasium ARX environment and pickup evaluator;
5. generic MuJoCo multi-view support;
6. ZMQ client and fake server tests;
7. Runtime policy backend and registration;
8. reset/termination audit improvements;
9. preset, toolkit, prompts, and artifacts;
10. checkpoint metadata pins and remote test evidence.

Do not combine checkpoint-generated files, model weights, recorded datasets, or
run outputs with source commits.
