# ARX CandidateBundle real-robot execution

## Dodo verification state (2026-10-04)

The current checkout freezes the sample CandidateBundle, tool catalog,
real-input contract, hardware settings, controller-EE kinematics, and runner
limits under `robots/arx/manifests/real/`. The hardware file's SHA-256 is
`cdf5ebaa574f84bd04a1393d3b1b5f404294c60324d442368a5df3d7e0484a4e`;
the real-input contract SHA-256 is
`e0ff9f10ee16cab57ebef291da3ff979679f8fe383ed27a0d9f44b09b1da1bf8`.
`freeze_arx_picktube_inputs.py` regenerates the catalog/contract and
`freeze_arx_dodo_hardware.py` regenerates the hardware file from the 50 raw
PickTube episodes. Its provenance file records the exact source hashes.
The joint command bounds are a narrow envelope of recorded controller
feedback plus 0.05 rad; they are task bounds, not mechanical hard stops.

The ARX Task7 training dataset README on aigc31 explicitly defines
`observation.state` as 14D joint feedback and `action[t] = state[t+1]` as a
joint-position action proxy. PickUpTestTube zeroes the inactive left 7D.
The raw episode 000048 used for model smoke inference has
`action_mode=joint`; the general data collection guide's EEF default does
not apply to that episode or the Task7 training data.

The read-only `audit_arx_live_cameras.py` opened the three actual D405s at
640×480@15, validated their pinned serials/intrinsics, and saw the pink
label with valid aligned depth in five fresh synchronized frame sets. Label
depth was about 382–385 mm, depth MAD 1–2.5 mm, and camera skew 11–28 ms.
This determines a camera/left-base target point, but it is not yet a measured
target-to-gripper distance without fresh arm feedback.

The old simulation `robot_calibration.json` disagreed with recorded
controller `end_pos` by about 0.29 m and must not be used on dodo.
`calibrate_arx_right_fk_from_raw.py` fitted the base and controller-EE offset
from 40 raw episodes and tested against 10 held-out episodes. The pinned
`dodo_right_controller_ee_fk.json` had held-out position error P95 1.10 mm,
maximum 1.24 mm, and orientation error below 0.001°. The PickTube feature
provider now checks this FK against fresh controller `end_pos`, then applies
the nominal gripper tool offset from `ac_one_nominal_chain.json`. The physical
gripper contact point and live target distance still need an observed check.

The full bundle `--check-config` passed on dodo with `hardware_opened: false`.
The sample recovery compiled to five calls: one gripper, two EEF increments,
one reentry review, and one VLA continuation. The robot control processes
remain stopped, so synchronized live 14D feedback, task start-state match,
and motor command/arrival have not yet been verified. Use
`audit_arx_live_observation.py` for the next read-only check once the status
controllers are running; it never publishes a command.

## Direct inference and the H100 Model A checkpoint

Dodo already has a direct real-robot inference client in the separate
`/home/dodo/chenfu/inference` repository:
`x5_cosmos3_edge_pick_tube.py` calls `cosmos3_edge.cli`. Its
`run_zeva_task7_eval.sh` wrapper starts a Task7 model service and passes
`--yes` to the client, which can publish motor commands. The
`/home/dodo/chenfu/Agentic-Embodied` checkout instead has the
`run_arx_real_bundle.py` gateway/runner for CandidateBundle monitoring and
recovery. Both execute locally on dodo; the extra boundary is the bundle
gateway, not a network hop to another host.

The H100 export at
`/mnt/100T/users/dingxin/WAM/playground/packages/arx_model_a_5task_iter5000_20260817`
was copied to dodo's
`/mnt/hdd16t/chenfu/cosmos_models/arx_model_a_5task_iter5000_20260817`.
All 17 source files passed the package's SHA-256 list, excluding the list's
invalid self-referential entry. The original package was not edited.
`prepare_arx_model_a_dodo.py` creates a symlinked inference view with local
asset paths and disabled-memory compatibility fields for dodo's newer Cosmos
framework. `start_arx_model_a_dodo.sh` then serves it on dodo loopback port
5583. Its `start`, `status`, and `stop` operations do not open ROS or the robot
controller.

On 2026-10-04 this exact iter5000 package loaded and answered a Zetta
`CosmosEdgeClient` request using recorded PickTube episode 000048 RGB images
and its 14D state. The prediction was finite with shape `32×14` and a 0.83 s
round trip. This proves model loading and offline contract compatibility; it
does not validate physical actions or recovery. The model service was stopped
after the test, and the robot controller remained stopped. In particular, the
first predicted gripper coordinate differed from the recorded state by about
0.344 on the left and 0.140 on the right; those values need controller-unit
and limit checks before any motor execution.

## Dodo chemistry model (2026-10-04)

The supplied model checkpoint is
`/mnt/hdd16t/chenfu/cosmos_models/arx5_chemistry_edge_stride2_gbs256_8gpu_2250/iter_000002250_ema_bf16_hf`
on **dodo**. Run the model service and this repository's gateway/runner on
dodo, using loopback `127.0.0.1`; no cross-host inference connection is needed.
`scripts/deployment/start_arx_realdata_server_dodo.sh check` validates the
checkpoint shards, matching 14D mean/std statistics, runtime config, processor,
VAE, and Python installation without starting a service or robot controller.
The same script accepts `start`, `status`, and `stop` for the model service on
port 5580. Its environment variables allow replacing the dodo-specific paths.

This is a **RealData ARX5 chemistry** model, not the earlier Task7 checkpoint.
It uses `realdata_arx5`, internal mean/std normalization, a training-style
prompt, and continuous raw gripper coordinates. The present model contract
loader only accepts Task7 domain 17, raw normalization, and JSON prompts, and
the real runner defaults to Task7 port 5581. Do not pass this checkpoint to
`start_zeva_arx_task7_server.sh` or run robot motion through the Task7 contract.
The RealData model contract and runtime compatibility need implementation and
verification before a full real-robot episode.

The alternative H100 package
`/mnt/100T/users/dingxin/WAM/playground/packages/arx_model_a_5task_iter5000_20260817`
is Task7 Model A: 14D raw absolute actions, 32-step horizon, 15 Hz, and
`concat_view`. It is a closer match for the current gateway model contract.
For dodo-local serving, copy the immutable package to
`/mnt/hdd16t/chenfu/cosmos_models/arx_model_a_5task_iter5000_20260817`,
verify `checksums/SHA256SUMS`, then run
`/home/dodo/chenfu/cosmos-framework-edge-arx5/.venv/bin/python
scripts/deployment/prepare_arx_model_a_dodo.py`. This creates a small runtime
view with local processor/VAE paths while leaving the source package and its
checksums unchanged. `scripts/deployment/start_arx_model_a_dodo.sh start`
serves it on dodo loopback port 5583 with Dynamo disabled; stop it with the
same script's `stop` argument. The Zetta runner must receive
`--zeva-host 127.0.0.1 --zeva-port 5583` when this model is selected.

Dodo's older Task7 s4000 service also returned a 32×14 finite prediction from
recorded three-camera images after `TORCHDYNAMO_DISABLE=1` was applied.

On 2026-10-04, dodo's checkpoint `SHA256SUMS` passed for all seven listed
files. The isolated model service loaded on GPU 0, answered `ping` and
`get_modality_config` at `127.0.0.1:5580`, and was then stopped. Its reported
action shape was `[1,32,14]`, `action_normalization` was `meanstd`, and its
prompt was a training-style instruction. No robot controller was started.
The sample CandidateBundle from aigc31's
`runs/arx_privileged_test_20260929_061734/campaign/bundle.json` is now
tracked byte-for-byte as `robots/arx/manifests/real/sample_picktube_candidate_bundle.json`.
The separate chemistry RealData contract remains unfinished; the Task7 Model A
path has the frozen inputs described above. Live controller feedback and
physical motion checks remain necessary before a full episode.

The real deployment entry point is `python -m scripts.deployment.run_arx_real_bundle`.
It owns one gateway process and one episode. Run it on the host that has the ARX
controller, RealSense devices, ROS2 or the official SDK, and a reachable Zeva
server. The runner writes `result.json`, gateway journal data, tool requests and
results, and `recovery/<incident>-<call>.json` with the observation before and
after every recovery call. The gateway also rejects out-of-order or altered
bundle calls; `arx.finish` remains available to stop an episode.

For the structured sample, `arx.move_eef`'s 2 cm request becomes two 1 cm tool
calls. The compiled recovery allows exactly five decisions and reserves 75
physical action steps: 15 gripper steps and 30 for each EEF call. The actual
planner may stop early on measured convergence. A successful command setter
return is recorded separately from measured arrival. Review grants a reentry
token only after fresh synchronized RGB and joint feedback, healthy device
status, measured arrival, and clearance of the triggering critic rule. The
subsequent `arx.zeva` call must use that token.

## Required frozen inputs

Supply absolute paths and SHA-256 values for:

- hardware config, task manifest, model contract, gateway limits, runner limits;
- CandidateBundle JSON, tool catalog JSON, real input contract JSON;
- one Python feature provider and its SHA-256;
- a reviewed right-arm kinematics calibration when recovery uses `arx.move_eef`.

Generate the full recovery tool catalog with
`python -m scripts.deployment.export_arx_real_catalog --with-eef --output /absolute/path/tool-catalog.json`.
The printed digest goes into the real input contract. Omit `--with-eef` only
when the bundle has no EEF step. The gateway compares the entire catalog with
its actual registered tools before opening hardware.
The older simulation campaign's catalog has a different execution output
schema, so its digest cannot be reused for this gateway; the CandidateBundle
JSON itself remains unchanged.

The feature provider is a single Python module exporting `create_provider()`.
The returned object implements `feature_sources()` (a list of
`RealFeatureSource` dictionaries) and `observe(observation, images)` (a map of
feature names to scalar values). Each source declares its camera and/or 14D
joint feedback channels, scalar type, unit, maximum age, and the module's
SHA-256. The runner and gateway compare these declarations with the real input
contract. Runtime values must be finite, correctly typed, and computed from
fresh timestamped sources. The sample's real provider is `robots/arx/deployment/picktube_rgbd_provider.py`.
It uses aligned front D405 metric depth, the pinned front-to-left-base extrinsic,
and right-arm controller forward kinematics to estimate the pink-label-centre
to TCP distance. `front_rgb` must enable depth in the frozen hardware config,
and the input contract must list `front_depth_mm`. Missing or inconsistent
depth, target visibility, or TCP feedback stops feature evaluation.

The 14 real input channel names, in order, are `left_joint_1` through
`left_joint_6`, `left_gripper_policy`, `right_joint_1` through `right_joint_6`,
and `right_gripper_policy`. Revolute joints use radians; grippers use the
calibrated policy coordinate defined in the hardware config. The control
frequency must match the VLA model contract. The camera serial mapping in
`/home/dodo/chenfu/data_collect/data_collect_todo.md` is front
`260422272500`, left `260422271945`, right `260422275847`. Camera intrinsics
files and their SHA-256 values must be supplied separately.

Before opening the devices, run the gateway with `--check-config` and the
same `--hardware-config`, `--expected-hardware-sha256`, `--task`,
`--model-contract`, `--runtime-config`, `--bundle`, `--tool-catalog`,
`--real-input-contract`, `--expected-real-input-sha256`,
`--feature-provider`, `--expected-feature-provider-sha256`, and optional
`--kinematics-calibration` arguments. A successful report has
`hardware_opened: false`; starting the runner still verifies live camera
identity, synchronization, state shape, and position bounds on reset before
any command.

Then pass the same inputs to `run_arx_real_bundle`, using `--python` for the
runtime with robot drivers, `--output` for a new attempt directory, and
`--runner-limits` for the separate runner budget. The CLI also accepts
`--zeva-host`, `--zeva-port`, `--listen-host`, and `--listen-port`.

On dodo, the checkout is `/home/dodo/chenfu/Agentic-Embodied` at Git commit
`3d842e2` (or a later commit from the same branch). An isolated Python 3.12
environment is at `/home/dodo/chenfu/.venv_arx_real`; the existing collection
environment provides NumPy, Pydantic, HTTPX, and RealSense bindings. In the
shell that launches the runner, source `/opt/ros/jazzy/setup.bash` and
`/home/dodo/chenfu/ARX_X5/ROS2/X5_ws/install/setup.bash`, then prepend
`/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages` to
`PYTHONPATH`. Set `--python` to
`/home/dodo/chenfu/.venv_arx_real/bin/python`. Import checks for ROS2
`RobotStatus`, RealSense, FastAPI, Uvicorn, NumPy, Pydantic, and HTTPX passed;
this is an environment check, not a hardware motion test.

## Joint bounds checked against the SDK

On dodo, `SingleArm(type=2)` loads `x5_2025.urdf`. Its six joint limits are
all `[-10, 10]` rad. The Python SDK exposes joint setters and feedback, but
no public joint-limit getter. Its internal motor code has a bound-restriction
symbol; this does not establish six safe controller-coordinate limits. The
checked official AC one CAD URDF is pinned in
`robots/arx/manifests/real/ac_one_urdf_limits.json`: its six revolute ranges
are narrower, but the transformation from CAD coordinates to the deployed
controller coordinates has not been verified. Accordingly, deployment requires
explicit controller-coordinate `joint_min_rad` and `joint_max_rad` for each arm.
Commands and initial feedback outside these bounds are rejected. Type-2
bounds cannot exceed the SDK URDF's `[-10, 10]` envelope. The CAD numbers are
not silently substituted for measured controller limits.
On a read-only dodo status sample, left joint 2 was about `-0.00286 rad` while
the CAD lower bound for its corresponding joint is `0 rad`; the right gripper
native status was about `-1.54`, while the CAD finger joints are specified in
metres. Direct substitution of CAD bounds would reject observed idle feedback;
a measured coordinate and gripper calibration is required.
The active dodo controllers use `remote_slave`, CAN `can1`/`can3`, end type 2,
and the default status/command topics used by this backend.

## PickTube RGB-D evidence

`scripts/deployment/evaluate_picktube_rgbd.py /home/dodo/chenfu/data/raw/PickTube`
checks the saved front RGB frames without moving hardware. On 2026-10-01 it
located the label in the initial frame of all 50 valid episodes; sampling one
frame in ten across the trajectories found it in 1251/1595 frames. The saved
front depth JPEGs are 480×640×3 color previews, so they cannot validate a
metre distance. Synthetic aligned uint16 depth plus a known right TCP gives
the expected 0.20 m in the full provider wrapper. A read-only live front
D405 sample had no visible pink tube; real object-distance accuracy is still
unmeasured. Three cameras plus front depth worked at 640×480@15, while
simultaneous 30 fps startup failed with a USB I/O error.

This path has been tested with fake hardware, the frozen sample bundle, and
read-only dodo cameras. It has not commanded dodo's motors. The live 14D
status/start-state check and small-motion arrival check are the remaining
hardware gates before a full CandidateBundle episode.
