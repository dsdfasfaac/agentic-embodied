# ARX CandidateBundle real-robot execution

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

This path has been tested with fake hardware and the frozen sample bundle.
It has not commanded dodo's motors. A real trial requires the operator's
calibrations, feature provider, frozen limits, and an available Zeva endpoint.
