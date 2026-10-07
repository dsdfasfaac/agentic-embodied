# ARX target-conditioned grasp recovery

## Implemented boundary

The critic proposes interruption. The runner executes only the frozen bundle's
ordered calls. Grasp proposals have no robot handle and cannot send motor commands.
The gateway retains synchronized RGB/depth snapshots, proposal/review results and
each physical command/feedback observation in the episode journal.

Tools:

| Tool | Operation | Admission / result |
| --- | --- | --- |
| `arx.propose_grasp` | Fresh pink-label D405 cloud; geometry, Contact-GraspNet or GraspGen proposals | Read only; observation/target ID, calibration, cloud SHA, camera timestamp, poses and model evidence |
| `arx.review_grasp` | Same-target drift check, phase prerequisite, bounded joint IK and observed TCP sweep clearance | Read only; single-use token, planned step count, goal and certificate limitations |
| `arx.execute_grasp` | One `pregrasp`, `engage` or `lift` phase | Fresh sensors, same target, matching unexpired review, joint increments and physical budget; the existing gateway target loop performs all writes |
| `arx.review_reentry` | Existing health/arrival/critic checks plus this recovery's same-target measured pregrasp | Fresh VLA token only when all checks pass |

Execution distinguishes command convergence from measured TCP completion.
The current sensor certificate covers TCP clearance and joint IK. It does **not**
certify full arm/finger collisions, invisible obstacles or complete target geometry.
The existing episode cleanup handles verified homing and disabling, or requests
unloading if an object may be held.

## Reusable interfaces

`robots/manipulation/grasp_proposals.py` defines `TargetCloud`,
`GraspProposalEngine` and the two learned-service protocols without importing ARX,
ROS, MuJoCo or an arm SDK. Another robot supplies its camera-to-base transform,
target observer, TCP calibration and motion adapter. `SensorBackend.observe()`
specifies fresh sensor acquisition without publishing a command.

The ARX observer uses the existing audited D405 intrinsic/extrinsic and controller
FK artifacts. Optical points are metres. The left-base camera calibration is
converted to right-base with `[0, +0.5, 0]`. Controller joints are radians; gripper
feedback/commands preserve the existing policy coordinate and `+0.9` command preload.
Capture remains 640×480@15 Hz, downsampled to 320×240 with aligned millimetre depth;
command interpolation uses the configured control frequency (currently 15 Hz).

The model input contains **only the selected visible pink-label surface points**.
This excludes other colours from proposal inference, but is a partial surface,
not an entire segmented tube. The separate scene cloud is retained for review.

## Engines and gripper frames

`tube_geometry` creates a target-proxy pose with the current ARX TCP orientation.
It is a geometric proposal, not learned grasp confidence. The surface correction
defaults to zero; change it only from a measured tube/label offset.

`contact_graspnet` accepts camera-frame proposals from a configured local
`/health` + `/propose` service. `graspgen` centres the selected object cloud,
calls local `/health` + `/generate`, then restores its centroid and transforms
poses into right-base coordinates. There is no silent engine fallback.

The upstream [GraspGen repository](https://github.com/NVlabs/GraspGen) lists Panda,
Robotiq 2F-140 and suction models. ARX is not one of those supplied grippers.
Its [frame convention](https://github.com/NVlabs/GraspGen/blob/main/docs/GRIPPER_DESCRIPTION.md)
uses +Z approach and X closing. ARX's audited tool forward direction is +X.
For learned-pose execution, freeze `learned_gripper_id`, `learned_model_sha256`
and `learned_grasp_to_tcp = T_grasp_from_arx_tcp`, and verify the physical gripper
transfer. A matrix alone does not establish matching finger geometry or opening.
Unverified learned proposals remain available for inspection and are rejected
at motion review. The service health identity must match the frozen model/gripper.

## Local model deployment

`scripts/deployment/serve_arx_graspgen_local.py` loads the official
`GraspGenSampler` directly in a separate Python environment on dodo. It binds
loopback only, records generator/discriminator/config SHA and never imports an
arm or camera controller. Put source, weights, environment and caches on the
large disk, e.g. `/mnt/hdd16t/chenfu/grasp_recovery`, away from the repository.

After installing the official GraspGen runtime and downloading its checkpoints:

```bash
GRASPGEN_CONFIG=/mnt/hdd16t/chenfu/grasp_recovery/models/checkpoints/graspgen_robotiq_2f_140.yml \
  bash scripts/deployment/start_arx_graspgen_dodo.sh
curl http://127.0.0.1:18093/health
```

The launcher defaults to GPU 1 and keeps inference on dodo. The Robotiq model
can be inspected as a proposal baseline; its geometry has not been validated for
ARX. Source inspected for this implementation: GraspGen commit
`2dd8852e1be60f5f9d277fafcc621835cdf59110`; model repository revision
`ec1ccbb5eec0680db669246ac312a3636f16ee43`. The supplied Panda config uses PTv3;
the Robotiq config uses PointNet. Install the dependencies required by the
chosen config and verify an actual inference before commissioning.

The 2026-10-07 dodo deployment now passes native model loading and camera-only
proposal inference. See [deployment evidence and reproduction](../experiments/arx-grasp-recovery-20261007/README.md).
The launcher uses `venv313` to match the installed Torch Python ABI. The two
checked-in upstream patches and separate environment versions are recorded
there. Physical ARX transfer and synchronized joint/path acceptance are pending.

## Bundle and commissioning

The provisional generation 2 bundle, configuration, frozen catalog and input
contract are in `docs/experiments/arx-grasp-recovery-20261006`.
The main candidate reserves 316 physical steps per recovery: opening 60,
pregrasp at most 240 and a fresh VLA prefix of 16. The full grasp/lift example
reserves 561 steps and 12 decisions; it is a separate provisional variant.

The delivered grasp configuration has `motion_enabled=false` because this new
motion primitive has not had a physical commissioning run. The existing baseline
catalog/bundles remain usable without `--grasp-config`. For commissioning, review
the retained RGB-D proposal/path, create a separate enabled configuration and
freeze its SHA/catalog/input contract; pass the added `--grasp-config` and
`--grasp-config-sha256` arguments to `run_arx_real_bundle.py`. Both paired arms
must use the same hardware, code, model, limits and catalog.

Freeze the full variant separately if testing its sequence:

```bash
python scripts/deployment/freeze_arx_picktube_inputs.py \
 --bundle docs/experiments/arx-grasp-recovery-20261006/candidate-grasp-and-lift.json \
 --grasp-config docs/experiments/arx-grasp-recovery-20261006/grasp-config.json \
 --output-dir runs/arx_grasp_full_frozen
```

`probe_arx_grasp_snapshot.py` takes a retained `grasp-sensors/*.npz`, its
`grasp_sensor_evidence` observation JSON, frozen hardware/config and engine.
It checks offline candidate IK/clearance without opening the SDK and never emits
a live motion token. Original clocks are retained; replay is not reported as live.

Acceptance requires a measured pregrasp, then correct pink-target acquisition
and at least 1 cm lift held for five fresh frames. Swap colour positions between
matched trials. Reachability or a reduced target distance alone cannot promote
a bundle. VLA and grasp model weights are fixed during bundle evolution.
