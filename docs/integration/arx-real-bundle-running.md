# ARX CandidateBundle real-robot execution on dodo

## Current verified state (2026-10-04)

The direct deployment checkout is `/home/dodo/chenfu/Agentic-Embodied` on
`dodo`. The model, gateway, runner, ROS2 controller and three D405 cameras all
run on dodo; there is no cross-host inference transport. The H100 Task7 Model A
package was copied, SHA-checked and served locally from
`/mnt/hdd16t/chenfu/cosmos_models/arx_model_a_5task_iter5000_20260817` on
loopback port 5583. The alternative chemistry RealData checkpoint is not
compatible with this Task7 model contract.

The current frozen inputs are:

| Input | Tracked path | SHA-256 |
| --- | --- | --- |
| Hardware | `robots/arx/manifests/real/dodo_picktube_hardware.json` | `4578b5abf38262b59e8a85d6cae1026ba4517e50b284d0275499983f8b1e5ccf` |
| CandidateBundle file | `robots/arx/manifests/real/sample_picktube_candidate_bundle.json` | `d3549226cd19d571684978171803ee689535aad3e80663e7dfe89551909d309e` |
| CandidateBundle semantic identity | same file | `4ca69f3260760bf8d0df54bcd907023df2c86a2c3c4d5d80f7b32958a63a3e7a` |
| Sample real input contract | `robots/arx/manifests/real/dodo_picktube_real_input_contract.json` | `0859d266bdea538842de7398c805d6a1ae24f8a117b881f1c71c8706aded5557` |
| Provisional retention bundle | `robots/arx/manifests/real/proposed_retention_candidate_bundle.json` | `65cb030aa0efae17ad5deaddb268ee3791fd7dafd83611991284f8c08178fbb2` |
| Retention real input contract | `robots/arx/manifests/real/dodo_retention_real_input_contract.json` | `c8710253e3a1e26334ab4e71269bff6050ed9797b4675ae66f6e8f9b12519bb0` |
| Feature provider | `robots/arx/deployment/picktube_rgbd_provider.py` | `274dbb5f628f6b5e7fd85104dae68b2ad2749201c9d5a622c6c41c06fb327f2b` |

The hardware file pins the README camera mapping: front `260422272500`, left
`260422271945`, right `260422275847`, each with its measured intrinsics. It
sets 640×480 capture at 15 Hz and 320×240 RGB/depth input. The controller is
ROS2 `remote_slave` on CAN `can1`/`can3`; 14D feedback and actions use six
radian joints plus a native policy gripper coordinate per arm. The right
outgoing gripper action alone receives `+0.9` to tighten grip. Feedback is
stored unmodified. The controller-coordinate joint bounds are task envelopes
from 50 recorded PickTube episodes plus a small margin, not mechanical hard
stops. The SDK's type-2 URDF uses broad `[-10, 10]` rad bounds; the AC one CAD
URDF is in a different, unverified coordinate system. The command envelope
and per-step tracking gates are enforced before each send.

The provider computes `privileged.interaction.gripper_closed` from fresh
right-gripper feedback, and target distance from front D405 aligned metric
depth, its pinned camera-to-left-base extrinsic, and fresh right-arm joint
feedback. The controller-EE FK was fitted from 40 raw episodes and
held out on 10; position error P95 was 1.10 mm. The provider cross-checks
that FK against the controller's fresh `end_pos`, then applies the nominal
gripper tool offset. The observed pink-label centre is a grasp-target proxy,
not a direct measurement of finger contact. When depth drops briefly while
the gripper is open, the last depth point may be reused for at most 1.5 s only
while the label remains within 5 pixels of its depth-validated location in a
fresh RGB frame. Missing target, longer dropout, target motion, or a closed
gripper fails the feature check.

The sample bundle's critic proposes an interrupt for a closed gripper far
from the tube. The runner enforces the bundle's tool order and budgets: open
the gripper, execute two 1 cm EEF increments, review fresh real observations,
then invoke VLA only with a granted reentry token. Five decisions and 138
physical recovery steps are reserved. Every physical step records command
send and measured arrival separately, with camera, state, timestamp and
health data. Reentry requires fresh synchronized sensors, healthy transport,
measured arrival and clearance of the triggering rule.

Trial `runs/arx_real_picktube_20261004_trial05` on dodo recorded 443 physical
steps in the runner result and one additional step committed before a safe
failure. It exercised four critic interrupts; three complete recovery chains
reached `reentry_accepted` and VLA continued. During the fourth chain, a D405
metric-depth gap exceeded the previous 500 ms fallback, so feature evaluation
failed and the runner stopped with `recovery_step_failed`. The RGB label was
still visible and the gripper was open. The 1.5 s, 5-pixel fallback above is
the subsequent fix; it has not yet been proven by another live trial. The
trial did not establish successful tube pickup. The controller was stopped
after the failure. The journal is the authoritative record of the partial
444th step; that trial's `result.json` reports only 443 completed physical
steps. Subsequent runner code also counts a gateway-reported known partial
step in the final result.

## Provisional retention bundle

The supplied provisional CandidateBundle is tracked byte-for-byte. Its six
features have real sources in the frozen retention input contract:

| Feature | Real observation and decision |
| --- | --- |
| `gripper_closed` | Fresh right gripper position; the recorded closed-grasp envelope is at most 30% of the calibrated open span. |
| `gripper_contact` | A right-gripper `RobotStatus.joint_cur[6]` closing-load event above 0.16 native units, a target within 5 cm of the FK tool point, and continuing target/tool spatial agreement within 12 mm. The current threshold exceeds the 0.0904 99th percentile in the first 100 open-gripper frames of 50 accepted recordings. This is a contact proxy, not a tactile measurement. |
| `lift_m` | Current pink-label 3D height from aligned D405 depth and the pinned extrinsic, minus its height at episode reset. |
| `grasped` | Contact proxy remains true while the observed tube rises at least 5 mm and the tool has moved at least 5 mm since the load event. |
| `success` | Observed grasp and at least 1 cm tube lift persist for five distinct observations, matching the task's lift/hold thresholds. This is a real-vision proxy; it does not reproduce MuJoCo's bilateral finger-contact evaluator. |
| `target_gripper_distance_m` | Pink-label 3D point to nominal right gripper tool point from depth, camera extrinsic, and fresh FK. |

The ROS2 backend records the gripper current with the same monotonic timestamp
as the 14D state. A missing or stale current, target, depth, FK, or camera
sample rejects feature evaluation. The source contract lists all six feature
names and their real channels; the separate hardware contract pins the cameras
and 15 Hz controller. Contact and success thresholds are provisional and need
measured loaded/empty validation on the actual arm before interpreting them
as physical ground truth.

The original two-step recovery is compiled to three allowed tool calls:
`arx.set_gripper`, an automatically inserted read-only
`arx.review_reentry`, then `arx.zeva` with that review's live token. The
gateway enforces the order, tool arguments and 76-step recovery budget (60
gripper plus one 16-step VLA chunk). The prose fallback is retained as
candidate metadata; any execution failure stops the episode. The candidate's
65-step cooldown is frozen in its real input contract. This new bundle has
passed offline compilation and tests; it has not been run on the robot.

## Deployment commands

The dodo environment is `/home/dodo/chenfu/.venv_arx_real`; prepend
`/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages`
to `PYTHONPATH`, and source `/opt/ros/jazzy/setup.bash` and
`/home/dodo/chenfu/ARX_X5/ROS2/X5_ws/install/setup.bash`. Check the
controller and model service with `scripts/deployment/manage_arx_dodo_controller.sh
status` and `scripts/deployment/start_arx_model_a_dodo.sh status`. The model
service does not open robot hardware.

Run `scripts/deployment/audit_arx_live_observation.py` for a read-only
camera/depth/14D/start-state check. The staging script
`scripts/deployment/stage_arx_picktube_start.py` sends bounded steps to the
empty arms and grippers and verifies measured tracking. It is a motion
operation. Stop the controller on any unexpected motion or failed arrival.

The gateway `--check-config` validates the bundle schema, semantic and file
SHA, task, model, camera serials/calibrations, 14D channel names, feature
provider and its source declarations, catalog, recovery tool sequence, and
budgets without opening hardware. On reset, the runner also checks live
camera identity, synchronization, fresh state, joint bounds and effective
start state before sending a command. `scripts/deployment/run_arx_real_bundle.py`
accepts these tracked paths plus their expected SHA values, `--python
/home/dodo/chenfu/.venv_arx_real/bin/python`, and `--zeva-host 127.0.0.1
--zeva-port 5583`. Each attempt must use a new output directory. It writes
`result.json`, gateway journal and per-call recovery observations under that
directory. A runner error can follow a partially committed tool step; inspect
the journal before any restart.

`freeze_arx_picktube_inputs.py` regenerates the tool catalog and real input
contract after a bundle or feature-provider change. The hardware provenance
script `freeze_arx_dodo_hardware.py` reads the 50 raw PickTube episodes.
