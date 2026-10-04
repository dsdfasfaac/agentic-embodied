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
| Real input contract | `robots/arx/manifests/real/dodo_picktube_real_input_contract.json` | `835d92311753c17b06eb3e84d124afb77af604fe6943a93a3af332ac48c23c01` |
| Feature provider | `robots/arx/deployment/picktube_rgbd_provider.py` | `21640d792ea53631499a6aa5d1ae18fd219c100bae8c9d7b8112f00fa01ffb13` |

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
right-gripper feedback, and
`privileged.selected.target_gripper_distance_m` from front D405 aligned
metric depth, its pinned camera-to-left-base extrinsic, and fresh right-arm
joint feedback. The controller-EE FK was fitted from 40 raw episodes and
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
then invoke VLA only with a granted reentry token. Five decisions and 122
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
