# Dodo retention CandidateBundle trial, 2026-10-05

The first live trial of the supplied retention bundle ended after 40 physical
steps at the gateway's measured-arrival barrier. No critic proposal, recovery,
grasp, or task-success event occurred. The runner stopped publishing commands;
the assistant then stopped the ROS2 controller through its management script. The model service was stopped
after evidence collection. This trial did not establish successful pickup.

## Frozen inputs and evidence

- Code: `a12f7ab` on dodo, aigc31, and GitHub.
- Bundle: `proposed_retention_candidate_bundle.json`, SHA
  `65cb030aa0efae17ad5deaddb268ee3791fd7dafd83611991284f8c08178fbb2`.
- Hardware SHA: `4578b5abf38262b59e8a85d6cae1026ba4517e50b284d0275499983f8b1e5ccf`.
- Real input SHA: `982a0ed0ca1f5b0ea5331b0d5a699d79efeaecc4d52b902fc520f569a5024d34`.
- Provider SHA: `274dbb5f628f6b5e7fd85104dae68b2ad2749201c9d5a622c6c41c06fb327f2b`.
- Trial directory on dodo:
  `/home/dodo/chenfu/Agentic-Embodied/runs/arx_real_picktube_20261005_retention_trial01`.
- Authoritative physical journal: `private/gateway/journal.sqlite3`.
- Saved result: `result.json`; extracted evidence: `diagnostic-summary.json`.
- Startup camera and state audits:
  `/home/dodo/chenfu/Agentic-Embodied/runs/arx_live_readonly_20261005`.

The model startup initially failed while jsonschema imported the RFC3987 Lark
grammar. Preloading `rfc3987_syntax` before Cosmos resolved the startup path;
the local Model A server became ready on loopback port 5583. Both CAN interfaces
were up. Five camera/depth observations passed. The on-site observer confirmed
supervision, emergency stop, tube placement, and empty grippers. One small
staging step was verified, followed by 75 measured staging steps. The final
start-state audit passed with fresh synchronized RGB-D and 14D feedback.

## Stop cause

The third VLA invocation stopped at `obs-40` with
`PHYSICAL_ARRIVAL_UNVERIFIED`. The runner result is `execution_uncertain`,
`termination_reason: gateway_unknown`, `physical_steps: 40`, `recoveries: 0`,
and `task_success: null`. The last command was dispatched, and synchronized
post-command feedback was observed for the 3 s arrival timeout. All arm-joint
and left-gripper errors met their configured tolerances. Only the right
gripper failed its 0.1 policy-coordinate tolerance:

| Right gripper quantity | Value |
| --- | ---: |
| Processed target before tightening offset | -0.8859540224 |
| Tightening offset | +0.9 |
| Sent native / expected feedback target | +0.0140459538 |
| Final measured feedback | -0.0944156647 |
| Final position error | -0.1084616184 |

There was one `arrival_unverified` record and no `critic_proposal`, `interrupt`,
or `task_success` record. The ROS2 publish receipt does not provide a CAN ACK.
The protected stop therefore came from the gateway's position-arrival check;
stopping the ROS controller afterwards was an explicit assistant action.

## Manipulation observations

Forty feature records cover `obs-0` through `obs-39`; `obs-40` was published
but did not cross the arrival barrier and was not evaluated by the critic.
The target-to-tool distance began at 0.4067 m, reached a minimum of 0.4041 m,
and ended at 0.4247 m in the last assessed observation. Maximum observed label
height change was 0.00273 m. Contact, grasped, and success stayed false. The
closed-gripper predicate was true for 14 observations.

The final front image shows the pink tube still in the yellow rack. The right
wrist image shows an empty gripper. These images agree with the absence of
contact and pickup evidence. The trial ended early and does not establish how
the full nominal policy or retention recovery would perform.

## Next corrective work

Review the gripper arrival contract before another trial. The +0.9 command is
intended to tighten the grip, yet the current backend treats the resulting
coordinate as an exactly reachable position. Separate position tracking from
verified closed/loaded gripper behavior using measured feedback and explicit
evidence. Preserve the arm-joint tracking checks and the distinction between
dispatch, position arrival, and load-limited closure. Do not classify either
empty closure or successful reentry as task success.

The supplied retention rule requires a prior positive lift and nearby target,
so it did not cover this initial empty closure far from the tube. Any additional
acquisition recovery must be supplied as an explicit bundle rule and bounded
program rather than silently added to this retention candidate.
