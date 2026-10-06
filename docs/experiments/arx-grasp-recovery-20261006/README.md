# Provisional ARX grasp recovery — 2026-10-06

This is an implementation and offline validation artifact. No new robot motion
or learned-model inference has been validated for this candidate.

- `candidate.json`: generation 2, measured pink-target pregrasp followed by VLA.
- `candidate-grasp-and-lift.json`: optional full pregrasp/engage/close/lift/hold variant.
- `grasp-config.json`: motion disabled; optional dodo loopback GraspGen endpoint.
- `tool-catalog.json`, `real-input-contract.json`, `freeze-report.json`: frozen main-candidate declarations.
- `retained-colour-audit.json`: selected pink label in a retained real RGB frame;
  no paired depth/live joint evidence claimed.

Implementation tests cover target-only clouds, right-base conversion, model
centering/identity, stale observations, held-object opening refusal, candidate
drift, changed obstacles, gripper transfer, single-use tokens, measured pose
completion, bundle ordering/budgets, runner/gateway result binding and reentry.

96 focused tests passed locally with the asynchronous gateway tests enabled.
Earlier remote run passed 64 and skipped one async test before connection loss.
The real feature provider itself remains byte-for-byte unchanged, preserving the
previous generation's input contracts.

Deployment work started on dodo's large disk at
`/mnt/hdd16t/chenfu/grasp_recovery`: GraspGen source was copied and an isolated
environment was created. Package installation was interrupted by SSH connection
loss (`192.168.20.56:2222` timed out). It must be audited/resumed before being
reported ready. No grasp weights were loaded and no new control process was
started by this implementation task. The camera/robot state after connection
loss has not been independently checked.

Remaining physical acceptance: retained aligned RGB-D replay on dodo, actual
local GraspGen inference, learned-gripper/TCP transfer validation if enabling
learned-pose execution, and supervised paired colour-position-swap trials.
See [deployment details](../../integration/arx-real-grasp-recovery.md).
