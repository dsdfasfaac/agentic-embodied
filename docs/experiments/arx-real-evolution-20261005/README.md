# ARX real evolution pilot — 2026-10-05

## Scope and current state

This is a Zetta-style physical pilot: collect → cluster → diagnose → propose
CandidateBundle → TemporalCritic shadow replay → physical parent/candidate
comparison → reject or continue validation. Codex is acting as the learner in
this chat. The legacy ARX campaign scaffold is not claimed to run unattended.
One physical comparison cannot establish held-out generalization or promotion.

Generation `g0001` completed the physical pilot. The final matched pair is
`parent-retry03` / `candidate-retry03`. All 14 comparison checks passed, including
the same backend implementation, hardware, model, budgets, calibration, measured
start state, verified arrivals and actual candidate intervention. **The gate
rejects this candidate for promotion: neither arm achieved verified pickup.**
The candidate did complete four real gripper reopen/review/VLA reentry sequences.
The robot has returned to the frozen PickTube start and is disabled. Model A
remains available locally on dodo. Earlier infrastructure interruptions are
preserved separately and are excluded from the final comparison.

## Completed physical comparison

| Result | Parent | Candidate |
| --- | --- | --- |
| Physical rollout steps | 600 | 391 |
| Verified arrivals | 600/600 | 391/391 |
| Recovery / accepted reentry | 0 / 0 | 4 / 4 |
| Minimum measured target distance | 0.238488 m | 0.107362 m |
| Maximum observed tube lift | 0.002765 m | 0.003161 m |
| Contact / grasp / success observations | 0 / 0 / 0 | 0 / 0 / 0 |
| Termination | 600-step limit | Critic proposal after recovery budget exhausted |
| Post-episode home / disable | Verified / completed | Verified / completed |

Candidate interruption steps were 41, 133, 219, 305 and 391. The first four
ran the compiled program: gripper opening in 51 bounded physical steps,
read-only measured reentry review, and fresh VLA continuation. Each review
passed at observations92,184,270,356. The fifth proposal caused
`RECOVERY_BUDGET_EXHAUSTED`; no assistant motion stop caused this termination.
The standard Zetta critic resets cooldown state when activation gates become
false (the gripper opens), so the configured cooldown is not a guaranteed
96-step separation across recovery. This pilot preserves that behavior.

The candidate's closer approach is a diagnostic observation, not demonstrated
task rescue or a generalized improvement. The reopened gripper was again
closed before target contact after each continuation. Reopening alone does
not repair the remaining approach/grasp behavior. The next learner diagnosis
should examine the observed target/tool geometry, approach direction, and
closure timing before authoring a new recovery mechanism. Keep the current
parent; this candidate is not promoted.

Artifacts:

- `g0001/pilot-gate-retry03.json`: final gate, identities, budgets and outcomes.
- `g0001/paired-execution-audit-retry03.json`: interruption steps, actual opening
  results, all reentry checks, post-episode observations, full SQLite-export SHA
  and ordered-record SHA (including records still present in WAL).
- `g0001/parent-retry03-home.json`, `g0001/candidate-retry03-home.json`: verified
  homing before controller disable. Home movements are separate from rollout.
- `g0001/frames/`: original front start/end frames. Visual review confirmed the
  same pink rack tube in both starts and the tube remaining in the rack at end.
- Raw journals, RGB, receipts and per-step feedback remain on dodo under
  `runs/arx_real_evolution_20261005/g0001/{parent,candidate}-retry03`.

## Deployment corrections and excluded attempts

The original `parent` stopped at step516 because an operator hand obscured the
yellow rack. Its error was an observation/critic execution fault, not a critic
rule proposal. `parent-aborted-evidence.json` preserves it. After operator scene
clearance, `parent-retry01` completed600 steps. The initial `candidate` completed
three recoveries before a gripper arrival failure at step297: target and feedback
were both within the calibrated closed band, but the backend accepted stable
closure only for commands beyond the zero endpoint. `pilot-gate.json` therefore
correctly reports that older comparison as inconclusive.

The backend now accepts stable measured closure for any target in the calibrated
closed band while retaining position errors, fresh feedback, arm arrival gates,
and position requirements for opening. `parent-retry02` exposed a different issue:
the VLA predicted a joint target beyond the frozen deployment envelope, rejected
before SDK publication. The backend now limits joint commands to that existing
envelope before dispatch and records `joint_command_limited`; the filter keeps
the bounded command as its state to avoid saturation windup. No joint envelope
or gripper preload was widened. The final pair uses the same corrected code.

The operator requested homing before disable. The wrapper now invokes
`finish_arx_real_episode.py` after a terminal runner result: check current
observations, home empty arms/grippers in measured bounded steps, independently
verify the start state, then disable. A possible held tube waits for operator
unloading; a failed home/observation leaves the controller enabled. Recoverable
critic interruptions retain motor control. The final parent used113 home steps
and candidate81, all with verified feedback. The23 focused backend, comparison
and post-episode tests passed on dodo using the existing runtime dependencies.

## Evidence and frozen candidate

- `g0001/development-evidence.json`: trial02's 600-step journal summary and SHA.
- `g0001/diagnosis.json`: provisional premature-empty-closure hypothesis.
- `g0001/candidate.json`: immutable generation-one bundle, linked to the parent.
- `g0001/protocol.json`: preregistered pilot, fixed model and 600-step/4-recovery budget.
- `g0001/real-input-contract.json`: pinned real feature/provider/tool sources.
- `g0001/shadow.json`: replay with the actual Zetta `TemporalCritic`.
- `g0001/original-control-audit.json`: comparison with dodo's original control code.
- `g0001/execution.sh`: reproducible deployment commands, including explicit motion modes.

Development trial02 first closed its empty right gripper at step26, 0.4132 m
from the target, then recorded 575 closed observations without contact or pickup.
The candidate retains the parent retention rule and adds `far_empty_closed`:
distance >0.10 m, closed gripper, no contact/grasp/success and lift <0.005 m for
16 assessed steps. Recovery opens the gripper, reviews fresh measured state,
then invokes one fresh VLA chunk. This tests the closure hypothesis without
adding an unsupported spatial displacement. Shadow replay produced five
proposals; the first is step41. Recovery reserves 76 steps and three tool calls.

The original-code audit found zero difference in model-state preprocessing
across 601 recorded states and zero difference in filter arithmetic across
600 raw targets with a common initial command. It does not prove RGB or full
deployment equivalence. Original source uses a 32-action prefix; this gateway
uses 16. Both proposed pilot arms keep the same 16-action prefix. The source
gripper start values differ by 0.02; both use the +0.9 outgoing offset.

## Camera startup and infrastructure history

The paragraphs below record the earlier failures and authorizations; the
completed experiment and current controller state are summarized above.

Cold startup images were dark/blue and did not expose a reliable pink label.
Waiting approximately four seconds restored normal color, consistent with
startup photometric settling; the earlier attribution to site lighting was
not established. The backend now drains four seconds of startup frames before
publishing observations. A test verifies that discarded frames never reach
the observation buffer. The 45 existing focused tests, three pilot-gate tests,
and one warmup test passed (49 tests in total).

All three cameras initially reported manual exposure=25000 and gain=16,
automatic white balance enabled. A diagnostic auto-exposure experiment was
followed by an attempt to restore the original manual settings. That attempt
blocked in the kernel. The current effective camera parameters must therefore
be rechecked after recovery. Fifty auto-exposure frames included inconsistent
label acquisition; they are not a valid startup gate. A warm image alone also
does not validate target identity or metric depth.

The operator confirmed onsite readiness and reconnected the cameras. Afterwards
process1391632 remained in `uvc_ctrl_del_event`, process1392501 in
`uvc_ctrl_cleanup_fh`, and process1393192 in `vb2_video_unregister_device`.
TERM, SDK reset, USB reset and physical reconnection did not clear these waits.
`modprobe -r uvcvideo` returned `Module uvcvideo is in use`. The kernel stack
shows the blocked V4L2 event-unsubscribe ioctl. No controller processes remain.
Host reboot requires separate authorization because it interrupts other dodo
jobs; it has not been performed.
The user subsequently requested no reboot and that robot control remain
stopped. Physical comparison is deferred until the camera driver is recovered.
After a second operator-confirmed reconnection, a bounded read-only camera
retry produced no observation. Its process1395764 also remained in
`uvc_ctrl_cleanup_fh` after timeout termination. The original kernel waits and
module-in-use condition persisted. Robot controller absence was verified
again; no new motion or comparison rollout was started. The no-reboot request
remains in effect.

The subsequent investigation identified the exact UVC status-worker
self-deadlock fixed by Linux commit `6d27f92`. Both original installed
`7.0.0-31` and `7.0.0-34` modules lacked the fix. Patched modules were built
from checksum-pinned Ubuntu source and installed as separate overlays for
both kernels; both initramfs images were updated. Disk deployment preceded the
subsequently authorized reboot. See `docs/integration/arx-dodo-uvc-driver-fix-20261005.md`
for evidence, reproduction, next-boot verification and rollback. The camera
USB reconnection also affected ARX USB2CAN enumeration; restore and verify CAN
mapping before a later authorized controller start.

The user subsequently authorized reboot. At 17:51 CST `systemctl reboot`
accepted the request; the operator's photograph showed an Ubuntu boot splash.
SSH later timed out, and the original address became unreachable. Esc did not
show logs. SSH subsequently recovered and confirmed a changed boot ID,
`7.0.0-34-generic`, and patched UVC srcversion `481BFC00E4FD5950221E0B9`.
The robot-stop request remains in effect; no controller-start instruction was issued.

All three cameras reverted to automatic exposure on reboot. Restoring the
recorded manual 25000 μs/gain16 baseline required active video streams;
outside-stream attempts returned SDK busy responses but no kernel deadlock.
With verified manual settings, `g0001/postboot-camera-audit.json` passed all five
observations: maximum age44.40ms, maximum skew38.17ms, actual rack-tube label
depth411–412mm and depth MAD1mm. Visual review confirmed the selected label's
identity. Parameters are in `g0001/postboot-camera-options.json`, and the exact
restoration sequence is preserved in `g0001/postboot-exposure-restore.py`.
This accepts camera/label acquisition only. It does not establish a robot
distance, grasp, lift or task success. No new parent/candidate rollout ran.

## Resume and evaluate

After camera-driver recovery, inspect effective exposure/white-balance settings
and verify five warm, fresh observations of the actual pink tube with metric
depth. Fix target identification if it is still unavailable; never interpret
a background pink sticker as the tube. Then run the parent and candidate from
matched physical starts with the same hardware, model and budgets.

From `/home/dodo/chenfu/Agentic-Embodied`, the tracked wrapper supports:

```bash
ARX_EVOLUTION_ARM=parent bash docs/experiments/arx-real-evolution-20261005/g0001/execution.sh check
ARX_EVOLUTION_ARM=candidate bash docs/experiments/arx-real-evolution-20261005/g0001/execution.sh check
```

`audit <name>` is read-only. `stage <steps> <name>` and `run` publish robot
commands; use them only in the authorized supervised experiment. Start the
controller separately with its management script. Each arm's output directory
is unique under `runs/arx_real_evolution_20261005/g0001`.

After both arms finish, source the deployment environment and run:

```bash
python scripts/evolution/audit_arx_real_evolution.py compare \
  --parent-trial runs/arx_real_evolution_20261005/g0001/parent-retry03 \
  --candidate-trial runs/arx_real_evolution_20261005/g0001/candidate-retry03 \
  --output runs/arx_real_evolution_20261005/g0001/new-independent-gate.json
```

The gate checks configuration identities, equal initial budgets, approximate
start-state/distance matching, actual candidate intervention, and execution
integrity. Visual scene matching still requires review. Primary outcome is
verified grasp, 1cm lift and five-frame hold. A valid pair without candidate
pickup rejects the candidate; distance gain alone cannot promote it. A rescue
requires additional regression and held-out physical validation. Infrastructure
errors make the comparison inconclusive.
