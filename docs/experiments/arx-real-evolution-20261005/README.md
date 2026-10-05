# ARX real evolution pilot — 2026-10-05

## Scope and current state

This is a Zetta-style physical pilot: collect → cluster → diagnose → propose
CandidateBundle → TemporalCritic shadow replay → physical parent/candidate
comparison → reject or continue validation. Codex is acting as the learner in
this chat. The legacy ARX campaign scaffold is not claimed to run unattended.
One physical comparison cannot establish held-out generalization or promotion.

Generation `g0001` completed the evidence, diagnosis, proposal, static contract
and shadow phases. **Neither new comparison arm has executed.** Camera-driver
failure prevented startup; it is not candidate rejection or a task failure.
The robot controller is stopped. The Model A inference service remains running.

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

## Camera startup and infrastructure failure

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
  --parent-trial runs/arx_real_evolution_20261005/g0001/parent \
  --candidate-trial runs/arx_real_evolution_20261005/g0001/candidate \
  --output runs/arx_real_evolution_20261005/g0001/pilot-gate.json
```

The gate checks configuration identities, equal initial budgets, approximate
start-state/distance matching, actual candidate intervention, and execution
integrity. Visual scene matching still requires review. Primary outcome is
verified grasp, 1cm lift and five-frame hold. A valid pair without candidate
pickup rejects the candidate; distance gain alone cannot promote it. A rescue
requires additional regression and held-out physical validation. Infrastructure
errors make the comparison inconclusive.
