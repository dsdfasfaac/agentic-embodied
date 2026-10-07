# Pink tube rollout success, 2026-10-07

## Observed outcome

One supervised **segmented** rollout reached `task_success` at `obs-336`. Five consecutive new wrist RGB-D frames `obs-332-reacquire-5`, `obs-333`, `obs-334`, `obs-335`, `obs-336` measured the same pink target closed/contact/grasped, lifted **13.863–14.877mm**, over **1.601s**. All336 dispatched commands have verified arrival records. The final lift segment issued16 commands and ended on the task success predicate before its nominal20mm pose target finished settling (`command_target_reached=false`, `physical_arrival_verified=true`). The requested >=10mm/five-frame terminal criterion was satisfied.

[Success audit](evidence/success-audit.json) records each frame's sensor SHA, acquisition timestamp, joint/current feedback, quality gates, source journals and results. Full RGB-D is retained under `/mnt/hdd16t/chenfu/grasp_recovery/` on dodo. This is not an uninterrupted autonomous rollout or a success-rate estimate; no candidate is promoted.

| Segment | New commands | Outcome |
| --- | ---: | --- |
|14 |278 | Observer unknown during closure; operator confirmed pink tube held. |
|15 |10 | Fresh closed/contact measured; original8-frame no-lift critic interrupted intended closure. |
|16 |32 | Closure physically settled; old1e-4 command convergence rejected residual0.000207. Original failure retained. |
|17 |0 | Fresh lift review passed, then next scene frame rejected the tube's own upper glass wall as an obstacle. |
|18 |16 | Revalidated contacted upright target volume; reviewed lift reached success. |

## Changes and provenance

- Wrist D405 range admits70mm, keeping79mm contact depth. Three cross-view confirmations establish identity. Established wrist partial depth needs >=30 points, >=50% support, >=80% spatial span on both axes, MAD<=3mm, and existing metric continuity. Front/unestablished wrist require80%. Colour thresholds are unchanged.
- Generation5 candidate extends no-lift dwell8→48 frames (3.2s at15Hz), with the identical recovery program and parent bundle SHA. The history budget is48 frames. This is an evidence-backed candidate proposal awaiting fresh/paired trials.
- Audited continuation preserves all consumed physical/decision budgets, source SHA and measured engage completion. No review token survives; cold reset retains its frozen task-start guard. A new synchronized observation must match checkpoint pose and target before a write.
- Closing budget60/60 was consumed across segments. Completion was reverified from32 arrivals, final commanded residual<=0.001, fresh closed/contact and matching14D state; no extra closing command was issued. Original failed result remains immutable. Last sent command is preserved against blocked fingers, including the existing +0.9 preload.
- For contact-admitted lift only, the commissioned upright target column is radius20mm and40mm above its label, in addition to the original20mm sphere. Marked rejected pixels lay on the selected tube's upper glass wall. Other scene points retain12mm TCP clearance; pregrasp/engage exclusion is unchanged. This is a task-specific geometric assumption, not a full arm/finger collision certificate or learned GraspGen execution.
- Successful continuation reused320 physical steps and15 decision admissions, then charged a new review and lift. No budget was refunded. See immutable [protocol18](protocol-18.json) and predecessors16/17.

## Reproduction and remaining validation

Use the saved generation5 bundle, `grasp-config-held-tube.json`, frozen input/catalog, wrist-depth hardware configuration, calibrated FK and current provider on dodo. The full runner supports a fresh rollout. Continuation is explicitly gated by the original checkpoint and separately hashed closing/review segments. Current success does not establish a new-start, uninterrupted, paired-validated success. The proposed geometry exclusions are scoped to the observed upright PickTube arrangement.

## Controller cleanup

After the operator confirmed the pink tube removed and both grippers empty, measured staging completed in 107 steps. The first finish audit still required the removed target to be visible and refused to disable; its [failed result](evidence/post-episode-first-result.json) is preserved. Cleanup now uses hardware observation only after explicit unloading; normal task feature checks remain enabled by default. Six focused cleanup tests passed, including missing-target cleanup and hardware-fault refusal.

The retry verified home from fresh synchronized 14D feedback (`task_start_eligible=true`, sensor age 49.55 ms, skew 48.99 ms), then stopped controller PID 119319. A separate controller status query returned `not running`. Available device health confirms responsive ROS status without reported faults; motor fault-bit diagnostics are unavailable. Homing is recorded outside the 336 rollout commands.

- [Measured staging](evidence/post-episode-staging.json)
- [Home recheck](evidence/post-episode-home-recheck.json)
- [Fresh after-home observation](evidence/post-episode-after.json)
- [Final homed-and-disabled result](evidence/post-episode-result.json)
