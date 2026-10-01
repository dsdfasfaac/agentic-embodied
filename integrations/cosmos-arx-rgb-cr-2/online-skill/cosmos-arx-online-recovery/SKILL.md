---
name: cosmos-arx-online-recovery
description: Independently operate an already-running MuJoCo ARX test-tube recovery episode from current RGB and a restricted tool interface, repositioning an open gripper to pregrasp before fresh Cosmos continuation. Not a real-robot controller or a guaranteed successful grasp policy.
---

# Online RGB recovery

You are the execution Agent, not the developer/evaluator. Your task is to finish
the current tube pickup if the permitted evidence supports safe progress. Use
only this skill, the provided tool client, current observations/proposals, and
the observations and actions you accumulate during this episode. Do not search
the repository, other tasks, old seeds, reference trajectories, saved recovery
actions, evaluator outputs, model development notes, or the parent conversation.
Do not ask the developer which motion to choose. General learned robotics
knowledge is allowed; hidden episode information is not.

Read [references/tools.md](references/tools.md) completely before acting. The
task launcher provides an absolute client path. Run its `state` command for the
initial observation. Inspect every returned RGB view before selecting a bounded
action. Physics pauses while you deliberate. Keep your own episode memory; a
fresh start means no development history, not forgetting your own observations.

## Observe, classify, and choose

The target is the tube with the magenta/pink band, supported by a yellow rack.
Identify the same tube across views; color thresholds may merge distractors.
`front_rgb` is the main fixed view and `right_rgb` is the wrist view; `left_rgb`
is an optional corroborating image. The critic uses only main + wrist RGB.

An `attempted_grasp_target_left_behind` proposal is a suspicion: a visual grasp
attempt followed by wrist departure while the target remains at the rack. Check
whether the fingers are actually empty and the target still supported.
`grasp_outcome_unknown` means uncertain, not failure or success. Motion in a
single image is not contact evidence. If the target is already held, do not
blindly open; review current evidence and choose nominal continuation or stop.
Stop if the rack topples, target is unsupported/unidentifiable, or a proposed
motion has no visible clearance. Do not invent depth when the estimator fails.

For an empty attempted grasp with an upright supported target, aim for an
**open, aligned pregrasp**, then let Cosmos close and extract. Do not manually
close-and-lift to substitute for the nominal policy. Do not immediately call
Cosmos again from the unchanged failed state.

## Restage with closed-loop Cartesian tools

1. If current RGB supports opening without releasing a held object, open the
   gripper. Inspect the result. If fingers are wedged around rack geometry,
   choose a small clearing motion based on visible space; do not replay an old
   path or assume opening guarantees collision clearance.
2. Request simultaneous RGB `geometry`. It returns band-to-commanded-TCP
   displacement in tool/world coordinates, or explicit uncertainty. The band
   center is a visual alignment landmark, not exact object pose or grasp center.
   Cross-check its sign and magnitude against the images and residual.
3. With credible geometry and a clear corridor, aim for tool displacement
   approximately `[0.025, 0, 0]` metres: the band ahead of the open pad midpoint,
   laterally/vertically centered. For a small translation with fixed orientation,
   a candidate correction is measured displacement minus that desired offset.
   Choose only a bounded portion consistent with visible clearance (typically
   2–5 mm near the rack, never norm >10 mm), inspect RGB, and re-estimate. This
   equation is an option, not an automatic move: discard it if geometry/identity
   is uncertain. Do not treat commanded position as measured arrival.
4. Tool X is the finger approach direction; tool Y is the closing axis. Keep
   both near horizontal for this rack. If images show tilt, use small justified
   rotations about world/tool axes, at most 0.1 rad per decision, then recheck.
   Do not rotate blindly in contact. An IK rejection executes no move; shrink or
   revise a motion only if RGB still supports it, otherwise stop.
5. If depth is unavailable from simultaneous views, you may explicitly choose
   active observation: keep jaws open and the supported target stationary,
   move the wrist through visibly free space in small steps, and inspect each
   view. A 2–3 mm reversible probe can establish how the target moves in the
   wrist image before a larger observation baseline. Do not infer metric depth
   from pixel shifts alone. Collect at least five distinct snapshots with the
   target fully visible, within the last 150 simulation frames, including the
   present frame. For a handoff-valid temporal fit, the commanded camera
   baseline must span >=25 mm, and both interleaved subsets must independently
   have enough parallax. Five nearly identical images cannot provide depth.
   Use `geometry` with those frame IDs, then explicitly choose temporal mode in
   `review_pregrasp`. If a safe baseline cannot be obtained, stop as unverified;
   do not waive the gate or use a historical target location.

## Verify pregrasp and hand back

After actual EEF restaging, hold >=6 consecutive frames (normally 8). Review the
fresh RGB. Request `review_pregrasp` with simultaneous or explicitly selected
temporal measurement. The gate requires an open jaw, supported/stable target,
near-horizontal axes, tool X offset 15–35 mm, |Y|<=6 mm, |Z|<=8 mm, and consistent
fresh geometry. Correct a rejected candidate only from the reported evidence
and current images. The gate is a provisional candidate test, not a proof of
collision safety or downstream grasp success.

If eligible, cite the returned proposal ID and provide five grounded checks:
target identity, open finger corridor, clear approach, supported target, and
appropriate policy phase. Send `resume_vla`, initially for 1–2 chunks. After
handoff continue requesting fresh Cosmos chunks and reviewing RGB; do not end
just because one handoff succeeded. You may increase the chunk review interval
when the visible scene permits it. Stop a renewed confirmed failure after this
one recovery cycle. Unknown evidence may justify inspection or bounded nominal
continuation, not a success claim.

## Stop and report

The experiment permits one recovery cycle, 1400 total simulation steps, and
at most 100 online action decisions. These limits include hold/measurement
decisions and are not a target to exhaust. Finish sooner if safe progress is
not supported. No autonomous reset, replay, alternative seeds, or code changes.

A visual success candidate requires the entire tube clearly separated from the
rack and stably carried between the fingers, not merely a few millimetres of
band motion. If this is visible, hold 15 frames, inspect again, then `finish`
with a qualified visual verdict. Do not read contact/pose/success flags. An
independent evaluator will check physical completion after the worker closes.
Always explicitly `finish` and await `closed`; this preserves videos/logs.
Report what you observed, which tools you selected, whether handoff happened,
and why you stopped. Distinguish uncertainty from failure and success.
