# GraspGen ARX live commissioning — 2026-10-08

This package tests learned Robotiq proposals transferred to the ARX TCP. It is supervised commissioning, not a promoted candidate or a certified general gripper transfer.

## Measured pregrasp

`pregrasp-live03` reached the reviewed learned pose: 195 execution steps, 211 total steps including holds. The gateway separately recorded command completion and measured physical arrival. Right wrist identity remained observed (150 cross-view validations); open gripper, no contact, target distance 0.05403 m. Full raw journals, synchronized RGB-D and commands are retained on dodo at `/mnt/hdd16t/chenfu/grasp_recovery/adaptation-20261008/`.

`pregrasp-live02` sent no pregrasp movement: fresh replanning selected a different numerical IK branch and exceeded the 0.4 m arc budget. Execution now keeps the final IK branch selected by review, and the bundle settles feedback before proposal.

## Implementation changes

- Actual depth samples connected to the pink seed supply the visible upright tube surface; no synthetic depth completion. The original 12-pixel seed is admitted only with existing rack, color and depth evidence.
- The configured motion envelope retains the recorded zero offsets and expands selected joints within the SHA-pinned AC one URDF ranges. This is CAD evidence, not a claimed SDK hard-limit measurement.
- Learned orientations are filtered for an approximately horizontal closing axis and approach alignment; learned centres and rotations are preserved. Parallel-jaw half-turn variants explicitly inherit the model score without rescoring.
- Cartesian IK remains preferred. Bounded joint interpolation may reach pregrasp, with joint increments, TCP arc length, speed, scene points and ARX gripper CAD hull checks.
- Engage and lift retain the candidate that actually reached pregrasp. They cannot silently select another higher-score candidate.
- Dedicated full commissioning permits engage, close and contact-gated lift. Normal deployment still rejects an unverified commissioning configuration. Only observed target lift of at least 1 cm held five fresh frames terminates successfully; no VLA is called by the commissioning harness.

## Configurations

`grasp-config-commission.json` and `bundle-pregrasp.json` are the measured open-pregrasp test. `grasp-config-full-commission.json` and `bundle-full-commission.json` are the follow-up complete grasp test; use `commission_arx_pregrasp.py --full-grasp`. Both require `hardware.json` and their own frozen input/catalog directory.

The Robotiq model SHA is `6a378f83e3b691db76992d62fceb088b04d31d3827923d668f911e045e683acd`. The nominal metric transform is under physical validation, not marked verified. CAD checks cover the visible scene and gripper hulls, not hidden obstacles or a full arm certificate. Wider generalization needs additional scenes and trials.

## Historical pause after pregrasp-live03

The empty open pregrasp was followed by an attempted measured home. The 2 mrad progress cutoff first rejected a 1.907 mrad encoder movement; the staging check now requires 1.5 mrad **toward** the goal, while retaining tracking and final-home tolerances. Subsequent bounded commands still did not move right joint six away from 0.451858 rad, so this is not resolved by the quantization adjustment. Other joints responded. Motion is paused, controller remains enabled, both grippers remain open, and the target is still observed without contact. Onsite cable/contact inspection is required before another movement. No learned closure or lift has been executed or declared successful.

The source is committed and deployed; the current GraspGen service was restarted from the synchronized source and its health/model SHA confirmed. Full commissioning reserves 546 physical steps in 12 compiled calls under a 600-step runtime cap.

`baseline-geometric-bundle.json`, `baseline-geometric-grasp-config.json` and `frozen-baseline/` preserve a runnable static contract for the previous geometric candidate against the current provider/schema. Its hardware remains the previous wrist-observer configuration, not the expanded learned-motion profile. This refresh is a static check, not another physical acceptance. Historical experiment files were not rewritten.

## SDK bound follow-up and re-enable

Onsite confirmation and explicit disable/re-enable authorization were received. A single supervised stage probe with small lag corrections (at most 5 mrad per correction, never outside the existing feedback tolerance minus 5 mrad) did not restore right wrist progress. The controller was then disabled and restarted, followed by fresh feedback, a successful single-step stage, and measured open-gripper start completion. Root cause of the prior stalled feedback remains unconfirmed; restart restored the start checks, not a proven diagnosis.

Inspection of the **actually loaded** dodo SDK library found position-control clipping ranges: lower `[-2.618, -0.1, -0.1, -1.29, -1.4835298642, -1.7453292520]`, upper `[3.14, 3.6, 3.0, 1.29, 1.4835298642, 1.7453292520]` radians. The pinned inspection artifact is `robots/arx/manifests/real/dodo_sdk_position_limits.json`; its library SHA and constructor/control addresses are recorded. These are installed SDK command bounds, not universally valid physical hard limits.

Use `hardware-sdk-bounded.json` and `frozen-full-sdk/` for subsequent full commissioning. Fourth and fifth joint commands now use the intersection of CAD, installed SDK and selected envelope, with margins. Hardware loading checks the installed SDK binary SHA. Earlier frozen packages remain historical and cannot be substituted for the current provider SHA.

### Recorded PickTube homing (2026-10-08)

The operator identified an obstruction below the end effector. Post-episode
cleanup now calls `replay_arx_picktube_home.py` before opening empty grippers
at home and disabling the controller. The previous straight joint interpolation
from an arbitrary rollout pose to the task start is no longer the cleanup path.

The pinned compact trajectory is
`robots/arx/manifests/real/dodo_picktube_recorded_home.json`: accepted PickTube
`000015`, source frames 165–300, measured `observation.state[:,7:13]` in radians.
Raw source SHA: `72aa96b6816c65af8314824eda6fa53a9c613e165d4edb9137066f634b2fa761`.
The return lifts from the rack, retracts, and descends near home. Replay preserves
left arm and both grippers; recorded closing commands are discarded. Grippers
are opened only after measured arm-home verification and unloading checks.

A nearby measured pose joins the corresponding recorded frame. Other poses
need a precomputed vertical escape at their own x/y, followed by an entry
bridge that stays above the escape plane. The remaining recorded frames retain
their ordering, at at most 0.2 rad/s and 15 Hz. Invalid IK, limits or entry
geometry reject the plan before commands. A pose already near home needs only
a short final alignment. This is an empty-gripper taught corridor, not a
certificate of full-arm collision clearance in a changed scene.

Every command records its ROS receipt separately from fresh feedback and
arrival checks. Tracking pauses the replay clock and faults if stalled; failure
holds fresh measured posture and keeps the controller enabled. Controller
startup/disable is a separate action and may affect posture.


### Latest physical result

`home-commission-result.json` records the deployed `a8c5a88` result. After the
operator-authorized controller restart, feedback was already near home; only
the short home alignment was physically executed. Empty-gripper opening took
110 measured staging steps and completed. Full GraspGen attempt `full-grasp-live02`
ended after 16 hold/settling steps: complete-path preview rejected the candidates
before pregrasp movement, closing or lifting. The retained collision audit
locates intersections on the selected tube's lower surface and nearby tubes;
these points have not been erased or declared false positives.

Post-episode cleanup completed, independently measured task-start eligibility,
and then disabled controller PID 274650. The robot is currently disabled.
The full taught return corridor has passed offline planning tests from the
previous learned pregrasp and the interrupted home posture; it still needs
physical replay from a future grasp episode. GraspGen closure/lift acceptance
and general gripper-transfer verification remain pending.
