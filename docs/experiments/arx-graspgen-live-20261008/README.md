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
