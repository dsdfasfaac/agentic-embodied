# ARX supervised pregrasp commissioning — 2026-10-07

## Outcome

The gateway executed reviewed physical recovery commands on dodo. **Pregrasp
was not reached; no tube grasp or lift occurred.** At the end, fresh feedback
verified the task start, then the controller was disabled. See
`evidence/finish/result.json`, `before.json` and `after.json`.

| Attempt | Physical steps | Result |
|---|---:|---|
| `pregrasp-short-check-02` | 6 (hold + 5 pregrasp) | All arrivals verified; commissioning cap, target distance 0.3928 m |
| `pregrasp-full-check` | 1 hold | IK review rejected before recovery motion |
| `pregrasp-full-check-02` | 1 hold | All orientation candidates rejected before recovery motion |
| `pregrasp-full-check-03` | 121 (hold + 120 pregrasp) | Rack visibility computation failed; last valid target distance 0.1690 m |
| `pregrasp-full-check-04` | 140 (hold + 139 pregrasp) | Pink label itself became occluded; last valid target distance 0.1242 m |

The initial hold was interrupted by the commissioning bundle's critic rule.
Both longer motion interruptions were **observation computation errors**, not
normal task critic verdicts. Gateway stopped subsequent writes; controllers
remained enabled while cleanup returned the empty, open grippers to the start.
The six-step cap was an explicit commissioning boundary, not a success verdict.
No VLA inference, engagement, closure, or lift was executed by this harness.

The full04 preflight reviewed 218 planned steps and minimum observed TCP
clearance 0.02575 m. TCP clearance does not certify all arm/finger geometry.
The final images show front-camera occlusion while the right wrist still sees
the pink tube. They are original 320x240 images, not annotated reconstructions.

## Changes supported by this evidence

- Preserve command coordinates at real-backend reset, avoiding double
  application of the existing +0.9 gripper preload on the first hold.
- Solve IK within the existing measured joint limits using bounded least
  squares and a rotation logarithm. Invalid measured starts remain rejected.
- Optionally search six small world-axis rotations for geometric proposals;
  commissioning uses 0.05 rad. Target translation is preserved, and every
  candidate receives the same independent path review.
- Keep rejected bundle-step outputs in the durable journal.
- Require yellow rack context on initial target acquisition. Subsequent
  tracking still needs a fresh bounded pink component and metric depth, but
  allows the rack to be hidden. All 125 front frames from full03 replay with
  this change; full04 still stops when the pink label disappears.
- Retain synchronized RGB-D/state snapshots for offline proposal/path audits.
- Provide a commissioning harness that rejects learned, closing, and engagement
  prefixes before its first command. Only the frozen geometric pregrasp prefix
  is executed; it stops before reentry/VLA.

`grasp-config.json` expands only this supervised commissioning travel budget to
0.40 m, speed 0.03 m/s and pregrasp step budget 240. Existing deployment configs
retain their previous settings. The bundle is an explicit commissioning
candidate; it is not self-evolved or promoted. GraspGen model-to-ARX finger
transfer remains unverified; these physical commands used `tube_geometry`.

## Reproduce

From `/home/dodo/chenfu/Agentic-Embodied`, with onsite readiness established and
controller running:

```bash
source /opt/ros/jazzy/setup.bash
source /home/dodo/chenfu/ARX_X5/ROS2/X5_ws/install/setup.bash
export PYTHONPATH=.:/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages:$PYTHONPATH
PY=/home/dodo/chenfu/.venv_arx_real/bin/python
DOC=docs/experiments/arx-pregrasp-commission-20261007
"$PY" scripts/deployment/commission_arx_pregrasp.py \
 --bundle "$DOC/bundle.json" --grasp-config "$DOC/grasp-config.json" \
 --frozen "$DOC/frozen" --output runs/pregrasp-new-attempt \
 --hardware-sha256 f47a5847219cb3d83b0c1de12d902adb0461f475d65508efc165f1764ccea836 \
 --max-physical-steps 6 --execute
```

Use a new output directory. A larger cap up to 301 admits the entire reviewed
pregrasp prefix, but the current front-only feature provider will stop if the
arm hides the target. Every proposal/review uses a fresh synchronized sample;
full journals and sensor NPZ originals remain on dodo under
`runs/arx_grasp_recovery_20261007`. The evidence index records their SHAs.
Provider SHA changed after full03, and each result retains the provider identity
that actually produced its features. The final frozen contract names the fixed
provider, not the provider used by earlier runs.

## Next acceptance gate

1. Verify the right wrist camera-to-gripper transform against synchronized
   front RGB-D and joint FK across poses, retaining residuals and held-out
   samples. `/home/dodo/chenfu/docs/ARX_X5_CAMERA_TRANSFORMS_HANDOFF.md`
   supplies the transform copied from the left wrist; it labels the right side
   unvalidated. Right RGB intrinsics were independently pinned in this repo.
   PickTube `000015` contains depth folders of 3-channel JPEG visualizations,
   not uint16 metric depth; those images cannot serve as metric depth samples.
2. Enable aligned right metric depth and same-target observation handoff while
   the front target is occluded. Test colour distractors, target loss, camera
   freshness and disagreement; don't count cached target positions as fresh
   lift/hold evidence.
3. Reach measured pregrasp, then validate learned frame/finger transfer,
   engagement, closure and lift. Success needs the observed pink tube raised
   at least 1 cm for five fresh frames. Resume VLA only after real reentry review.
4. Run matched parent/candidate colour-position trials before promotion.

Verification: 109 tests passed, one RealSense-dependent local test skipped.
Physical RGB-D/FK observations and command/arrival records are retained above.
