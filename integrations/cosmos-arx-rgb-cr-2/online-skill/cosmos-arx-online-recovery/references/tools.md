# Restricted online tool contract

The launcher provides a local `client.py` path. This is a transport, not a
recovery policy. It connects to a single prepared simulation and cannot reset
it or open other runs. Do not inspect the backend filesystem or server code.

```
python3 /absolute/provided/client.py state
python3 /absolute/provided/client.py send '{"decision_id":"unique-id","evidence_frame":CURRENT_FRAME,"tool":"hold","args":{"frames":8},"reason":"your grounded RGB reasoning"}'
```

`send` returns immediately with `pending=true`. Poll `state` until pending is
false; allow 5–15 seconds between polls while a request is executing. Never
retry/send another decision while pending, including after a client timeout.
Each state has an observation (frame, local RGB PNG paths, critic proposals,
active tool, episode-ended flag), most recent tool result, and public budget.
Use image-viewing tools on the returned paths. No contact, object coordinates,
reward, evaluator verdict, past seed/recovery records, or measured joints are
available. Online geometry is RGB + static calibration + command prediction.
Viewpoints previously collected by YOU in this episode may be reread.

Every decision requires unique `decision_id`, current `evidence_frame`, one
`tool`, an `args` object, and nonempty `reason`. Choose parameters yourself:

| tool | args | Meaning |
|---|---|---|
| `hold` | `{"frames":8}`; range 1..15 | Keep current command, refresh RGB. |
| `set_gripper` | `{"opening":1}`; 0..1 | 0 closed, 1 open; preserve other targets. |
| `move_eef` | `{"delta_xyz_m":[dx,dy,dz],"delta_rotvec_rad":[rx,ry,rz],"frame":"tool","speed_m_s":0.012}` | One relative move, norm <=10 mm /0.1 rad; frame tool or world; speed (0,0.03]. No collision planner. |
| `geometry` | `{}` or `{"observation_frames":[...]}` | Read-only simultaneous or temporal estimate at current frame; does not move or approve handoff. Only snapshots from this online session may be referenced. |
| `review_pregrasp` | `{}` or `{"measurement_mode":"temporal_rgb","observation_frames":[...]}` | Evidence gate; no motion. Temporal >=5 ordered distinct frames, latest=current, age<=150. |
| `resume_vla` | `{"max_chunks":2}` for nominal continuation | Fresh Cosmos request; 1..32 chunks, normally 16 executed steps/chunk, may interrupt on a critic proposal. |
| `finish` | `{}` | End trial and close video/logs; no reset. |

First handoff after recovery additionally needs:

```json
{
  "max_chunks": 2,
  "pregrasp_proposal_id": "COPY_CURRENT_ELIGIBLE_ID",
  "reentry_reason": "Explain why current RGB is an open supported pregrasp",
  "visual_checks": {
    "target_identity": "Current evidence for the intended tube, not a distractor",
    "open_finger_corridor": "Current evidence for open jaws and target alignment",
    "approach_clear": "Current visible clearance from rack edges and other objects",
    "supported_target": "Current evidence target is upright and still supported",
    "appropriate_policy_phase": "Current evidence this is pregrasp, closure/extraction remains"
  }
}
```

Each check must be >=12 characters of actual reasoning. A move or hold invalidates
the old gate proposal; review again before handoff. A rejected request is not
authorization to weaken thresholds. Observe the current frame returned after
rejection; it may be unchanged. Geometry failures return `status=unknown`; they
are not target-pose observations.

Temporal geometry assumes the target stayed still through the selected views.
The handoff gate further requires residual<=2 mm, baseline>=25 mm, two subset
fits agreeing within 3 mm, stable rack/support cues, and current settled RGB.
Simultaneous gate residual<=3 mm, with >=2 valid views. Neither metric is a
certified error bound. If only one side sees the target, a current single view
can still support identity/qualitative inspection but cannot certify depth.
