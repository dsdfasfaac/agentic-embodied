---
name: cosmos-arx-rgb-cr
description: Reproduce and operate the H20 MuJoCo ARX test-tube pickup RGB critic and Agent-owned pregrasp recovery with fresh Cosmos handoff. Use this bundle for recorded recovery regression, new RGB-driven recovery trials, and post-close evaluation; not a real-robot controller or a proven general grasp recovery skill.
---

# Cosmos ARX RGB C&R

This entire directory is the skill and runnable distribution. Keep `cr.py`,
`vendor/`, `fixtures/`, and documentation together. No particular Agent API is
required: the running worker accepts one JSON decision per stdin line and pauses
physics between decisions. The operator is responsible for connecting a
multimodal Agent; there is no built-in LLM or autonomous recovery policy.

Read [README.md](README.md) for H20 setup and the distinction between recorded
recovery reproduction and a new live Agent trial. Before making decisions, read
[docs/agent_protocol.md](docs/agent_protocol.md) in full. Use
[docs/reproduction.md](docs/reproduction.md) for regression checks and troubleshooting.

## Non-negotiable observation and ownership boundaries

- The critic provides RGB evidence/proposals; the external Agent selects each
  gripper, EEF move, hold, measurement mode, nominal handoff or stop.
- Inspect the current three RGB views before each bounded tool action. Use RGB,
  static camera/robot calibration and own command history only. Do not read
  measured joints, object positions, contacts, rewards, private trace or offline
  reference results to choose a live recovery action. `original_prefix.npz` also
  contains evaluator-only state: leave it to the runtime, do not inspect its
  private arrays online. Nominal Cosmos retains its original proprioception.
- Treat commands as commanded motion, not measured arrival. Tools are bounded
  IK, not collision-aware planners. At most 10 mm / 0.1 rad per Agent decision;
  review fresh RGB afterward, do not send a blind sequence for a new seed.
- This playbook leaves acquisition/extraction to Cosmos: open jaws, visually
  restage to supported pregrasp, hold and review. Do not replace recovery with a
  direct Cosmos retry, or turn it into manually extracting the tube.
- Only a fresh eligible gate proposal plus the Agent's five explicit visual
  checks can authorize handoff. A rejected/occluded/ill-conditioned estimate is
  not permission to invent a depth or waive thresholds. Temporal RGB is an
  explicit Agent choice and assumes a stationary target over its window.
- A toppled rack or unsupported target is outside this pregrasp playbook. Stop
  and record the limitation rather than improvising an unvalidated reset.
- For a comparable v1 regression, freeze the code, use max_steps=1400 and one
  recovery cycle; new confirmed failure after handoff ends that trial. The
  one-cycle limit is operator protocol, not enforced by the original runtime.
  Continue nominal after an uncertainty proposal only if current RGB supports
  doing so, with no success claim and within the same budget.
- Close ALL live trials in a batch before reading any privileged audit. Report
  detection, recovery effort, gate acceptance, nominal handoff and actual full
  task completion separately. Never count a recorded recovery replay as another
  independent Agent success.

## Execution shape

1. Run `python3 cr.py verify`, `doctor`, and `test`. Check H20 paths/permissions.
2. For integration regression, `smoke` explicitly replays the recorded seed183173
   recovery and checks RGB/gate. For new evaluation use `live`, not replay.
3. Start the isolated loopback Cosmos server on an idle GPU/port if nominal
   continuation is needed. Do not stop or alter unrelated jobs/environments.
4. Use current `agent_observations.jsonl` and its PNGs; send unique decision IDs
   with matching `evidence_frame`. Let one request finish before sending another.
5. After handoff, request fresh Cosmos chunks and review RGB until success
   candidate, renewed failure, unsafe/unsupported scene, or budget. Do not stop
   merely because the handoff call succeeded.
6. Send `finish`, audit after all live trials close, then annotate the video.

## Evidence limits

Frozen v4.1 currently has delayed/unknown grasp outcomes, color/template identity
limitations and provisional pregrasp geometry. The latest three historical C2
trials had two accepted handoffs and zero full stable pickups. These results do
not isolate whether failure is entirely due to Cosmos or its compatibility with
the recovery-produced starting state. Use this as a reproducible development
harness, not a validated successful pickup skill.
