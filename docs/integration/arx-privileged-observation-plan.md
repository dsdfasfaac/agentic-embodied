# ARX privileged observation and gateway plan

Status: proposed implementation plan. This document deliberately does not
change the current RGB-only ARX contract (`arx_rgb_public_v1`).

## 1. Current MuJoCo state surface

`robots/arx/environment.py` owns the prepared `mujoco.MjModel` and
`mujoco.MjData`. At every reset/step it can read, after `mj_forward`/`mj_step`:

- `data.qpos`, `data.qvel`, `data.time`, and mapped robot joint/actuator values;
- world-frame body poses in `data.xpos` and `data.xmat` (including the logical
  target body resolved by `metadata.json.logical_body_map`);
- site/camera/geom transforms (`data.site_xpos`, `data.site_xmat`, camera and
  geom model tables);
- active contacts through `data.contact[:data.ncon]`, with geom/body IDs and
  contact forces available through MuJoCo contact-force APIs;
- task evaluator state already derived from those values: target height/lift,
  bilateral finger contact, workspace/drop failure, hold count, progress, and
  terminal success.

The current observation intentionally exports only 14-D command/proprioceptive
state and RGB. `DirectBackend._commit` places evaluator output in
`StepCommit.private_evaluation`; `ArxSessionCore._publish` serializes only RGB
references. This is the exact insertion point for a privileged path.

## 2. Information boundary

Add an explicit, immutable `PrivilegedObservation` owned by the trusted episode
executor. It must never be added to `PolicyObservation`, the existing public
observation, or the RGB critic package ABI by accident.

Proposed schema: `arx.privileged.observation.v1`, keyed by observation ID and
step index, containing finite JSON values only:

- `selected`: the task-contract-selected manipulated object and target, each with
  logical ID, world position, and only the orientation fields required by the
  pickup evaluator; include derived object-to-target and object-to-EEF distances.
- `interaction`: trusted booleans/scalars derived from contacts and evaluator
  state: gripper contact, robot contact, grasped, retained, released-now,
  released-ever, in-target, lift/progress, stage, and task status.
- `contact_summary`: bounded robot/gripper contact counts, force availability,
  and maximum normal force. Do not expose an unbounded raw contact-pair list.
- `joint_summary`: normalized values for the small, task-relevant mapped joints
  and joint count. Do not export raw `qpos`/`qvel` arrays.
- `simulation`: time and finite-state diagnostics, excluding filesystem paths,
  seeds, RNG state, model bytes, and raw unbounded arrays.

All fields above must be directly derivable from existing `MjModel`/`MjData`,
the frozen logical-body map, and the existing pickup evaluator. No new simulator
sensor, learned estimator, or agent-authored feature extraction is required for
the initial path. Logical names come from the frozen scene metadata/task
contract, never from an agent-supplied body name. Pose conventions (world frame,
quaternion order, units) and numeric bounds must be documented and validated in
the contract.

## 3. Backend and gateway wiring

1. Extend `StepCommit` with an optional privileged payload and make
   `DirectBackend` construct it from trusted environment/evaluator helpers after
   each reset and step. Copy arrays before crossing the executor boundary;
   reject NaN/Infinity and oversized contact/object maps.
2. Add a session-core mode, frozen at episode creation, such as
   `observation_visibility = rgb_public | privileged_agent`. In privileged mode
   `_publish` adds a `privileged` envelope to the agent snapshot/events and
   journals it under a separate internal record kind. In RGB mode the field is
   absent and attempting to request it is a contract error.
3. Add strict Pydantic contracts and catalog metadata advertising the schema,
   visibility mode, feature names, units, and maximum payload size. Do not reuse
   the permissive private evaluation dictionary as an API.
4. Keep Critic and Recovery on the original CandidateBundle path. They receive
   the same candidate-defined feature extractor interface, but its observation
   input is selected by campaign policy: RGB-only for the existing path, or the
   validated privileged feature projection for the new path. They must not get a
   live `MjModel`, `MjData`, evaluator object, or arbitrary private dictionary.
5. Expose privileged values through the existing `/observation` and event flow,
   not a second simulator-control endpoint. Add redaction/publication guards so
   `record_public_trajectory` continues to reject privileged fields under the
   RGB policy and writes a separate authorized privileged artifact under the
   privileged policy.

## 3a. Required file changes

Implement the path as one vertical slice across these files (plus focused tests):

- `robots/arx/environment.py`: add a trusted helper that resolves the frozen
  manipulated-object/target body IDs, reads their `xpos`/orientation, computes
  EEF/object distances, normalized mapped-joint values, contact summaries, and
  pickup interaction state. Keep extraction adjacent to the existing evaluator
  and return a bounded plain mapping.
- `robots/arx/gateway/backend.py`: add `PrivilegedObservation` and attach one
  copied payload to every `StepCommit` from reset and step.
- `robots/arx/gateway/contracts.py`: define strict versioned privileged schemas,
  visibility mode, feature names, units, and payload bounds.
- `robots/arx/gateway/session_core.py`: retain the payload per observation ID,
  include it only in privileged snapshots/events, and journal a separate
  privileged record with matching sequence/step IDs.
- `robots/arx/gateway/service.py`, `client.py`, `episode.py`, and `worker.py`:
  transport and validate the mode without exposing environment handles; make
  stale/replayed privileged observations fail like RGB observations.
- `robots/arx/critics/contracts.py`, `registry.py`, `feature_worker.py`, and
  `isolation.py`: add the privileged feature ABI and pass only the bounded
  mapping to `TemporalCritic`; preserve sandbox limits and reject undeclared
  fields. Add a privileged reentry implementation alongside the RGB threshold
  implementation.
- `robots/arx/deployment/agent.py` and `runner.py`: update the frozen prompt/event
  schema so Role1Agent can read privileged fields, while still requiring
  `review_reentry` and a fresh VLA decision after recovery.
- `robots/arx/gateway/recording.py` and `trajectory_recorder.py`: export camera
  MP4s plus privileged JSONL/state timelines; hash and index every artifact.
- `zetta/evolution/evidence_policy.py`, `zetta/evolution/visual_artifacts.py`,
  `zetta/evolution/lifecycle.py`, and `zetta/evolution/trajectory.py`: register
  the privileged policy, allow only its bounded fields, and preserve RGB
  fail-closed publication.
- `scripts/deployment/serve_arx_gateway.py` and
  `scripts/deployment/run_arx_evolution_rollout.py`: accept frozen visibility,
  load the structured candidate path, configure `TemporalCritic` plus Role1Agent,
  and enforce campaign identity/policy digests.
- `scripts/evolution/prepare_arx_campaign.py`: add the privileged mode flag,
  structured candidate kind, privileged policy, schema/catalog digests, and
  rollout placeholders to the immutable manifest.
- `zetta/evolution/candidate_artifacts.py`, `store.py`, `gate_runner.py`, and
  `supervisor.py`: resolve/register `CandidateBundle` artifacts for this mode
  and retain existing gate transitions.

## 3b. Concrete rule bundle and execution contract

Add a fixture such as `tests/fixtures/arx_candidate_privileged_v1/` containing a
structured `CandidateBundle` with one critic rule:

```text
activation: interaction.gripper_closed == true
feature: selected.target_gripper_distance_m
operator: gt
threshold: 0.10
proposal: interrupt
```

Its recovery binding must authorize, in order: `arx.set_gripper(open)`,
`arx.move_eef` with a 0.02 m translation toward the selected gripper/object
direction (bounded by the existing EEF validator), then `arx.review_reentry`.
After a successful review, `Role1Agent` resumes with `arx.zeva` using the fresh
reentry token. The direction is computed by trusted gateway motion code from the
privileged object/EEF positions; the bundle contains no seed-specific
coordinates or direct simulator access.

The integration test must instantiate `TemporalCritic` with this rule and the
real Role1Agent/runner adapter, trigger it on a deterministic observation, verify
the critic proposal, execute both recovery tool calls, verify the reentry token,
and verify the next decision is VLA continuation rather than a second recovery.

## 4. Rollout integration

`run_arx_evolution_rollout.py` should accept a frozen visibility/evidence-policy
choice from the campaign manifest. Remove the hard-coded RGB-only assertion only
after validating the new policy and its candidate kind. The rollout must record
the selected mode in trial identities, reject a campaign/CLI mismatch, and keep
authoritative success evaluation in the trusted gateway journal.

The deployment agent sees privileged observations at every step only when the
campaign explicitly selects the privileged mode. Critic and Recovery continue
to be invoked through CandidateBundle-compatible interfaces and receive the
same step IDs/history, preventing hidden alternate state channels.

## 5. Tests and acceptance criteria

- Unit-test extraction against a tiny MuJoCo model: object pose, quaternion
  convention, contact force bounds, evaluator progress, and reset/step alignment.
- Contract-test that RGB mode has no privileged key and fails closed on a request
  for it; privileged mode has no raw arrays, paths, seeds, or model handles.
- Gateway tests verify per-step IDs, immutable copies, schema/catalog digests,
  payload limits, and event replay ordering.
- Publication tests prove `arx_rgb_public_v1` rejects privileged output while a
  new privileged policy emits a content-addressed, bounded state stream.
- End-to-end rollout tests verify the agent receives poses, Critic/Recovery use
  CandidateBundle feature extraction, and authoritative outcome is unchanged.
- Preserve all existing RGB campaign tests and make the new mode opt-in.

## 6. Final end-to-end acceptance run

After unit, contract, and fake-gateway tests pass, run the privileged campaign
against every scene under `runs/arx_pickup_test_tube_10_new` (the ten
`episode_*` directories), using the prepared structured bundle and the frozen
privileged campaign manifest. For each scene:

1. Start the ARX gateway/Zeva services with the privileged visibility mode.
2. Execute one fresh rollout through
   `scripts/deployment/run_arx_evolution_rollout.py`.
3. Record front, wrist, right, and combined camera videos using
   `robots/arx/gateway/recording.py`.
4. Assert that every step has matching observation ID, privileged state,
   critic proposal/result, tool request/result, recovery transition, and VLA
   reentry record; all payloads and artifacts are content-hashed.
5. Assert that the rule fires exactly when gripper-closed and distance exceeds
   10 cm, opens the gripper, moves EEF 2 cm toward the gripper/object, then
   resumes VLA through a valid reentry token. Assert no duplicate motion occurs
   after uncertainty and no privileged fields leak into RGB-public artifacts.
6. Run video decoding/frame-count/hash checks and write one per-scene audit plus
   an aggregate report identifying missing records, incorrect rule execution, or
   infrastructure failures.

The campaign is accepted only if all ten scenes produce complete recordings and
audits, the rule execution trace is correct in every applicable scene, and the
authoritative evaluator result and privileged timeline agree with the journal.
