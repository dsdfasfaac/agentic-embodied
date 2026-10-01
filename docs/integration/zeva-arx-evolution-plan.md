# Zeva ARX evolution integration plan

## Objective

Connect the ARX AC one + Zeva Task7 rollout to Zetta's existing auditable
evolution framework. The resulting system must learn and promote supervisory
critic/recovery bundles; it must not fine-tune Zeva online or replace Zetta's
campaign, evidence, actor-authority, gating, or promotion semantics.

The target online loop is:

```text
Zeva proposes an action chunk
  -> Harness evaluates the frozen Critic before every physical action
  -> accepted action is executed by Runtime
  -> rejected action is not executed
  -> online Role1 reviews the bounded evidence and frozen recovery rule
  -> Recovery Actor alone invokes allowlisted recovery tools
  -> official MuJoCo evaluator remains the sole success authority
```

Offline, the existing Zetta lifecycle remains:

```text
rollout -> cluster -> diagnose -> propose -> same-seed gate
        -> regression gate -> held-out gate -> atomic promotion
```

## Semantic invariants (must not change)

These are compatibility requirements, not design suggestions.

1. `zetta.evolution` remains the campaign source of truth. Do not create an
   ARX-specific lifecycle or promotion implementation.
2. A `CandidateBundle` is immutable and hash-addressed for an episode.
3. Only online Role1 participates in episode recovery. Cluster, Diagnoser, and
   Evolver remain offline campaign-control agents.
4. The Critic is read-only and proposal/rejection-only. It never writes the
   environment.
5. The Recovery Actor is the sole environment writer during recovery.
6. A rejected live action is not executed. Every rejecting critic rule must be
   covered by an executable frozen recovery rule.
7. Recovery calls only tools in the campaign's frozen tool catalog. No invented
   tool names, arbitrary joint vectors, generated Python, or direct MuJoCo writes.
8. The Actor and Zeva never receive privileged MuJoCo state or absolute target
   coordinates. Privileged values may be available to the Harness/Critic and
   offline diagnosis only, under an explicit campaign contract.
9. The environment's `terminated/truncated/is_success` output is authoritative.
   Visual judgment, Critic output, Role1 output, and planner text cannot declare
   success.
10. The safety layer is frozen and identical between baseline and candidate.
    Learned heuristics never become hard safety checks.
11. Same-seed comparisons use the exact scene, reset, policy RNG, task,
    checkpoint, cameras, and safety configuration on both arms.
12. Held-out identities are preregistered and hidden from online agents.
13. Infrastructure-invalid attempts are never scored as task failures and never
    silently disappear.
14. Promotion remains gated and atomic through `CampaignStore.promote`; a live
    improvement alone cannot activate a candidate.
15. The learned artifact is the supervisory `CandidateBundle`, not modified
    Zeva weights.

## Existing code to reuse unchanged

| Zetta interface | Existing owner | ARX use |
|---|---|---|
| `CampaignManifest`, `CandidateBundle`, `CriticRule`, `RecoveryRule`, `EpisodeRecord` | `zetta/evolution/models.py` | Use the existing schemas without ARX forks |
| `EvolutionProtocol` | `zetta/evolution/protocol.py` | Use existing rollout/same-seed/regression/held-out limits |
| `preregister_seed_schedule` | `zetta/evolution/schedule.py` | Generate frozen reset and policy-RNG schedules |
| `CampaignStore` | `zetta/evolution/store.py` | Persist state, candidates, gates, and promotions |
| `EvolutionSupervisor` | `zetta/evolution/supervisor.py` | Drive every campaign phase |
| queue and worker execution | `zetta/evolution/queue.py`, `zetta/evolution/campaign.py` | Enqueue ARX rollout commands and ingest results |
| failure segmentation/indexing | `zetta/evolution/trajectory.py` | Validate and index ARX trajectory artifacts |
| deterministic Critic engine | `zetta/evolution/critic.py` | Evaluate frozen ARX predicates per action |
| Role1/diagnosis/proposal stages | `zetta/evolution/stages.py`, `zetta/evolution/lifecycle.py` | Use existing agent roles and bundle synthesis |
| candidate gates | `zetta/evolution/gate_runner.py`, `zetta/evolution/gating.py` | Run paired ARX comparisons without new statistics |
| visual evidence | `zetta/evolution/visual_artifacts.py` | Produce synchronized ARX contact sheets and videos |
| fault injection | `zetta/evolution/fault_injection.py` | Exercise crash/retry/partial-artifact behavior |
| Role1 contract/decision persistence | `robots/robocasa/role1_agent.py` | Reuse the generic online Role1 contract and decision store |
| recovery orchestration | `robots/robocasa/recovery_controller.py` | Reuse lifecycle/control flow; inject ARX bindings |

Generic code should change only when an existing environment-neutral interface
cannot represent required ARX data. Such changes require tests proving LIBERO
and RoboCasa behavior is unchanged.

## ARX files to add

```text
robots/arx/evolution_defaults.py
robots/arx/critic_runtime.py
robots/arx/tool_catalog.py
robots/arx/tool_bindings.py
robots/arx/role1_recovery.py
robots/arx/trajectory_recorder.py
robots/arx/run_evolution_rollout.py
scripts/evolution/prepare_arx_campaign.py
scripts/evolution/prepare_arx_heldout_validation.py   # only if generic helper cannot be reused
```

The current `ArxAgenticTaskAdapter.run()` remains the simple non-evolution
entrypoint. Evolution episodes use `run_evolution_rollout.py`; do not overload
the single-episode CLI with campaign state.

## Phase 1: expose bounded Runtime execution

### Required behavior

The evolution runner must regain control at every physical-action boundary.
The current `run_zeva_policy()` retains control until terminal and is therefore
not suitable for Critic interception.

Reuse these existing APIs:

- `RuntimeGateway.create_sessions`
- `RuntimeGateway.reset`
- policy inference request construction and `Cosmos3EdgeArxPolicyCore`
- MuJoCo step/result normalization
- `ArxMujocoEnv.step`

Add a bounded execution seam to Runtime only if no existing operation can:

1. request one Zeva chunk without immediately executing all actions;
2. expose each proposed action to the Critic;
3. execute an accepted action exactly once;
4. return the new observation and environment verdict.

Prefer composing existing inference and environment-step operations in the ARX
rollout runner. If a Runtime API addition is unavoidable, make it generic (for
example, `infer_policy_chunk` plus `step_actions`) and preserve `policy_step`
unchanged for all existing callers.

### Acceptance tests

- rejected action leaves MuJoCo time, qpos, qvel, controls, and step index
  unchanged;
- accepted action advances exactly `1/15 s` policy time;
- termination stops the remaining chunk prefix;
- inference failure does not advance the environment;
- the next chunk uses the advanced state and newly rendered images;
- baseline mode with no candidate is behaviorally identical to today's direct
  Zeva rollout.

## Phase 2: ARX trajectory and evidence contract

Implement `robots/arx/trajectory_recorder.py` as a thin producer of the existing
`TrajectoryArtifacts`/`EpisodeRecord` input format. Finish each episode with
`index_episode_trajectory`; do not write a parallel ARX index format.

Record, at minimum:

```text
trajectory/states.jsonl
trajectory/actions.jsonl
trajectory/chunks.jsonl
trajectory/events.jsonl
trajectory/object_relative_state.json
videos/front_rgb.mp4
videos/left_rgb.mp4
videos/right_rgb.mp4
videos/three_view.mp4
episode_record.json
result.json
```

Per physical step, record:

- proposed raw Zeva action;
- accepted/rejected decision and matching critic rule ID;
- processed action after locks/filter/step limits;
- measured 14-D state and realized delta;
- front/left/right image references and hashes;
- target lift, right-finger contacts, gripper aperture, progress;
- environment `terminated`, `truncated`, `is_success`, and terminal reason;
- whether recovery owns the environment;
- recovery rule/tool invocation and Role1 decision reference.

Per chunk, record request ID, modality digest, checkpoint identity, state/image
hashes, raw horizon, executable prefix, latency, and server metadata.

Write to an attempt-local temporary directory, fsync append-only streams, then
publish the final `EpisodeRecord` only after trajectory validation succeeds.
Partial or malformed episodes must be infrastructure-invalid.

## Phase 3: ARX Critic feature schema

Add `extract_arx_critic_features(...)` in
`robots/arx/critic_runtime.py`, following
`robots/libero/critic_runtime.py`'s interface and naming conventions.

Start with deterministic scalar/boolean features that can be replayed offline:

### Deployable observation-derived features

- action magnitude and direction changes by arm/gripper;
- realized-vs-commanded joint delta;
- repeated/oscillating chunks;
- action saturation/filter activation;
- progress delta and no-progress duration;
- gripper close/open transitions;
- camera frame validity, frozen-frame detection, and observation timestamp skew;
- inference latency/error/reconnect status.

### Privileged Critic/offline-only features

- `tube_01` position, initial-relative lift, and workspace validity;
- left/right finger contact booleans;
- gripper-to-target distance summaries;
- grasp acquisition/retention transitions;
- target displacement/drop state;
- MuJoCo contact/force summaries.

Every privileged key must use the existing `privileged.*` convention and be
excluded from Role1 tool inputs except for the bounded evidence explicitly
carried by a fired Critic rule. Role1 must never query arbitrary privileged
state.

Add replay tests demonstrating identical feature sequences from an immutable
trajectory and live execution.

## Phase 4: frozen ARX recovery tool catalog

Implement `robots/arx/tool_catalog.py` using the same public catalog structure
as RoboCasa/LIBERO. Initially keep the catalog deliberately small:

1. `arx_reobserve` — read-only synchronized three-view observation.
2. `arx_discard_pending_chunk` — reject the unexecuted remainder; no motion.
3. `arx_fresh_zeva_chunk` — request a new chunk using the current observation
   and the immutable full task instruction.
4. `arx_open_right_gripper` — bounded actuator-space recovery through Runtime.
5. `arx_retract_right_arm` — audited relative joint-space retreat with fixed
   maximum deltas; no target coordinates exposed.
6. `arx_return_observation_pose` — audited, task-independent safe pose sequence.
7. `arx_wait` — bounded physics settling with continued Critic/terminal checks.
8. `arx_stop_episode` — safe failure termination, never success.

All motion tools must:

- execute through Runtime and the existing `ActionProcessor`/joint mapping;
- honor the same safety layer as baseline;
- have fixed parameter schemas and action/time bounds;
- check official termination after every physical step;
- refuse execution outside Recovery Actor ownership;
- emit complete action and evidence records.

Do not initially allow a generic `move_joints`, Cartesian pose, Python, XML, or
MuJoCo-state tool. Add a semantic recovery tool only as an isolated tool-plugin
candidate under the existing Zetta rule.

## Phase 5: Role1 Recovery Actor

Implement `ArxRole1RecoveryActor` in `robots/arx/role1_recovery.py`, following
`LiberoRole1RecoveryActor` and the shared `RecoveryController`.

Inputs are limited to:

- immutable task instruction;
- fired critic rule and bounded evidence;
- frozen matching recovery rule;
- public ARX tool schemas;
- current non-privileged three-view observation;
- remaining recovery and episode budgets.

Outputs use the existing Role1 decision schema and `Role1DecisionStore`. Tool
calls are validated against the frozen rule before dispatch. A model error,
unknown tool, parameter violation, budget exhaustion, or missing evidence fails
the recovery closed; none falls back to uncriticized VLA execution.

## Phase 6: ARX evolution rollout runner

Implement `robots/arx/run_evolution_rollout.py` by adapting the control structure
of `robots/libero/run_evolution_rollout.py`, not the pure-VLA runner.

The runner must:

1. validate the frozen campaign arguments and bundle digest;
2. validate exact repository, task, scene, camera, mapping, model, checkpoint,
   and safety identities;
3. create/reset one Runtime ARX session;
4. load the candidate bundle or explicit baseline mode;
5. request Zeva chunks and evaluate the frozen Critic per physical action;
6. hand rejected actions to Role1 + Recovery Actor;
7. return to Zeva only according to the frozen recovery stop condition;
8. record all trajectory/evidence artifacts;
9. use only the environment's terminal verdict for success;
10. publish an existing-schema `EpisodeRecord` and result sidecar.

Baseline execution must still pass through the same recorder and safety layer;
it merely has no candidate critic/recovery interventions.

## Phase 7: scene identity and schedule

Create an immutable scene catalog containing each approved Real2Sim starting
scene:

```json
{
  "schema_version": "zetta_arx_scene_catalog_v1",
  "task": "pickup_test_tube",
  "scenes": [
    {
      "scene_id": "...",
      "prepared_model_sha256": "...",
      "reset_state_sha256": "...",
      "composition_sha256": "...",
      "camera_config_sha256": "..."
    }
  ]
}
```

Use `preregister_seed_schedule` unchanged. Add a frozen, deterministic
`environment_seed -> scene_id` mapping to the campaign manifest/runtime
contract. A paired baseline/candidate logical ID must resolve to the same scene
and reset. Held-out scenes/seeds must never be selected dynamically by Role1.

Before formal learning, replace the estimated front-camera position and record
independent right-wrist validation, or explicitly preregister their provisional
status so every arm uses exactly the same camera contract.

## Phase 8: campaign preparation

Add `scripts/evolution/prepare_arx_campaign.py`, structurally matching
`prepare_libero_campaign.py`:

- create a secret-free output directory exactly once;
- load and validate `EvolutionProtocol`;
- preregister rollout/held-out seeds and policy RNGs;
- freeze task, scene catalog, camera, mapping, checkpoint, and server modality
  identities;
- write `prompt-contract.json` using existing Role1/cluster/diagnosis/proposal
  contracts;
- write the frozen ARX tool catalog;
- register an empty generation-0 `CandidateBundle` or validated parent bundle;
- construct the tokenized `run_evolution_rollout.py` command using the existing
  `{seed}`, `{policy_rng}`, `{logical_id}`, `{bundle_file}`, and result tokens;
- construct `CampaignManifest` using existing fields;
- bind hashes through `CampaignStore` and preregister the schedule.

Secrets such as service tokens remain inherited environment variables and must
not enter the manifest. The Cosmos endpoint may be frozen in the runtime
contract; its checkpoint and modality digest must be verified at episode start.

Use `EvolutionSupervisor` and the existing worker CLI unchanged to run the
campaign.

## Phase 9: generic offline stages and gates

No ARX-specific cluster, diagnosis, proposal, statistics, or promotion code is
planned. Supply environment-specific evidence through existing artifact/tool
catalog interfaces, then call:

- `analyze_failures`;
- `run_diagnosis_stage`;
- `run_proposal_stage`;
- `CandidateGateRunner`;
- `PairedGateRunner`;
- `CampaignStore.promote` / `promote_and_spawn_generation`.

Only add small prompt-contract clauses describing ARX feature names and action
semantics. Preserve the existing requirements for competing causal hypotheses,
evidence IDs, atomic candidates, intervention attribution, and inconclusive
diagnosis.

## Phase 10: validation ladder

### A. Unit and contract tests

- ARX critic feature extraction and replay equivalence;
- catalog/schema validation;
- Actor ownership and allowlist enforcement;
- rejected-action non-execution;
- recovery bounds and terminal interruption;
- candidate/checkpoint/scene hash mismatch rejection;
- trajectory completeness and partial-write rejection.

### B. Deterministic integration tests

- fake Zeva server, fixed scene, no candidate: exact direct-rollout parity;
- deterministic critic trigger rejects the expected action index;
- bounded recovery executes only its frozen steps;
- official success overrides remaining recovery/Zeva work;
- recovery cannot manufacture success;
- second chunk observes the post-recovery state and images.

### C. Failure injection

- Cosmos timeout, disconnect, malformed shape, NaN action, modality mismatch;
- environment worker crash during baseline and recovery;
- Role1 timeout/malformed decision;
- partial videos/JSONL/result publication;
- queue retry and duplicate ingestion;
- stale bundle, scene, camera, or checkpoint digest.

### D. Development campaign

- at least two approved training scenes and one held-out scene;
- small preregistered seed block;
- demonstrate rollout -> cluster -> diagnosis -> candidate -> same-seed gate;
- inspect intervention attribution and false positives manually;
- do not promote this smoke candidate.

### E. Formal campaign

- preregister the full protocol and identities;
- establish pure/baseline Zeva paired evidence;
- require recovery interventions to explain candidate rescues;
- run regression and held-out gates;
- promote only through the existing atomic store transition;
- start the child generation from the promoted bundle.

## Implementation order and review boundaries

1. Bounded inference/action execution seam and parity tests.
2. ARX trajectory recorder and existing-schema indexing.
3. Critic features and offline replay.
4. Frozen recovery tools and ownership tests.
5. Role1 recovery actor and controller integration.
6. ARX evolution rollout runner with fake Zeva integration tests.
7. Scene catalog and complete provenance binding.
8. Campaign preparation script.
9. Failure injection and queue/retry validation.
10. Small development campaign.
11. Live-checkpoint campaign and formal gates.

Keep these as separate commits. Do not combine environment geometry/camera
changes, checkpoint artifacts, recorded episodes, or candidate outputs with the
framework implementation commits.

## Definition of done

The ARX evolution path is complete only when all of the following hold:

- a baseline episode and candidate episode execute through the same Runtime and
  safety layer;
- the Critic is evaluated before every physical action;
- a rejected action provably does not alter the environment;
- Role1 can execute only a matching frozen recovery rule through allowlisted
  Actor-owned tools;
- complete ARX trajectories pass `index_episode_trajectory`;
- the generic supervisor completes all phases without ARX-specific lifecycle
  forks;
- paired gates bind identical scene/reset/policy/checkpoint identities;
- success is always the environment-owned `is_success` result;
- at least one candidate is rejected by a formal gate, proving fail-closed
  behavior, before any candidate is promoted;
- a promoted bundle is atomically inherited by a preregistered child generation;
- direct Zeva mode remains available and unchanged as a baseline/control path.
