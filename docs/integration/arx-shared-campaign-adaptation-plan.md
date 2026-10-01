# ARX adaptation to the shared Zetta evolution campaign

## Decision

ARX should reuse the durable Zetta campaign, queue, gate, and promotion engine,
but it cannot be connected by configuration alone. Two seams must first become
environment-neutral:

1. candidate storage/execution must support ARX's immutable multi-file package,
   rather than assuming every candidate is a JSON `CandidateBundle`; and
2. learning evidence must be governed by a frozen visibility policy, rather
   than the current LIBERO-oriented privileged-state allowlists.

The ARX rollout implementation remains an adapter below those seams. It owns
fresh MuJoCo gateway episodes, remote Zeva calls, RGB feature extraction,
critic interrupts, recovery decisions, and conversion of an attempt into a
shared `EpisodeRecord`. It does not own campaign phases, queues, gates, or
promotion.

```text
prepare_arx_campaign
  -> CampaignManifest + prompt/task/tool/evidence contracts
  -> EvolutionSupervisor / CampaignStore / SharedHostQueue
  -> ARX rollout command (ordinary subprocess; no Ray/env-slot lease)
  -> RolloutRunner -> local one-use MuJoCo gateway -> remote Zeva server
  -> ARX result/evidence adapter -> EpisodeRecord + public artifact index
  -> shared cluster/diagnose/propose phases through an ARX candidate adapter
  -> shared same-seed/regression/held-out gates
  -> shared atomic promotion and child-generation handoff
```

## Audit result

### Directly reusable

- `scripts/evolution/run_campaign.py` and
  `zetta/evolution/supervisor.py`: phase orchestration has no Ray or benchmark
  dependency.
- `zetta/evolution/queue.py`: jobs are ordinary subprocess commands. Ray is not
  required. `requires_environment_slot=False` prevents use of the RoboCasa
  environment-slot broker and forbids an `{env_endpoint}` token. The worker can
  call a remote VLA from inside the ARX rollout process.
- queue claim fencing, heartbeats, watchdogs, infrastructure retry accounting,
  append-only attempt ingestion, and fail-closed blocked-job handling.
- seed/policy-RNG preregistration, provided ARX distinguishes a scheduled policy
  RNG from whether the remote Zeva provider actually guarantees determinism.
- `EpisodeRecord`, campaign ledgers, failure clusters, paired success statistics,
  gate state transitions, and promotion semantics.
- provider admission for online recovery-model calls. This admission is for the
  Role1/API provider, not for Zeva. It is enabled only on candidate arms.

### Reusable after ARX produces the required contract

- `zetta/evolution/trajectory.py` requires complete `chunks`, `actions`,
  `states`, and `tools` JSONL files plus synchronized visual artifacts. ARX can
  produce these, but its current attempt directory has different names/shapes.
- diagnosis and cluster artifact reading is reusable once ARX publishes a
  seed-blind, path-blind, public content-addressed artifact index.
- paired gates are reusable once the ARX command consumes all frozen fields and
  produces an `EpisodeRecord` whose seed, policy RNG, bundle digest, initial
  observation identity, and intervention metadata agree with the gate plan.

### Incompatible assumptions that must be changed

1. `SafetyLayerConfig` has historical RoboCasa defaults, including
   `robocasa_named_action_v1`. `prepare_robocasa_campaign.py` uses those defaults;
   `prepare_libero_campaign.py` explicitly overrides all five fields with
   LIBERO-specific identifiers. ARX preparation must likewise supply all five
   ARX-appropriate identifiers rather than inherit the RoboCasa values.
2. `CampaignStore`, `resolve_bundle_file`, `CandidateGateRunner`, promotion,
   child-generation handoff, shadow replay, and lifecycle refinement assume a
   canonical JSON file at `candidates/<sha>/bundle.json` parseable as
   `CandidateBundle`. ARX uses a hash-addressed directory containing a manifest,
   executable RGB feature code, replay fixtures, skill, and bindings.
3. shared Stage2 emits scalar `CriticRule` plus fixed `RecoveryRule.steps`.
   ARX needs an RGB extractor/temporal-rule package and an adaptive recovery
   skill. Treating that as `tool_plugin` would bypass the shared validations and
   is not acceptable.
4. lifecycle telemetry and prompts contain LIBERO privileged-state behavior.
   `trajectory._progress_value` also recognizes goal-distance residuals. These
   are optional rather than universally required, but without an explicit ARX
   policy they can leak simulator-derived learning evidence.
5. visual artifacts optionally generate a privileged-state summary. ARX must
   freeze this off, and the evidence publisher must reject—not merely omit from
   prompts—private simulator fields.
6. the standard task contract validator assumes `suite`, `task`, and `language`.
   The first ARX test uses `runs/arx_pickup_test_tube_10_new/scenes.json` and
   its per-scene `task.yaml` files. Preparation must derive one shared task
   identity from their common task name/instruction and separately freeze each
   scene's model, mapping, task, and evaluator provenance. Future ARX scene
   sets follow the same registry/task-file format; do not introduce a second
   campaign task identity.
7. formal lifecycle code requires failures to contain failure segments and
   formal campaigns to contain synchronized images/video. ARX result conversion
   must materialize them before queue ingestion.
8. held-out gate code supports the formal 20-episode fixed gate or the 10/50
   two-stage gate. The first ARX campaign should use 20 preregistered held-out
   trials.

## Non-negotiable ARX information boundary

Set `runtime.evidence_policy` to `arx_rgb_public_v1`. Under this policy:

- online critic features are derived only from current/past public RGB and
  public execution acknowledgements;
- the recovery agent sees current RGB, critic proposals, bounded public history,
  tool schemas/results, and remaining budgets;
- the offline Cluster/Diagnoser/Evolver sees the same public artifacts plus
  infrastructure error classifications;
- MuJoCo qpos/qvel, contacts, object poses, target coordinates, reward, goal
  residuals, seeds, evaluator internals, capability files, and gateway journals
  are private;
- the evaluator may privately compute task success, but publishes only the
  authoritative boolean and terminal reason;
- private values may be used by the harness for integrity/evaluator checks, but
  never as candidate features, recovery inputs, or learning evidence;
- candidate preflight rejects feature names or code imports that request the
  private ABI. Runtime isolation remains the final enforcement boundary.

The policy must be enforced twice: while publishing attempt artifacts and while
resolving evidence for an agent. Prompt instructions alone are not enforcement.

## Interface design

### 1. Candidate artifact adapter

Add an environment-neutral protocol in
`zetta/evolution/candidate_artifacts.py`:

```python
class CandidateArtifactAdapter(Protocol):
    kind: str
    def author(...diagnosis/evidence/tool contract...) -> CandidateRef: ...
    def validate(candidate: CandidateRef, *, parent: CandidateRef | None) -> ValidationReport: ...
    def resolve(root: Path, sha256: str) -> Path: ...
    def public_summary(candidate: CandidateRef) -> dict: ...
    def shadow_replay(candidate: CandidateRef, evidence: EvidenceView) -> ReplayReport: ...
```

`CandidateRef` is a small shared record containing `sha256`, `kind`, `path`,
`parent_sha256`, `diagnosis_sha256`, `generation`, and public mechanism/
validation text. Existing environments use `kind=structured_bundle_v1` and an
adapter wrapping `CandidateBundle`. ARX uses `kind=arx_rgb_package_v1` and an
adapter wrapping `robots.arx.critics.packages`.

Campaign state, gate plans, and promotion continue to bind the candidate SHA.
They must not depend on its internal representation.

### 2. Evidence policy adapter

Add `zetta/evolution/evidence_policy.py`:

```python
class EvidencePolicy(Protocol):
    policy_id: str
    def publish_attempt(attempt_dir: Path, destination: Path) -> ArtifactIndex: ...
    def validate_public_event(event: Mapping[str, Any]) -> None: ...
    def validate_candidate_feature(name: str) -> None: ...
    def resolve(content_id: str, selector: object | None) -> AgentArtifact: ...
```

Keep current behavior behind `libero_privileged_critic_v1` and the RoboCasa
policy selected by its manifest. Implement `arx_rgb_public_v1` as a strict
allowlist. Unknown fields fail publication instead of being silently copied.

### 3. ARX rollout command

Before creating rollout commands, `prepare_arx_campaign.py` must load the
prepared scene registry and construct the authoritative shared task contract.
For the initial fixture `runs/arx_pickup_test_tube_10_new/scenes.json`, each
entry points to `episode_XXXXXX/`, its `mapping.json`, and its `task.yaml`
(JSON-formatted despite the suffix). Use a structured parser, not text matching.
Require a unique episode/scene ID; existing scene directory, `model.mjb`,
`mapping.json`, `task.yaml`, and `metadata.json`; and agreement between
`task.yaml.name`, `metadata.json.task_name`, and the requested campaign task.
Require a single nonempty instruction, task ID, task schema version, and
success/failure evaluator IDs across this campaign's entries; reject mixed
identities rather than silently choosing the first entry. Scene-specific
`start_state` and `starting_scenes` are not part of the public task identity.

For this fixture, set `CampaignManifest.task` and `runtime.task_contract.task`
to `pickup_test_tube`, `suite` to `arx_mujoco_task7`, and `language` to the
shared `task.yaml.instruction` (`Pick up test tube with the pink label.`).
Compute `normalized_language` with the shared whitespace/casefold convention
and `language_sha256` with `canonical_sha256({"language": normalized_language})`.
Include the shared task ID and task schema, evaluator IDs, and a digest of the
frozen evaluator implementation/contract in `task_contract`. Record the same
contract in `task-contract.json`, `runtime.task_contract`, and preregistration,
including its canonical digest; ensure the lifecycle's
`_authoritative_task_contract` accepts it without special ARX parsing.

Freeze a separate per-scene registry for the rollout resolver: scene ID and
split, plus byte SHA256 digests of `model.mjb`, `mapping.json`, `task.yaml`,
`metadata.json`, and the frozen model contract; include `scene.xml` and
`reset_state.npz` hashes when these are inputs to reset or integrity checks.
The task files vary by scene (for example `start_state`), so one digest of
`entries[0]['task']` is **not** a campaign-wide evaluator digest. At rollout,
resolve the preregistered entry and check its digests before constructing the
ARX `Trial`; bind the resolved scene/reset identity to the attempt for paired
gates. Keep paths, reset state, and evaluator parameters in harness-private
provenance; publish only allowed identity hashes and task language under the
ARX evidence policy. The ten supplied entries are the scene pool for both
training and held-out trials. Preregister 20 held-out trials over these scenes
with independent resets and disjoint trial seeds; do not require 20 distinct
scenes. Verify reset identity before claiming a paired comparison. Formal
held-out testing is not the first implementation milestone.

`scripts/deployment/run_arx_evolution_rollout.py` becomes the queue-facing
adapter. Required CLI fields:

```text
--campaign-root --logical-id --attempt-index --generation
--task --seed --policy-rng --bundle --bundle-sha256 --baseline-mode
--scene-registry --scene-block --vla-host --vla-port
--output-dir --result-file --heartbeat-file
--runtime-limits --runner-limits --evidence-policy
```

It resolves the preregistered scene by split block and logical index, writes the
existing ARX `Trial`, runs `RolloutRunner`, then converts the result. It must
never start Zeva, request a Ray environment, or accept `{env_endpoint}`.

The converter writes:

- `episode_record.json` (`EpisodeRecord`);
- `trajectory/chunks.jsonl`, `actions.jsonl`, `states.jsonl`, `tools.jsonl`;
- public RGB frames/contact sheets/video and their hashes;
- a public intervention summary;
- private evaluator/gateway artifacts below `private/`, excluded by policy;
- a heartbeat record during gateway startup, nominal chunks, recovery calls,
  reconciliation, and artifact finalization.

### 4. Outcome mapping

Map ARX result status as follows:

| ARX result | Shared record |
| --- | --- |
| `completed` with evaluator boolean | `status=valid`, that boolean as `success` |
| configuration/infrastructure/agent error before a certain write | `status=infra_invalid`, `success=None` |
| execution uncertain | `status=infra_invalid`, `success=None`; never retry automatically after an uncertain write |
| interrupted | `infra_invalid`, unless a completed authoritative result was already sealed |

A valid failed episode must include at least one public failure segment. When
RGB evidence cannot localize onset, use `earliest_divergence_step=None`; do not
invent step zero.

### 5. Remote Zeva topology

Zeva is a frozen external service dependency:

```json
{
  "mode": "external",
  "host": "<deployment host>",
  "port": 5581,
  "checkpoint_id": "<attested id>",
  "modality_contract": "arx_task7_rgb_action_v1",
  "maximum_inflight": 1,
  "identity_attestation": "required-before-formal-run"
}
```

The campaign preparer probes/records identity before freezing the manifest.
Workers only connect. VLA concurrency is controlled by the external service
contract or an ARX VLA admission semaphore, not by Ray slots and not by
`requires_api` (which is reserved for recovery-model admission).

## Concrete configuration for the first formal campaign

Use these manifest choices:

```json
{
  "environment": "arx_mujoco",
  "baseline_mode": "strict_pure_vla",
  "expected_rollouts": 10,
  "expected_heldout": 20,
  "initial_logical_slots": 1,
  "continuous_logical_slots": 1,
  "maximum_logical_slots": 2,
  "maximum_api_concurrency": 1,
  "episode_timeout_s": 1800,
  "no_progress_timeout_s": 600,
  "max_infrastructure_attempts": 2,
  "safety_layer": {
    "action_contract": "arx_gateway_named_tools_v1",
    "control_limits": "arx_task7_bounded_operation_limits_v1",
    "simulation_health": "arx_finite_state_and_gateway_lease_v1",
    "joint_limit_shield": "not_implemented",
    "contact_policy": "environment_native_only"
  },
  "runtime": {
    "candidate_kind": "arx_rgb_package_v1",
    "evidence_policy": "arx_rgb_public_v1",
    "rollout_requires_environment_slot": false,
    "rollout_requires_api": false,
    "candidate_rollout_requires_api": true,
    "heldout_gate_kind": "heldout_20",
    "vla_service_mode": "external",
    "vla_maximum_inflight": 1,
    "reuse_rollout_parent_evidence": true
  }
}
```

`safety_layer` is a required manifest record, not a runtime safety controller:
the current `SafetyLayerConfig` validator only requires five nonempty strings.
Set these identifiers explicitly in `prepare_arx_campaign.py` for both baseline
and candidate arms, and mirror them in preregistration as the LIBERO preparer
does. Map each identifier to an existing, tested ARX runtime check before
claiming it: gateway tool schemas and motion bounds, execution budgets and
lease/health checks, and native contact behavior. In particular, keep
`joint_limit_shield=not_implemented` unless a model/controller joint-limit
shield is verified and covered by tests; change the identifier only alongside
that evidence. These strings must describe enforcement performed by the ARX
gateway/controller, not replace it. Preserve historical manifest decoding and
RoboCasa defaults for existing campaigns.

`initial_logical_slots` here controls queue population, not Ray workers. Start a
normal shared-queue worker with concurrency one. Increase to two only after the
remote Zeva server and distinct gateway-port allocator pass a concurrency test.

Generate disjoint train and held-out schedules with
`preregister_seed_schedule`. Bind each schedule entry to a scene/reset digest.
Keep `policy_rng` in the shared record as the requested/scheduled RNG. Also put
`policy_rng_effective`, provider stochasticity mode, and Zeva request identity in
private integrity metadata. Do not claim deterministic pairing if Zeva cannot
attest it; the gate compares paired frozen conditions while reporting the known
stochasticity limitation.

Use separate planner configurations:

- offline Cluster/Diagnoser/Evolver: campaign `model` and `reasoning_effort`;
- online ARX recovery: `planner_type=api`, its own model/token/timeout settings;
- never copy the offline learner's `planner_type=codex` into deployment
  `AgentSettings`.

## File-level implementation plan

### Shared evolution code

- `zetta/evolution/models.py`
  - add `CandidateRef` and manifest validation for `runtime.candidate_kind` and
    `runtime.evidence_policy`;
  - change the default safety layer only by requiring preparers to materialize
    explicit values; preserve old manifest compatibility.
- `zetta/evolution/candidate_artifacts.py` (new)
  - define adapters/registry and structured-bundle compatibility adapter.
- `zetta/evolution/evidence_policy.py` (new)
  - define policy protocol/registry and common seed/path/private-key checks.
- `zetta/evolution/store.py`
  - register, recover, verify, promote, and copy `CandidateRef` artifacts through
    the adapter;
  - retain existing JSON bundle layout for Libero/RoboCasa;
  - copy ARX package trees immutably and atomically without following symlinks.
- `zetta/evolution/campaign.py`
  - replace `resolve_bundle_file` with `resolve_candidate_artifact`;
  - retain `{bundle_file}` as a compatibility alias and add canonical
    `{candidate_path}`/`{candidate_kind}` substitutions.
- `zetta/evolution/gate_runner.py`
  - load candidate metadata through the adapter;
  - render candidate path/kind for both arms;
  - keep paired plan, retry, identity, and statistics unchanged.
- `zetta/evolution/supervisor.py`
  - make child-generation artifact handoff adapter-driven instead of copying
    `bundle.json` directly.
- `zetta/evolution/lifecycle.py`
  - select evidence and candidate adapters from the manifest;
  - remove unconditional JSON-bundle parsing from generic paths;
  - route proposal/preflight/refinement/shadow replay to the selected adapter;
  - apply the evidence policy before constructing any agent artifact index;
  - keep phase transitions and diagnosis records shared.
- `zetta/evolution/stages.py`
  - split the universal role/authority prompt from candidate-kind supplements;
  - add an explicit ARX supplement prohibiting all privileged simulator data;
  - make proposal output schema adapter-provided;
  - keep bounded evidence reads and evidence-ID validation.
- `zetta/evolution/trajectory.py`
  - accept a visibility/feature policy during indexing;
  - disable implicit goal-residual/contact progress for `arx_rgb_public_v1`;
  - validate that ARX public state rows contain only RGB-derived features and
    public execution acknowledgements.
- `zetta/evolution/visual_artifacts.py`
  - keep privileged summary opt-in;
  - validate it is impossible under `arx_rgb_public_v1`;
  - support ARX frame/window names without LIBERO-specific labels.
- `zetta/evolution/shadow_replay.py`
  - dispatch replay to the candidate adapter. Structured rules retain the
    existing implementation; ARX runs package replay cases in isolation.
- `zetta/evolution/fault_injection.py`
  - replace direct `bundle.json` assumptions with adapter resolution.

### ARX campaign adapter

- `scripts/evolution/prepare_arx_campaign.py`
  - replace the separate `CampaignContract` output with standard manifest,
    prompt contract, task contract, tool catalog, seed/scene schedule, evidence
    policy, candidate-kind, remote VLA identity, rollout commands, and formal
    gate policy;
  - parse the `scenes.json` + per-scene `task.yaml` layout used by
    `runs/arx_pickup_test_tube_10_new`; validate shared task identity, write
    matching manifest/sidecar/preregistration task contracts and a hash-bound
    per-scene split registry; replace the current first-task-file evaluator
    digest and reject insufficient/disjointness-violating gate schedules;
  - explicitly populate all five ARX `SafetyLayerConfig` fields and record the
    same identifiers in preregistration; do not inherit RoboCasa defaults;
  - populate train, regression, and held-out scene blocks;
  - remove hard-coded Python paths and local Zeva startup settings.
- `robots/arx/evolution_defaults.py` (new)
  - centralize safety IDs, timeouts, evidence policy ID, candidate kind, gate
    defaults, and formal rollout limits.
- `robots/arx/evolution_candidate_adapter.py` (new)
  - author/seal/load/validate `arx_rgb_package_v1` packages;
  - enforce parent inheritance, RGB-only ABI, replay cases, tool bindings,
    skill/reentry completeness, and package hash identity.
- `robots/arx/evolution_evidence.py` (new)
  - implement the strict public publisher/resolver and private-field rejection.
- `robots/arx/trajectory_recorder.py` (new)
  - normalize existing gateway/runner events into the four shared JSONL streams
    and synchronized RGB artifacts.
- `robots/arx/evolution_result.py` (new)
  - map `RolloutResult` to `EpisodeRecord`, classify retry safety, construct
    failure segments, and bind artifact hashes/initial observation identity.
- `scripts/deployment/run_arx_evolution_rollout.py`
  - implement the queue CLI contract, scene lookup, heartbeat, candidate
    resolution, recorder/finalizer, and atomic result write.
- `robots/arx/deployment/runner.py`
  - expose structured callbacks or a sealed event stream for the recorder;
  - keep private gateway artifacts separate;
  - do not expose simulator/evaluator state to candidate or agent.
- `robots/arx/deployment/contracts.py`
  - keep deployment `AgentSettings` independent from offline stage settings;
  - add frozen VLA identity/stochasticity metadata if it belongs to the trial.
- `robots/arx/critics/packages.py`
  - expose deterministic tree verification and public manifest summary required
    by the candidate adapter; retain isolation and no-symlink checks.
- `robots/arx/gateway/session_core.py`
  - no campaign responsibility; only ensure emitted public observations conform
    to the declared RGB/execution ABI and evaluator-private fields remain on the
    harness side.

### Retire after parity

- `scripts/evolution/run_arx_learning.py`
- `zetta/evolution/arx/session.py`
- `zetta/evolution/arx/coordinator.py`
- job/group/evaluation behavior in `zetta/evolution/arx/tools.py`
- the parallel checkpoint/store lifecycle in `zetta/evolution/arx/store.py`

Package validation helpers may move to the new adapter. Remove these paths only
after restart, gate, and artifact-equivalence tests pass.

## Tests and acceptance gates

Add or update:

- `tests/test_evolution_candidate_adapters.py`: both candidate kinds, immutable
  resolution, promotion, and child handoff; regression coverage for existing
  environments.
- `tests/test_arx_evidence_policy.py`: reject qpos/qvel/contact/pose/target/
  reward/residual/seed/path/capability fields at publication and read time.
- `tests/test_arx_campaign_prepare.py`: complete standard manifest, disjoint
  schedules, explicit safety contract, external VLA, no env-slot token; use the
  `arx_pickup_test_tube_10_new` registry/task layout to verify task-language
  consistency, per-scene digests, first-file digest rejection, private reset
  provenance, and failure on insufficient held-out coverage.
- `tests/test_arx_evolution_rollout.py`: baseline and candidate conversion to
  valid `EpisodeRecord`, heartbeat, artifact completeness, and failure mapping.
- `tests/test_arx_candidate_adapter.py`: package authoring, isolation, replay,
  parent inheritance, recovery coverage, and RGB-only imports/features.
- `tests/test_candidate_gate_runner.py`: ARX directory candidate, no environment
  lease, parent/candidate command rendering, identical reset identity.
- `tests/test_evolution_lifecycle.py`: ARX prompt/schema selection and absence of
  privileged evidence; existing Libero/RoboCasa behavior unchanged.
- `tests/test_evolution_queue_recovery.py`: ARX subprocess crash, published-result
  recovery, uncertain-write no-auto-retry, and remote-VLA outage classification.
- `tests/test_arx_shared_campaign_e2e.py` (new): fake remote Zeva + fake provider
  completes prepare -> baseline -> evidence -> diagnosis -> package -> replay ->
  same-seed -> regression -> held-out -> promotion, then repeats from checkpoints.

Release criteria:

1. byte-for-byte unchanged existing Libero/RoboCasa fixture results;
2. zero private-field reads in all ARX agent access logs;
3. no Ray or environment-slot broker process in the ARX E2E process tree;
4. no campaign-owned Zeva startup;
5. every valid ARX failure has complete public artifacts and a segment;
6. every gate arm binds scene/reset, requested RNG, VLA identity, safety, package,
   and evaluator identity;
7. crash/restart does not duplicate a physical attempt or lose a sealed result;
8. promotion is possible only through `CampaignStore.promote` after all formal
   gates.

## Current verification snapshot

### Implementation progress (2026-09-28)

- Initial candidate references resolve structured bundles and ARX package
  directories; campaign and gate commands accept `candidate_path` and
  `candidate_kind`. Store registration, verification, and promotion accept ARX
  packages. The 28 focused campaign, gate, and adapter tests pass.
- Added a strict ARX public-event policy, RGB-only package feature-name
  validation, fixture-backed task/scene and manifest preparation, preregistered
  ten-scene/20-reset scheduling, frozen queue command and trial construction,
  gateway public-RGB recorder, private evaluator extraction, and shared outcome
  mapping. These paths have focused tests; no full MuJoCo campaign has run.
- Implemented in this pass: preparation honors optional Zeva/gateway
  configuration, records evaluator provenance, and keeps generation 0 pure VLA;
  legacy ARX learner writes and fault-injection recovery now use the candidate
  adapter boundary.
- Implemented since the previous snapshot: Stage2 now authors and registers
  executable RGB packages through `CandidateArtifactAdapter`, runs fixture and
  published-RGB shadow replay, and recovers registrations without interpreting
  ARX packages as scalar bundles. Preparation emits the exact frozen
  `tool-catalog.json` consumed by the gateway. Public artifact indexing rejects
  private simulator fields, and fault-injection/no-op/rejection paths use the
  candidate-kind adapter boundary. Failed package authoring is staged
  atomically so retries cannot be poisoned by partial drafts.
- Remaining validation is operational rather than an information blocker:
  start Zeva and run the ten baseline scenes, then exercise a generated package
  through same-seed and held-out gates. Formal external VLA identity attestation
  still depends on the deployment service's attestation endpoint; the local
  pipeline fails closed when it is unavailable. Focused shared/ARX tests now
  pass (including campaign preparation, artifact policy, adapter, lifecycle,
  gate, and queue-recovery coverage).

The current focused selection passes 60 tests, including the ten-scene/20-trial
preparer and shared lifecycle/gate/recovery suites. Critic worker integration
tests that require unrestricted subprocess/socket behavior remain environment-
dependent in the restricted sandbox. A real end-to-end campaign still requires
the user-provided Zeva server and therefore cannot be claimed from unit tests.
