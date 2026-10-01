# ARX privileged CandidateBundle campaign plan

Status: concrete implementation contract. This document defines the ARX
privileged campaign ABI and completion checklist.

## 0. Symmetry with RoboCasa and Libero

Use the common lifecycle: `prepare -> baseline -> analyze/diagnose -> propose
CandidateBundle -> validate -> candidate rollouts -> paired/regression/heldout
gates -> promote`. ARX differs only in its deployment adapter. Each job is one
direct invocation of `scripts/deployment/run_arx_evolution_rollout.py`; the
learner never calls the gateway or Zeva directly.

## 1. Goal and mode selection

Add an explicit `observation_visibility` (or equivalent) configuration to
`scripts/evolution/prepare_arx_campaign.py`, with two frozen modes:

- `rgb_public`: current `arx_rgb_public_v1` + `arx_rgb_package_v1` behavior;
- `privileged`: a new privileged evidence policy + structured
  `CandidateBundle` representation.

The default remains RGB. Preparation must reject incompatible combinations,
write the selected mode into `manifest.runtime`, campaign contract, rollout
command, and tool-catalog identity, and include it in the manifest digest so it
cannot change mid-campaign.

## 2. Candidate representation and policy

The privileged path uses `zetta.evolution.models.CandidateBundle` and the
existing `zetta/evolution/candidate_artifacts.py` `STRUCTURED` adapter. A bundle
contains auditable critic rules/recovery rules (or one tool plugin), diagnosis
and parent digests, causal hypothesis, mechanism change, and validation plan.
Do not fork the ARX package format or silently convert a bundle to an ARX ZIP.

Add a named privileged evidence policy beside `ArxRgbPublicEvidencePolicy`.
It should allow only the versioned privileged observation projection and bounded
CandidateBundle evidence, while still rejecting seeds, RNG, model paths,
filesystem paths, raw simulator handles, and unbounded simulator arrays.
`get_evidence_policy`, lifecycle allowlists, visual artifact handling, and
publication code must all recognize the new policy explicitly.

## 3. High-level orchestration flow

Use the existing `scripts/evolution/run_campaign.py` / `EvolutionSupervisor`
state machine and gate semantics, changing only adapter configuration and
candidate loading:

1. Preparation freezes privileged schema/catalog/policy digests and sets
   `runtime.candidate_kind = structured_bundle_v1`.
2. Baseline rollouts run without a candidate but expose privileged observations
   to the deployment agent according to the frozen rollout mode.
3. The learner reads authorized privileged evidence, writes analysis, and
   submits an immutable `CandidateBundle`; store registration validates the
   structured digest and parent/diagnosis links.
4. Candidate rollouts load the bundle through `candidate_artifacts.load_artifact`
   and pass only its normalized critic/recovery rules to the trusted rollout
   runner. Critic and Recovery remain CandidateBundle consumers; neither role
   receives learner-private campaign history.
5. Replay, same-seed, regression, and held-out gates use the existing
   `CandidateGateRunner`/`EvolutionSupervisor` contracts. Privileged state is
   available for the configured online feature projection, but gate outcomes
   remain authoritative evaluator results and must not be inferred from a
   learner-supplied feature.

## 4. Rollout and identity changes

The prepared rollout command must carry policy/visibility and candidate-kind
placeholders. `run_arx_evolution_rollout.py` validates them against the frozen
manifest and trial registry, selects structured bundle loading for privileged
campaigns, and records the policy in episode identities and evidence indexes.
The current ARX-package manifest checks remain unchanged for RGB campaigns.

Define explicit baseline semantics for the structured path: `bundle_sha256 =
none` is valid only for strict pure-VLA baseline; candidate mode requires a
`CandidateBundle` digest. Reject an ARX package digest in privileged mode and a
structured bundle in RGB mode unless a future compatibility adapter is frozen.

## 5. Feature ABI and safety

Candidate-declared feature names must resolve only to the privileged projection
schema, with declared scalar type, validity behavior, sampling policy, and
history requirements. The sandbox receives JSON observations and images, not
MuJoCo objects or gateway clients. Enforce finite values, bounded history,
deterministic ordering, and per-feature timeout/resource limits. Recovery rules
must reference critic rule IDs and remain bounded by the existing gateway tool
catalog and episode budgets.

### Exact rollout command

Candidate jobs invoke:

```text
python scripts/deployment/run_arx_evolution_rollout.py --campaign-root ROOT \
  --logical-id ID --attempt-index N --generation G --task pickup_test_tube \
  --seed SEED --policy-rng RNG --bundle bundle.json --bundle-sha256 SHA \
  --baseline-mode strict_pure_vla --scene-registry scene-registry.json \
  --vla-host HOST --vla-port 5581 --gateway-port PORT \
  --runtime-limits runtime-limits.json --runner-limits runner-limits.json \
  --model-contract robots/arx/manifests/task7_model_a.yaml \
  --critic-runtime-limits critic-runtime-limits.json \
  --agent-settings agent-settings.json --calibration robot_calibration.json \
  --output-dir ATTEMPT --result-file ATTEMPT/episode_record.json \
  --heartbeat-file ATTEMPT/heartbeat.jsonl --evidence-policy arx_privileged_v1
```

Baseline jobs omit `--bundle`, use `--bundle-sha256 none`, and retain
`strict_pure_vla`. The command, candidate kind, visibility, policy, scene
registry, calibration, and all digests are frozen in `manifest.json`.

### Privileged readable feature ABI

The gateway emits `arx.privileged.observation.v1` with bounded fields:

```text
selected.manipulated_object.logical_id:string
selected.manipulated_object.position_m:float[3]
selected.target.logical_id:string
selected.target.position_m:float[3]
selected.target_gripper_distance_m:number (metres)
interaction.gripper_closed/gripper_contact/robot_contact/grasped/retained:bool
interaction.in_target/released_now/released_ever/success/failure:bool
interaction.lift_m:number; interaction.progress:[0,1]; interaction.stage:string
contact_summary.robot_count/gripper_count:integer
contact_summary.force_available:bool; contact_summary.max_normal_force_n:number
joint_summary.count:integer; joint_summary.normalized:map<string,number>
simulation.time_s:number (seconds)
```

CandidateBundle predicates may reference only flattened `privileged.*` names,
including `privileged.selected.target_gripper_distance_m`,
`privileged.interaction.gripper_closed`, `privileged.interaction.grasped`,
`privileged.interaction.progress`, `privileged.interaction.lift_m`, and the
bounded contact/joint summaries. Raw MuJoCo arrays, handles, paths, seeds,
RNG state, evaluator objects, and unbounded buffers are never readable.

### Frozen ARX tool catalog

```text
arx.zeva(max_chunks, reentry_token?)
arx.hold(steps)
arx.set_gripper(opening, max_steps)
arx.move_eef(delta_xyz_m, delta_rotvec_rad?, frame?, speed_m_s?)
arx.review_reentry(observation_ids)
arx.finish(reason)
```

`arx.move_eef` requires a frozen six-link calibration. Recovery bindings may
allow a subset. There is no `resume_vla` tool: successful
`arx.review_reentry` returns a token, and nominal VLA resumes through
`arx.zeva(reentry_token=...)`.

### Attempt artifacts

Completed attempts contain `result.json`, `episode_record.json`, `events/`,
`tools/`, `invocations/`, `catalog.json`, `reset.json`, `identities.json`,
`trajectory/{states,actions,chunks,tools}.jsonl`, and `frames/*.png`.
Privileged rows remain in the private journal/privileged JSONL; public
trajectory artifacts remain RGB-only. Video encoding is a separate optional
postprocessor, not part of the rollout ABI.

## 7. What the shared learner already supplies

The ARX adapter must use the existing campaign machinery rather than adding an
ARX learner loop. `run_campaign.py` constructs `EvolutionSupervisor`; the
supervisor/queue expands the frozen rollout command with `campaign_root`,
`logical_id`, `attempt_index`, `generation`, `task`, `seed`, `policy_rng`,
`candidate_path`, `bundle_sha256`, `baseline_mode`, `output_dir`, and
`result_file`. `zetta.evolution.lifecycle` then reads completed `episode_record`
artifacts, derives the scalar names co-observed on action rows, and passes those
names as `available_critic_features` to `CampaignStages.propose`. It also passes
the frozen tool catalog to cluster/diagnosis/proposal validation. Therefore ARX
must publish the same state rows and catalog shape; it must not implement a
second clustering, diagnosis, proposal, or queue.

The stage prompts are already environment-neutral: Cluster inspects contact
sheets and event windows; Diagnoser compares failed episodes, success
comparators, and compact telemetry; Proposal receives the diagnosis, catalog,
parent bundle, and observed feature catalog and must emit one atomic
critic/recovery pair. The ARX-specific prompt contract should preserve these
prompts verbatim and add only the ARX tool/feature ABI and visibility policy.

## 8. Detailed file changes

| File | Required implementation |
| --- | --- |
| `scripts/evolution/prepare_arx_campaign.py` | Refactor into the adapter equivalent of `prepare_robocasa_campaign`: accept `--observation-visibility {rgb_public,privileged}` (retain `--privileged` as a compatibility alias), require/validate `--calibration` for privileged mode, require exactly ten train scenes and twenty held-out entries, freeze `prompt-contract.json` containing the shared Cluster/Diagnosis/Proposal prompts plus ARX contracts, write `feature-catalog.json`, and include their digests in the manifest/contract. Emit command placeholders for `--candidate-path`/`--bundle`, `--calibration`, `--privileged`, `--observation-visibility`, and frozen policy. Set `runtime.bundle_files_by_sha` so supervisor promotion can resolve structured bundles. |
| `scripts/deployment/run_arx_evolution_rollout.py` | Validate every mode/visibility/policy/kind/calibration value against the frozen manifest; accept the shared `{candidate_path}` placeholder (with `--bundle` compatibility); load `CandidateBundle` via `candidate_artifacts.load_artifact` in `structured_bundle_v1` mode and reject ARX packages there; pass `privileged` and calibration into `Trial`; require `none` only for strict pure-VLA baseline; write state/feature rows in the standard trajectory locations consumed by lifecycle. |
| `robots/arx/deployment/contracts.py` | Add explicit `observation_visibility`, `evidence_policy`, and `candidate_kind` fields to `Trial`/environment identity, with strict validation and legacy defaults for non-campaign invocations. |
| `robots/arx/deployment/runner.py` | Route structured JSON bundles separately from ARX packages; pass visibility/policy/calibration to `serve_arx_gateway`; map the frozen provider credential reference to the worker environment; enforce fresh output directories and candidate identity checks. |
| `scripts/deployment/serve_arx_gateway.py` | Start the gateway with the frozen visibility/policy, expose only the catalog tools, adapt bundle rules to `TemporalCritic`, validate recovery bindings, and implement privileged reentry; never expose the private journal to the learner. |
| `robots/arx/environment.py` | Derive finite object/target pose, interaction, contact, joint, and simulation summaries. |
| `robots/arx/gateway/backend.py` / `session_core.py` | Carry, publish, flatten, journal, and record privileged observations without contaminating RGB public evidence. |
| `robots/arx/gateway/recording.py` | Export authorized privileged JSONL separately. |
| `robots/arx/critics/contracts.py` / `registry.py` | Enforce the privileged feature ABI, scalar/range/history bounds, and CandidateBundle assessment schema. |
| `zetta/evolution/evidence_policy.py` | Make `arx_privileged_v1` fail closed with an explicit allowlist/schema for the flattened feature catalog (not merely the `privileged.` prefix); validate bounded scalars, finite values, history/sampling metadata, and reject raw contact/pose arrays, paths, seeds, and handles. |
| `zetta/evolution/candidate_artifacts.py` / `store.py` | Resolve, register, and promote immutable structured bundles with parent/diagnosis linkage. |
| `zetta/evolution/campaign.py` / `gate_runner.py` / `supervisor.py` / `stages.py` | Read candidate kind/policy from runtime, resolve `bundle_files_by_sha`, pass ARX feature catalog to proposal validation, and preserve the existing direct/paired/replay/regression/held-out transitions. No ARX-specific clustering or diagnosis path. |
| `robots/arx/trajectory_recorder.py` | Preserve RGB-only trajectory export and optionally encode camera videos. |
| `tests/test_arx_privileged_campaign.py` | Cover preparation, digest/mode rejection, bundle registration, feature visibility, catalog, reentry, and one direct rollout. |
| `scripts/deployment/run_arx_privileged_test.sh` | Start Zeva, prepare artifacts, write bundle, run direct rollouts, and verify required records. |

## 9. Campaign invocation and completion criteria

For the target run, first start the remote Zeva/VLA servers and make their
addresses available to the worker environment. Then prepare the frozen
campaign (privileged mode and the calibration artifact) against
`runs/arx_pickup_test_tube_10_new`, start the normal queue workers, and invoke
`scripts/evolution/run_campaign.py` with the generated manifest, tool catalog,
campaign root, queue root, and worker hosts. The worker must receive the VLA
credential through its environment, never through the manifest. A preflight
must verify ten train scenes, twenty held-out trials, catalog/prompt/feature
digests, calibration digest, and that the rendered command contains no unresolved
placeholders except the supervisor-owned fields listed above.

The campaign is complete only when the direct test script succeeds for one
scene and then the ten-scene block. Each valid attempt must contain a completed
result, privileged observation rows for committed steps, the expected critic
proposal, recovery tool results, reentry assessment, and a trajectory index.
Task success remains evaluator-owned and is reported separately from
infrastructure validity or candidate semantic failure.

## 10. Tests and rollout gates

- Preparation tests cover both modes, defaults, incompatible policy/kind pairs,
  manifest immutability, and command placeholders.
- Artifact tests cover `CandidateBundle` registration, digest resolution,
  parent linkage, and rejection of ARX packages in privileged mode.
- Evidence-policy tests cover allowed privileged fields and fail-closed private
  names/paths/seeds.
- Rollout tests prove per-step privileged observations reach the deployment
  agent, while Critic/Recovery consume only the CandidateBundle projection.
- Supervisor/gate tests run baseline, candidate, same-seed, regression, and
  held-out transitions with structured candidates and preserve existing RGB
  behavior.
- Add an end-to-end smoke campaign with a tiny deterministic MuJoCo scene before
  enabling real ARX training budgets.
