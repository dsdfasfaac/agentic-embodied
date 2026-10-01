# Integrating Zetta supervision into the running Zeva–MuJoCo loop

Status: implementation plan, based on the current repository. Flags, modules,
and methods explicitly described as proposed below do not exist yet. This change
is documentation only.

## 1. Scope and recommended approach

Use the existing Zetta online supervision stack with an already available,
frozen critic/recovery bundle. Add ARX observation, VLA, and recovery-tool
bindings; preserve the Zetta decision schemas, authority boundaries, temporal
critic, persistence gate, and recovery sequencing.

Extend `scripts/deployment/run_arx_camera_chunk.py` with an opt-in supervised
mode. Keep the server command, scene, mapping, task, contract, rendering backend,
and default direct execution unchanged. Do not require a campaign, queue,
training process, diagnosis, gate, or promotion to run one supervised episode.
The types under `zetta.evolution` are useful online contracts even when evolution
is disabled. Importing those types does not start an evolution lifecycle.

This is the online subset of [the evolution plan](zeva-arx-evolution-plan.md).
For this task, prefer the incremental entrypoint below over implementing that
plan's campaign runner first. The current commands remain in
[the running guide](zeva-arx-running.md).

### What actually runs today

There are two distinct entrypoints:

| Path | Current behavior | Integration consequence |
| --- | --- | --- |
| Documented `run_arx_camera_chunk.py` | Direct `CosmosEdgeClient.predict()` followed by a Python loop over `ArxMujocoEnv.step()` | Already exposes each proposed action before execution. Add supervision here first; no Gateway API is needed for this path. |
| `ArxToolkit` → `ArxAgenticTaskAdapter.run()` | Planner observes, invokes `run_zeva_policy` once, then Runtime `policy_step` repeats until terminal | This is an outer planner wrapper, not an interruptible critic/recovery loop. A callback after `run()` or after a whole `policy_step` cannot reject an already executed action. |

Do not silently migrate the validated camera runner to Runtime as the first
change. Its environment is already the authoritative simulation owner. A later
Runtime binding can implement the same bounded execution interface; it must not
create a second simulator or reset during recovery.

## 2. The existing Zetta agentic framework

### 2.1 Roles and authority

| Component | Existing implementation | Responsibility |
| --- | --- | --- |
| Harness / episode driver | Online portion of `robots/libero/run_evolution_rollout.py` | Own scheduling, budgets, observations, evidence, pre-action checks, and control handoff. |
| Frozen Critic | `zetta/evolution/critic.py:TemporalCritic` | Evaluate machine-readable rules and emit proposals; never call tools or mutate the environment. |
| Online high-level agent, Role1 | `robots/robocasa/role1_agent.py` | Review evidence and proposals; produce a validated high-level decision using the existing planner backend. |
| Recovery controller | `robots/robocasa/recovery_controller.py:RecoveryController` | Retain the selected recovery and its current ordered step across calls. |
| Recovery Actor | `robots/libero/role1_recovery.py:LiberoRole1RecoveryActor` as binding reference | Turn a persisted Role1 effect into an allowlisted bounded tool invocation. Sole writer during recovery. |
| Tool catalog | `robots/robocasa/tool_catalog.py:ToolSpec, ToolCatalog` | Freeze public tool identities, capabilities, authority, and digest. |
| Simulator / evaluator | `robots/arx/environment.py`, `robots/arx/evaluators.py` | Apply controls and supply the authoritative task outcome. |

Role1 is event-driven in this frozen-recovery path. It is not invoked on every
15 Hz VLA action. The deterministic critic runs at each action boundary;
Role1 is called when a recovery step requires review. There is no need to add a
second free-running planner that competes with the recovery controller.

Normal motion is written by the harness through its environment binding.
During recovery, only the Recovery Actor may invoke motion bindings. The
Critic, Role1 model, and read-only model tools never receive a simulator handle.

### 2.2 Frozen supervision package: use `CandidateBundle`

Load the supplied JSON with `CandidateBundle.from_dict()` from
`zetta/evolution/models.py`, retain its canonical `sha256`, and freeze it for the
whole episode. Preserve these fields even without running evolution:

```text
CandidateBundle
  schema_version: 1
  candidate_id, generation, parent_sha256
  diagnosis_sha256
  causal_hypothesis, mechanism_change, validation_plan
  critic_rules: CriticRule[]
  recovery_rules: RecoveryRule[]
  tool_plugin: object | null

CriticRule
  rule_id, title, feature, operator, threshold
  dwell_steps, cooldown_steps, proposal, evidence_ids
  safety_only: bool
  activation_conditions: [{feature, operator, threshold}, ...]

RecoveryRule
  recovery_id, title, trigger_rule_ids, precondition
  steps: [{tool, parameters, stop_when}, ...]
  safety_constraints, stop_condition, fallback, evidence_ids
```

The constructor requires unique rule IDs, known triggers, evidence, and recovery
coverage for every critic rule. An empty bundle is invalid: direct mode uses
`bundle=None`, not a fabricated empty candidate. Existing provenance fields must
remain genuine supplied metadata; do not synthesize a diagnosis to deploy a
bundle. Reject unsupported tool-plugin-only bundles in this first integration;
loading arbitrary new executable plugins is outside the interface work.

The **format** is reusable across robots. A LIBERO or RoboCasa **bundle** is not
necessarily portable: its feature names, tool names, units, and parameters must
have real ARX equivalents. Preflight must reject unsupported semantics, not
rename Cartesian delta actions into ARX absolute joint targets. This plan assumes
an available bundle compatible with the advertised ARX feature/tool contract.

### 2.3 Critic input, output, and temporal behavior

`TemporalCritic.evaluate(features, step_index=...)` accepts a dictionary of
scalar features, with either flattened dotted keys or nested dictionaries.
Rule operators are `lt`, `le`, `gt`, `ge`, `eq`, `ne`, and `stagnant`;
activation predicates support the first six. All activation predicates must
hold. A false guard clears that rule's history, dwell count, and cooldown.
Ordinary predicates fire after `dwell_steps` consecutive matches; `stagnant`
uses the range of the last `dwell_steps` values. Cooldown advances on evaluation
calls, not elapsed wall-clock time.

Each fired output contains:

```text
rule_id, step_index, feature, observed_value,
activation_conditions (including each observed_value),
proposal, safety_only, environment_write=false
```

Reuse the evaluator unchanged. Keep its state across chunks and recovery;
reset only at episode reset. Do not evaluate it again merely because Role1
retries a decision. Cache the proposal for that blocked action. Missing features
and non-finite numeric inputs are errors, not implicit acceptance.
`safety_only` is metadata in this evaluator, not an implementation of hard
motion limits; the ARX safety processor remains independent.

The Actor converts fired dictionaries into existing `CriticProposal` objects:
`proposal_id`, `reject_current_action`, `reason`, `evidence`, and `details`.
Their serialized authority is `reject_current_action_only`, with
`proposal_only=true` and `environment_write=false`. The Critic must not insert a
replacement action, selected tool, or termination command into its output.

### 2.4 Role1 request and response format

Reuse `Role1Event` without an ARX-specific schema:

| Field | ARX binding |
| --- | --- |
| `event_id`, `task`, `step_index` | Opaque event identity, immutable full task instruction, executed action count. |
| `current_stage`, `allowed_stages` | Existing recovery pattern: `recover`; `recover`, `verify`, `closed_loop_execution`. |
| `current_tool`, `allowed_tools` | Current frozen recovery tool plus the explicitly interrupted VLA alternative, using `arx.*` public names. |
| `image_references` | Hash references for `front_rgb`, `left_rgb`, `right_rgb`; actual PNG payloads passed separately to the adapter. |
| `task_state` | Terminal flags, controller context, bounded approved evidence, budgets. |
| `tool_proposals` | Frozen recovery alternative and interrupted VLA proposal, both inert. |
| `critic_proposals` | Rejection proposals converted as above. |
| `history` | Bounded prior decision/tool summaries, without experiment metadata. |

`ToolProposal` fields are `proposal_id`, `tool`, `proposal`, and `evidence`;
serialization adds `source=tool`, `proposal_only=true`, and
`environment_write=false`. Follow the LIBERO Actor's pattern of explicitly
marking the VLA proposal `rejected_by_critic` and the recovery alternative
`ready_for_role1_approval`.

Role1's raw JSON decision uses:

```text
event_id, decision_id (optional/generated by existing validation)
proposal_disposition: accept | reject | modify
action_kind: continue | switch | recover | restage | regenerate | replace | terminate
selected_stage, selected_tool, direct_action
termination: {approved: bool, reason: string}
evidence: string[], confidence: number, rationale: string
proposal_ids: string[], modifications: object
```

These are the general Zetta action kinds, not permission to execute arbitrary
ARX actions. The frozen-recovery Actor requires selection of the current frozen
tool, rejects direct actions and approved termination, and only applies
`modifications.parameters` within the tool's supported bounds. Preserve the
LIBERO behavior that Role1 may modify parameters; do not silently disable it or
allow it to invent tools or reorder steps. Validate effective parameters before
any write. Audit both base and effective parameter hashes.

Use `Role1ModelAdapter` with the existing `build_planner` backend (`api` or
`codex`), a `Role1DecisionStore` configured with the ARX catalog, and the supplied
model/settings. Its sequence is model invocation → validation → immutable
persistence → activation → `Role1Effect`. An unpersisted decision is inert.
Do not parse free text, repair invalid JSON, or dispatch a tool before this gate.
Persisted serialization adds authority metadata; do not feed those additional
fields back as raw model decision fields.

Role1 events are seed-blind. Do not reuse `robots/arx/prompt_bundle.py`'s current
user prompt verbatim: it contains a seed. Keep seeds, reset metadata, output
paths containing seed identities, and checkpoint/provenance records in the
harness audit, outside Role1 events, evidence text, and history. Send only
approved non-geometric critic summaries; absolute simulator object/EEF poses
must not leak through nested proposal or tool-result fields.

### 2.5 Recovery lifetime and return to VLA

Reuse `RecoveryController` unchanged:

1. `activate()` finds matching trigger rules and deterministically selects the
   first recovery by sorted `recovery_id`. It never replaces an active recovery.
2. `context()` exposes the execution ID, bundle hash, current step, remaining
   steps, precondition, constraints, stop condition, and fallback.
3. Role1 reviews that step; the Actor acknowledges the pending rejection and
   executes the selected bounded tool.
4. If another critic rule interrupts the tool, retain the current controller
   step and return to Role1. A partial tool is not a completed recovery step.
5. Otherwise call `complete_current_step()` with the actual tool, executed
   horizon, and any verified no-op. Completion advances exactly one step.
6. Resume VLA only after controller completion and a nonterminal environment.
   Discard the old chunk remainder and infer from the post-recovery observation.

The controller records append-only events; it does not restore physics after
crashes. Its `precondition`, `stop_when`, `stop_condition`, and `fallback` strings
are context, not an executable condition language. Preserve that format.
Supported bindings must implement the stated bounded semantics; reject a
package whose conditions have no supported interpretation. Do not `eval()`
these strings or claim the controller automatically enforces them.

Preserve the existing triggering-rule suppression contract: keep the complete
critic and evaluate all rules during recovery, but filter active trigger IDs
from interrupt dispatch so they do not recursively block their own recovery.
Other fired rules still interrupt. Audit evaluated and filtered proposals;
restore suppression when leaving the recovery call. Hard safety checks are
never suppressed. Place this filtering before the local execution decision;
copying only LIBERO's post-RPC filter would be insufficient for a local
pre-action rejection gate.

Retain the LIBERO runner's bounded same-step Role1 contract retry behavior
(up to three attempts for its classified pre-write contract faults), Actor-call
limit, and distinction between method failure and infrastructure error. Do not
retry motion after an uncertain partial write. Neither errors nor recovery
budget exhaustion permit fallback to unsupervised VLA execution.

## 3. Adapt only the VLA and simulation boundary

### 3.1 Proposed local binding

Add a small `ArxEpisodeIO` in `robots/arx/episode_io.py` around the existing
client and environment. These are proposed internal methods, not planner tools:

```python
observe()                       # cached latest synchronized observation
infer_chunk()                   # no physics; preserve full Zeva request
critic_snapshot(raw_action)     # read-only features and approved sidecar
step_action(raw_action, owner)  # exactly one existing env.step; audit owner
```

The harness holds ownership in normal execution and grants it to the Recovery
Actor for recovery. Tool code must not access `env.data`, `qpos`, `ctrl`, or
`mj_step` directly. Observation and inference do not advance simulation time.
Expose episode step/terminal state explicitly for tools instead of duplicating
private counters in multiple adapters.

Preserve the current Zeva contract: three camera views, prepared 14D state
`[left 7D, right 7D]`, task-specified zeroing of the left model state, immutable
instruction, 32 absolute 14D output targets, and execution of the first 16 as
specified by the task. Keep `CosmosEdgeClient`, `prepare_model_state`, mapping,
locks, filters, limits, gripper conversion, and checkpoint unchanged. Do not
substitute LIBERO's delta Cartesian action or image/state conventions.

### 3.2 Exact interception point and timing

The current call chain is:

```text
raw action → env.step → ActionProcessor.process → 4 interpolated 60 Hz commands
           → MuJoCo substeps → evaluator → fresh three-view observation
```

Place the rejection gate **before** `env.step`. `ActionProcessor.process()`
updates `previous`, so even calling it merely to inspect an action changes
future execution. Initially use raw proposal features and the previous completed
step's diagnostics. If a supplied rule requires the proposed processed target,
add a pure preview based on copied processor state and test preview/commit
parity; do not process twice on the live processor.

Here one supervised action means one `env.step` / 15 Hz policy action, lasting
1/15 second under the current configuration. It does not mean each internal
60 Hz interpolation command or each `mj_step` substep. Checking within those
substeps would change the established control/evaluation boundary and is a
separate change. Check official termination after every VLA and recovery
`env.step`; never finish the remaining prefix after terminal.

Rejection must leave simulator time, qpos, qvel, controls, step counter,
`_last_command`, and processor history unchanged. Critic history and audit logs
may change. Inference failure must also leave physics unchanged.

### 3.3 ARX critic features

Implement `robots/arx/critic_runtime.py` as a feature adapter, importing the
existing `TemporalCritic` rather than copying its algorithm. Freeze a feature
schema with units, observation timing, and availability. Start with the features
actually required by the supplied bundle:

- Proposed absolute target minus prior command, per-arm/gripper norms and
  direction changes; label these as joint-space quantities, not Cartesian motion.
- Previous commanded-versus-measured joint changes and filter diagnostics.
- Observation validity and repeated-action history, when required.
- Critic-only `privileged.*` task progress, lift/contact/grasp summaries obtained
  through a read-only environment snapshot.

The evaluator's progress is simulator-derived, not automatically deployable
vision evidence. Current `info` exposes `progress`, `evaluation`,
`processed_action`, and `filter`; success is
`info['evaluation']['success']`, not a current top-level `is_success` field.
Normalize it in the binding without changing the evaluator. Do not call a
stateful evaluator again just to obtain critic features: retain its last verdict
and read any additional measurements without advancing its trackers.

Document pre-action versus previous-post-action values. Reject unavailable
required features during preflight/reset rather than silently filling zeros.
Keep private snapshots out of Zeva input and default Role1 context. Any approved
scalar evidence uses an explicit allowlist, not just a blacklist of pose keys.

### 3.4 Recovery tool binding

Build `robots/arx/tool_catalog.py` using existing `ToolSpec` and `ToolCatalog`.
One necessary shared interface patch is to allow the `arx.` namespace in
`ToolSpec.__post_init__`; currently it accepts only `robocasa.` and `libero.`.
Keep existing catalog serialization and digests unchanged for existing entries.
Declare motion tool contracts `proposal_only=True` and inject this catalog into
`Role1DecisionStore` instead of relying on its RoboCasa default.

Follow LIBERO's explicit short-name ↔ public-name mapping. For example,
`vla_execute` in an ARX bundle maps to `arx.vla_execute`; its implementation
uses Zeva. Never map it through the LIBERO whitelist. Bind exactly the tools
required by the available package. A possible initial supported set is:

| Short name (proposed) | Interface behavior |
| --- | --- |
| `vla_execute` | Fresh bounded Zeva prefix with the same per-action critic gate; usable as a frozen recovery step. |
| `reobserve` | Current three-view observation; verified read-only no-op if no physics advances. |
| `set_right_gripper` | Bounded hardware-semantic gripper target; preserve other channels and task locks. |
| `retract_right_arm` | Fixed, reviewed relative joint-space retreat profile; no model-generated vectors or target coordinates. |
| `wait` | Bounded hold/settling through the same action and safety path. |

Motion profiles and parameter limits are ARX interface assets, not something
Role1 invents online. Only register implemented and scene-validated profiles.
`ToolSpec` does not itself define JSON parameter schemas: keep a frozen schema
and validator beside each binding and record their identity in the run audit.
Reject unknown arguments and validate Role1 modifications before dispatch.

All multi-action tools yield on another critic rejection or official terminal.
On interruption, return executed horizon and incomplete status; retain the
controller step. Re-entry must use current state and a bounded remaining budget,
not blindly replay a full relative retreat from scratch. A no-op counts only
when the binding verifies its postcondition and reports `no_op_verified=true`
and `environment_advanced=false`. Zero executed actions alone do not mean success.

Discarding a stale chunk is a harness bookkeeping operation, not another model
choice. A fresh inference must not mark recovery complete unless that is the
frozen step's actual supported postcondition. Local budget stop is a run stop
reason; it must not fabricate simulator termination or success.

## 4. Concrete incremental code updates

| Order | File | Proposed change |
| --- | --- | --- |
| 1 | `scripts/deployment/run_arx_camera_chunk.py` | Extract inference/step/record helpers; preserve existing default loop and artifacts. Introduce optional supervision arguments with no default behavior change. |
| 1 | `robots/arx/episode_io.py` (new) | Wrap existing direct client/environment, bounded stepping, ownership, observations, and verdict normalization. |
| 2 | `robots/arx/critic_runtime.py` (new) | Feature snapshot and persistent `TemporalCritic` integration; pre-action rejection and audit. |
| 2 | `robots/arx/environment.py` | Only if needed, add read-only critic snapshot/accessors; preserve reset, dynamics, filtering, and evaluator. |
| 3 | `robots/robocasa/tool_catalog.py` | Add `arx.` as an accepted namespace; no Role1 schema or policy changes. |
| 3 | `robots/arx/tool_catalog.py`, `robots/arx/tool_bindings.py` (new) | ARX public catalog, names, bounded parameter validation, and physical bindings. |
| 4 | `robots/arx/role1_recovery.py` (new) | Thin binding modeled on `LiberoRole1RecoveryActor`: replace image keys, tool mapping, environment access, and evidence adapter. Import shared Role1/store classes. |
| 4 | `robots/arx/supervised_rollout.py` (new) | Compose critic, unchanged `RecoveryController`, Role1 adapter, Actor, and IO. Adapt only the online `handle_recovery` structure from the LIBERO runner, including suppression, retries, partial steps, and limits. |
| 5 | Existing camera runner | Dispatch into supervised loop only when a bundle is supplied; retain current video/NumPy/diagnostic outputs. |
| 6 | ARX tests and running guide | Add parity, rejection, recovery, and live validation examples. |

Prefer direct imports of shared classes. Do not copy `role1_agent.py`,
`recovery_controller.py`, or `TemporalCritic` into ARX. The LIBERO Actor contains
robot-specific image/tool/state handling and cannot be reused by merely changing
its class name. Initially adapt that thin binding; if eliminating duplicated
orchestration requires extracting an environment-neutral helper, preserve the
existing LIBERO public wrapper and prove identical behavior with its tests.
Do not copy campaign setup and reporting code into the new episode driver.

### Proposed loop (control-flow sketch)

```text
load and validate frozen bundle, catalog, feature contract and budgets
reset IO and critic once; record initial observation
while not simulator terminal and within run budgets:
    if recovery.active:
        review current frozen step using unchanged Role1 pipeline
        Actor executes bounded binding under trigger-proposal suppression
        if newly interrupted: retain step; review new evidence
        elif terminal: stop
        else: complete_current_step(actual execution or verified no-op)
        continue
    if no pending chunk:
        infer one fresh chunk (counts toward --chunks)
    get next raw action within task.execution_steps
    evaluate persistent critic on read-only pre-action features
    if fired:
        record rejection; invalidate unexecuted chunk suffix
        activate matching recovery; do not call env.step
    else:
        execute one action and record official result
```

Budgets apply inside tools too. Keep `--chunks` as a cap on Zeva inference calls,
including fresh calls made by recovery tools; report recovery calls separately.
An already activated non-VLA recovery may finish within explicit recovery and
episode budgets after the last permitted VLA call. It cannot request another
chunk beyond the cap. `task.max_steps` covers all physical actions, including
recovery; retain a separate Actor-call cap and bounded tool horizons. Thus four
chunks still allow at most 64 VLA actions, but total physical actions can include
additional recovery actions. Audit both counts and do not claim direct-run and
supervised-run motion budgets are automatically identical.

### Incremental command surface

The server command remains:

```bash
./scripts/deployment/start_zeva_arx_task7_server.sh
```

After implementation, add the following proposed flags to the existing client
command. The bundle path is an example supplied artifact, not a file created by
this documentation:

```bash
MUJOCO_GL=osmesa \
XDG_CACHE_HOME=/tmp/zetta-arx-cache \
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  scripts/deployment/run_arx_camera_chunk.py \
  --scene runs/arx_pickup_test_tube/dark_silver_estimated_top_scene \
  --mapping /data4/zhengyikai/Zeva_arx/assets/ac_one/ac_one_14d_mapping.json \
  --task robots/arx/manifests/pickup_test_tube.yaml \
  --contract robots/arx/manifests/task7_model_a.yaml \
  --host 127.0.0.1 --port 5581 --seed 17 --chunks 4 \
  --supervision-bundle /path/to/frozen-arx-bundle.json \
  --role1-planner api \
  --role1-model "$ROLE1_MODEL" \
  --max-recovery-actor-calls 16 \
  --output runs/arx_pickup_test_tube/agentic_seed17_run1
```

Omitting `--supervision-bundle` preserves direct mode. Reuse the installed
system's Role1 model and provider configuration; credentials stay in environment
variables. Fail early if the selected Role1 backend is unavailable. Do not make
the direct runner require a model provider or load planner dependencies eagerly.
Use `--offline-test` with an injected deterministic Role1 test fixture for
integration tests; offline Zeva alone does not make Role1 inference offline.

### Optional follow-on: the existing Runtime-backed agent command

After local supervised parity, bind the same supervisor to the Runtime path.
Keep `ArxToolkit`'s observe-before-run and evaluator-gated finish behavior. The
supervised implementation can remain internal to the one episode tool; do not
expose arbitrary joint actions to its outer planner. Update the supervised
prompt's description of execution while retaining the direct prompt for direct
mode.

First inspect reusable inference-only and action-step operations. If they cannot
separate proposal from execution, introduce the smallest bounded interface and
wire local/remote DTOs, workers, and tests together. Do not call `policy_step`
and try to veto its actions afterward. Keep existing `policy_step` callers
unchanged. Runtime has a layering rule against importing `zetta`/`robots` into
`rollout_runtime`; keep the Zetta supervisor on the application side and transport
only normalized observation/action/critic-sidecar data. Do not introduce a new
copy of the critic merely to support the documented local runner.

## 5. Evidence and outcomes

Preserve current `three_view.mp4`, camera videos/frame-zero PNGs,
`raw_actions.npy`, `executed_actions.npy`, `audit.json`,
`diagnostic_trace.jsonl`, and `native_fault.log`. Video and executed actions must
include recovery motion. Raw Zeva actions remain model outputs only; keep
proposed recovery actions in a separate action/event stream and label action
source so NumPy arrays need not have a one-to-one row mapping.

Add append-only action/critic/tool events and the existing Role1 invocation,
decision, and recovery-event artifacts under `role1/`. Record chunk/action IDs,
raw proposal, accepted/rejected result, processed action, observation references,
controller execution/step IDs, suppression, tool parameters and hashes, decision
IDs, actual executed horizon, and the official verdict. Record package/catalog,
scene/mapping/task/contract identities and Role1 settings in the private run audit.

Distinguish environment success/failure, environment time limit, local budget
stop, method failure, and infrastructure error. Retain the last real terminal
flags and verdict. A camera/transport/persistence error is not evidence of task
failure; an exhausted `--chunks` run is not necessarily environment-truncated.
Publish a final summary only after required artifacts are complete; preserve
partial traces on exceptions. Existing `TrajectoryArtifacts` indexing can be
added later if useful, without requiring a campaign-shaped `EpisodeRecord` for
this standalone integration.

## 6. Validation and completion criteria

1. **Direct parity:** fake Zeva and fixed reset produce identical requests,
   raw/processed actions, state evolution, step counts, and outcomes before and
   after extraction. Existing direct commands and ARX contract tests pass.
2. **No-write rejection:** snapshot physics, counters, commands, and processor
   history; force a rejection on action N; all remain unchanged and the remaining
   chunk is never executed. Inference/model/persistence failure does not step.
3. **Temporal parity:** feed recorded features to the shared critic; proposals
   match live execution across chunk boundaries, guards, dwell, cooldown,
   suppression, and Role1 retries.
4. **Authority and format:** reuse Role1 contract/store tests; test ARX namespace,
   catalog injection, three-view payloads, seed/private-geometry exclusion,
   persistence-before-write, allowlists, and parameter modification bounds.
5. **Recovery lifetime:** two-step recovery survives multiple Actor calls; a
   different rule interrupts mid-tool without advancing the controller step;
   trigger suppression cannot disable hard safety or unrelated rules. Verified
   no-ops complete; unverified zero-action results do not.
6. **Return to policy:** fresh inference contains the post-recovery state and all
   three fresh views. No reset, task rewrite, cached suffix, or processor reset.
7. **Terminal and limits:** official success/failure/time limit stops every tool
   and prefix; Role1 cannot manufacture success. Inference caps, total physical
   steps, tool horizons, Actor calls, and bounded contract retries all terminate.
8. **Failure handling:** malformed/non-finite Zeva responses, missing features,
   Role1 invalid JSON/timeouts, audit failures, and uncertain partial writes have
   explicit non-success outcomes and no uncriticized continuation.
9. **Live ladder:** original OSMesa one-chunk smoke; supervised run with no fired
   rules; controlled compatible test trigger with recorded recovery; then a
   four-chunk live run. Keep test-trigger bundles distinct from the supplied
   production bundle. Integration success does not imply recovery effectiveness
   or task success; report the evaluator's actual result.

The integration is complete when the existing command can opt into the frozen
Zetta supervision stack, a rejected action provably never reaches physics, the
persisted Role1 decision controls bounded recovery, and execution resumes with a
fresh Zeva observation—all without changing Zetta's online decision format or
adding evolution machinery.
