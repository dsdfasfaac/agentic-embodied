# ARX learner grounding and orchestration implementation plan

This plan aligns the ARX learner with the existing RoboCasa Role1 and Zetta
campaign-stage controls while preserving ARX's RGB-only, non-privileged boundary.

## Required behavior

The learner receives only a fixed phase contract, a bounded public payload, and
typed learner tools. It never receives repository shell/filesystem tools,
gateway administration, simulator objects, seeds, private paths, contacts,
poses, reward, evaluator state, or deployment-agent conversations.

The lifecycle remains:

```text
prepare -> baseline rollouts -> public evidence -> analysis -> package authoring
-> package preflight/replay -> candidate rollouts -> gates -> checkpoint
```

## Grounding contract and prompts

Add `zetta/evolution/arx/prompts.py` with immutable contracts for `COLLECT`,
`ANALYZE`, `AUTHOR`, `PREFLIGHT`, `EVALUATE`, and `CHECKPOINT`. Each contract
must specify the phase input schema, allowed tools, required result fields,
evidence citation rules, forbidden information, uncertainty handling, stopping
rules, and valid transitions. The contract must state that online features and
recovery knowledge are RGB/public-evidence only. Do not reuse RoboCasa action
semantics; reuse its fixed-contract and strict-output pattern.

## Typed learner tools

Replace the generic `additionalProperties` registration in `session.py` with
`zetta/evolution/arx/tool_adapter.py`. Register exact schemas derived from the
Pydantic models in `contracts.py`, parse dictionaries before invoking handlers,
reject unknown keys, serialize typed results, and record validation failures.

Required tools are `read_contract`, `request_rollouts`, `job_status`,
`read_evidence`, `save_analysis`, `write_candidate_file`, `submit_candidate`,
`evaluate_candidate`, `request_owner_change`, and `checkpoint_learning`.
Zero-argument tools must reject unexpected arguments. `request_rollouts` must
construct `RolloutRequest`; analysis/checkpoint tools must construct their
models. No aliases or implicit field translation are allowed.

## Public evidence boundary

Add `zetta/evolution/arx/evidence.py`, following the bounded artifact reader in
`zetta/evolution/stages.py`. Resolve only content IDs, expose only public or
explicitly offline-authorized records, enforce byte/read/frame budgets, verify
image hashes, and record access IDs. Never expose host paths, seeds, private
trial JSON, capability tokens, evaluator-private fields, or gateway journals.
Ingest only the public rollout subset into a content-addressed index.

## Planner isolation

Add an ARX planner adapter modeled on
`robots/robocasa/role1_agent.py:Role1ModelAdapter`. Construct a fresh Codex
planner for every phase, use no resume/thread ID, persist exact input/system
contract/payload/messages/stats/tool calls/timing/errors, and enforce wall-time,
turn, token, tool-call, evidence-read, and output limits. The planner toolkit
must contain only the current phase's learner tools; remove common filesystem
and shell tools. If the selected Codex backend exposes those capabilities,
launch it in a capsule containing only the public learner API and draft area.

## Structured state machine

Add strict `LearnerTurnInput`, `LearnerTurnResult`, `StageResult`,
`AttemptFeedback`, `EvidenceReadRecord`, `PackageDraftState`, and
`CampaignPhaseState` models. Results must contain phase/status, evidence IDs,
hypotheses, package digest, job groups, next action, and remaining budgets.
Enforce transitions:

```text
PREPARE -> COLLECT_BASELINE -> ANALYZE -> AUTHOR -> PREFLIGHT
         -> DEVELOPMENT -> PAIRED_GATE -> REGRESSION -> HELDOUT -> COMPLETE
```

Require terminal jobs before analysis, analysis before authoring, a sealed
package before candidate rollout, and successful replay/contract checks before
gates. Persist after every tool call and phase result; resume only from durable
boundaries without duplicating jobs.

## Coordinator and package lifecycle

The coordinator must materialize one absolute-path trial per scene, reserve
ports/output directories, launch `run_arx_evolution_rollout`, ingest public
artifacts, classify outcomes, and retry only safe pre-write failures. Baseline
trials have no candidate or agent. Candidate trials require an active immutable
package whose digest matches the request and whose contract/catalog/bootstrap
identities match the campaign.

Package activation requires manifest/hash validation, ABI and isolation checks,
bounded bindings for every failure mode, replay on firing/non-firing examples,
and mock-gateway checks. Active packages must be discovered from durable store
records, including across process restarts and later campaigns.

## Interactive/unattended parity

Both CLI modes must use the same session, planner adapter, tool adapter, state
machine, coordinator, evidence store, and package activation logic. Interactive
mode may inspect status/evidence and pause at durable boundaries, but cannot
edit trials, mutate sealed packages, run shell commands, or bypass gates.

## Acceptance tests

Add tests for exact schemas, typed dictionary conversion, zero-argument rejection,
private-field filtering, evidence budgets/hash checks, fresh planner/no resume,
phase preconditions, baseline and candidate package binding, restart persistence,
safe retry classification, and interactive/unattended controller parity.

The decisive fake-provider/fake-gateway test must complete:

```text
prepare -> baseline -> evidence ingest -> analysis -> seal package
-> replay/contract gate -> candidate rollout -> paired result -> checkpoint
```

Only after that test passes should live Zeva/Codex execution be enabled.

## File-level implementation map

The following changes are required. A file not listed here must not be changed
as part of this work except for import/export wiring or test fixtures.

### `zetta/evolution/arx/contracts.py`

Keep strict Pydantic models with `extra="forbid"`. Add:

- `LearnerToolError` and `AttemptFeedback` with phase, tool, error code,
  retry class, public message, and input digest;
- `LearnerTurnInput` with contract ID, phase, checkpoint ID, allowed tool names,
  bounded evidence IDs, active package digest, job-group IDs, budgets, and
  required output schema;
- `LearnerTurnResult` with schema version, phase, status, evidence IDs,
  hypothesis IDs, job-group IDs, package digest, next action, budgets, and
  owner-change payload;
- `EvidenceReadRequest` and `EvidenceReadRecord`;
- `JobStatusRequest`, `CandidateFileRequest`, `CandidateSubmissionRequest`,
  `CandidateEvaluationRequest`, and `OwnerChangeRequest`;
- `CampaignPhaseState` with the legal phase enum, completed job groups, analysis
  IDs, active package, gate IDs, budgets, and last checkpoint;
- `PackageDraftState` with immutable draft ID, allowlisted files, file digests,
  and sealed package digest.

Add validators for phase transitions, evidence-ID syntax, package requirements,
positive budgets, and candidate-only fields. Preserve v1 reading but reject v1
for live execution when v2 fields are absent.

### `zetta/evolution/arx/prompts.py` (new)

Define the immutable base and phase prompts listed above. Export canonical prompt
texts and SHA-256 digests. The campaign contract stores those digests. Prompts
must explicitly enumerate the current phase's tools, input/output schemas,
public-information boundary, forbidden fields, and terminal behavior.

### `zetta/evolution/arx/tool_adapter.py` (new)

Implement `ArxLearnerToolkitFactory` and `register_typed_tool`. It must create
an empty `Toolkit`, register only the supplied tools, derive JSON schemas from
strict input models, convert dictionaries to models, call handlers, serialize
Pydantic results, and emit typed validation feedback. It must never register
the common filesystem/shell tools. Include adapters for zero-argument tools and
keyword-argument tools.

### `zetta/evolution/arx/evidence.py` (new)

Implement `ArxEvidenceIndex` and `ArxEvidenceReader`:

- ingest only allowlisted public rollout files;
- assign content-addressed IDs;
- strip paths, seeds, episode IDs, credentials, capability data, evaluator
  private fields, and gateway journals;
- resolve bounded frame/window selectors;
- validate image encoding and SHA-256;
- enforce per-turn byte/read limits;
- produce access records and duplicate-read cache hits;
- expose JSON-safe learner results and image blocks.

### `zetta/evolution/arx/planner.py` (new)

Implement `ArxLearnerPlanner` using `build_planner`:

- create a fresh planner per phase call;
- use the campaign's planner type/model/reasoning settings;
- never pass a resume/thread ID;
- persist `input.json`, `system_contract.txt`, planner messages, tool calls,
  stats, timing, timeout/error, and structured result;
- enforce max turns/tokens/time/tool calls/output size;
- reject private paths and ungrounded evidence references;
- return a validated `LearnerTurnResult` or typed attempt feedback.

If the selected Codex backend exposes general tools, launch it through the
approved deployment capsule with only the campaign public API and draft root.

### `zetta/evolution/arx/stages.py` (new)

Implement `ArxLearningStages` as the durable phase controller. Each method must
validate preconditions, construct a `LearnerTurnInput`, call the planner, apply
the validated result, checkpoint atomically, and return the next phase. Add:

- `prepare()`;
- `collect_baseline()`;
- `analyze()`;
- `author()`;
- `preflight_package()`;
- `development()`;
- `paired_gate()`;
- `regression()`;
- `heldout()`;
- `complete()`.

No method may silently advance after a failed tool call, timeout, invalid result,
or incomplete job group.

### `zetta/evolution/arx/store.py`

Extend the durable store with atomic ledgers for contracts, phase state,
checkpoints, invocations, tool attempts, job groups, jobs, evidence index
records, analyses, drafts, sealed packages, active-package pointers, gate jobs,
and owner-change requests. Add idempotent request-ID lookup and restart-safe
recovery of `pending`/`running` jobs. Never overwrite an immutable contract,
package, result, or invocation.

### `zetta/evolution/arx/tools.py`

Convert all handlers to typed model inputs and JSON-safe outputs. Enforce:

- phase permissions;
- split and budget ownership;
- terminal-job requirement before evidence analysis;
- active-package digest requirement for candidate requests;
- immutable draft paths and file size limits;
- package activation only after preflight;
- no arbitrary content IDs, filesystem paths, seeds, or trial edits.

`request_rollouts` delegates to the coordinator and returns a typed job group.
`read_evidence` delegates to `ArxEvidenceReader`. `submit_candidate` records
the sealed package path/digest durably and activates it only after all checks.

### `zetta/evolution/arx/session.py`

Replace the current payload-only `turn()` and generic Toolkit construction with
`ArxLearningStages` plus `ArxLearnerPlanner`. `automated()` runs the durable
state machine. `interactive()` must call the same controller and support only
status, continue, pause, bounded evidence inspection, owner-change request, and
quit-at-boundary commands. It must not implement a second tool protocol.

### `zetta/evolution/arx/coordinator.py`

Replace the minimal subprocess wrapper with a durable coordinator providing:

- trial materialization from v2 registry entries;
- unique port/output leases;
- campaign-owned Zeva start/readiness/reuse/cleanup;
- baseline/candidate trial construction;
- subprocess launch via argument arrays;
- result polling and public evidence ingestion;
- safe pre-write retry and uncertain-write handling;
- stale lease/process recovery;
- per-campaign concurrency limits.

### `zetta/evolution/arx/cli.py` (new)

Provide shared command handling for automated and interactive modes. Enforce one
controller per campaign root, load/validate v2 contracts, resume durable state,
and close coordinator resources on all exits. `run_arx_learning.py` becomes a
thin compatibility wrapper around this module.

### `scripts/evolution/prepare_arx_campaign.py`

Generate the complete v2 contract: scene trial registry and explicit splits,
task/instruction digest, gateway/catalog/critic/bootstrap/prompt digests,
Zeva lifecycle settings, runtime and critic limits, planner/provider settings,
budgets, artifact visibility, and immutable campaign identity. Reject missing
model/runtime artifacts and nonempty output roots.

### `scripts/evolution/run_arx_learning.py`

Keep only argument parsing and delegation to `zetta.evolution.arx.cli`. It must
not construct planners, coordinators, or ad hoc tool adapters directly.

### `scripts/deployment/run_arx_evolution_rollout.py`

Preserve the one-trial entrypoint but validate v2-generated trial identities,
write a stable public-artifact manifest, distinguish configuration,
infrastructure, agent, uncertain-write, and completed statuses, and return the
documented exit code. Do not add learning logic here.

### `robots/arx/critics/packages.py`

Add package preflight/activation helpers that verify manifest hashes, campaign,
catalog, bootstrap, feature-schema, file allowlists, and immutable bytes. Return
typed reports consumed by `LearningTools.submit_candidate`.

### `robots/arx/critics/isolation.py` and `registry.py`

Expose deterministic preflight/replay entrypoints, enforce critic worker limits,
and report firing/non-firing replay results without privileged data leakage.

### Tests

Add:

- `tests/test_arx_learning_grounding.py` for exact schemas, typed conversion,
  tool allowlists, and private-field rejection;
- `tests/test_arx_learning_evidence.py` for ingestion, image hashes, budgets,
  selectors, and access records;
- `tests/test_arx_learning_state.py` for transitions, idempotency, restart, and
  package activation;
- `tests/test_arx_learning_planner.py` for fresh planners, no resume IDs,
  persisted artifacts, timeout classification, and bounded payloads;
- `tests/test_arx_learning_campaign.py` for fake-provider/fake-gateway full
  baseline-to-candidate flow;
- `tests/test_arx_learning_cli.py` for interactive/unattended parity and
  single-controller ownership.

The fake campaign test is a release gate and must complete baseline rollout,
public evidence ingestion, analysis, package seal/preflight, candidate rollout,
paired result, and durable checkpoint before live execution is considered ready.
