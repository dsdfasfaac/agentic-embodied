# ARX–MuJoCo tool gateway implementation plan

Status: proposed implementation contract. New module names, API methods, schemas,
and commands below must be implemented; they are not existing capabilities.

This document and [critic format](arx-observation-critic-plan.md) and
[learning orchestration](arx-learning-orchestration-plan.md) define the new
integration. Where they conflict with `zeva-arx-agentic-plan.md` or
`zeva-arx-evolution-plan.md`, these three plans govern the new path. Preserve
the existing [Zeva running pipeline](zeva-arx-running.md) as a baseline.

## 1. Architecture and scope

Build an ARX-specific gateway around `ArxMujocoEnv` and `CosmosEdgeClient`.
A transport-independent ARX session core owns one simulator instance per episode
and executes all targets through one serialized executor. The gateway service
validates and dispatches requests; the episode runner owns agent invocation. Zeva, gripper control, and EEF control are
peer tool implementations. The external Zeva service remains on its configured
TCP port; it returns targets and never owns simulation stepping.

```text
deployment agent -- authenticated tool API --> ArxGateway
                                                |
                                  single episode executor
                                   /          |          \
                             zeva tool   gripper tool   EEF tool
                                   \          |          /
                                    checked raw 14D targets
                                                |
                                      ArxMujocoEnv.step
                                                |
                              observation -> critic -> execution gate
```

Do not build on the HTTP/SSH bridge in `integrations/cosmos-arx-rgb-cr-2`.
That directory is a functional reference for visual recovery, bounded motion,
and policy reentry, not a transport, persistence, or orchestration architecture.
Do not import its worker/session classes into production gateway code.

Version one uses the direct environment binding from
`scripts/deployment/run_arx_camera_chunk.py`. It does not require Ray or the
general Rollout Runtime. `RuntimeGateway.policy_infer` and `action_step` already
exist and may support a later binding, but are not dependencies of this plan.
Keep the external ARX tool contract unchanged if that binding is added.

For explicit baseline mode, the harness installs a no-intervention monitor
instead of a learning package; the executor still records observations and
enforces tool bounds and terminal checks. This mode cannot be selected by the
deployment agent to bypass a critic.

## 2. Ownership and information boundaries

The episode executor alone may call reset, step, render, and close. Construct
and use MuJoCo/render resources on that executor's thread. Network handlers,
agents, tool planners, and critics never receive an environment object.

Use distinct objects, not a blacklist over one shared observation dictionary:

| Object | Contents | Consumers |
| --- | --- | --- |
| `PolicyObservation` | Three RGB arrays, measured 14D state | Zeva adapter only |
| `PublicObservation` | RGB references, observation ID, step/time, lifecycle, budgets | Deployment agent, critic adapter |
| `CommandState` | Last committed processed target and static robot calibration | Bounded motion planner |
| `PrivateEvaluation` | Environment reward/info, contacts, poses, outcome | Recorder/evaluator only |

Online critic and recovery must not use simulator object poses, contacts,
reward, evaluator progress, measured EEF pose, or hidden task predicates.
Version one also excludes measured joint state from recovery; Zeva retains its
existing proprioceptive input. Static kinematics and command-derived FK are
allowed, but must be labeled predictions, never measured arrival.

At reset, initialize the recovery command ledger from the explicit configured
robot start command, verified by a trusted reset invariant. Do not seed recovery
from arbitrary measured reset state. Unsupported reset configurations fail
preflight. Thereafter update the ledger only from successfully committed
`info['processed_action']`, not raw proposed targets. Keep evaluator fields out
of that interface. Add a narrow read-only committed-command accessor if needed;
tools must not reach into `_processor`, `_last_command`, or `env.data`.

The environment may terminate using its existing private evaluator. Publish
only `environment_ended` online; do not publish privileged termination reasons
or success feedback while control is possible. Authoritative outcome is an
offline feedback artifact after closure.

## 3. Modules and internal interfaces

Implement under `robots/arx/gateway/`:

| Module | Responsibility |
| --- | --- |
| `contracts.py` | Strict request/result/observation models and schema export |
| `service.py` | Authenticated HTTP handlers; no environment access |
| `session_core.py` | Physical execution owner, checked target loop, critic, terminal handling |
| `episode.py` | Episode driver, recovery context, nominal continuation, agent event delivery |
| `backend.py` | Direct `ArxMujocoEnv` and camera-runner bindings |
| `tools.py` | Frozen catalog, typed arguments, proposal-only target planners |
| `zeva.py` | `CosmosEdgeClient` tool adapter |
| `motion.py` | Static calibration, command FK/IK, bounded target planning |
| `journal.py` | Durable operation and physical-step records |
| `public.py` | Allowlisted serialization and image access |
| `client.py` | Typed agent client, polling, transport reconciliation |

Internal backend API:

```python
reset(private_config) -> InitialEpisodeState
policy_observation() -> PolicyObservation   # cached, no stepping
public_observation() -> PublicObservation   # cached, no stepping
committed_command() -> CommandState
step(raw_target_14d) -> StepCommit          # one env.step, executor only
close() -> PrivateEpisodeResult
```

`StepCommit` separates new policy/public observation, committed command,
terminal flags, and private evaluation. It must not become an agent tool result.
The backend creates the observations once; reobserving does not advance time.

### Session core and reference mapping

Model `ArxSessionCore` on RoboCasa's transport-independent `RoboCasaSession`,
not its privileged observation contents. Expose `reset`, `execute_targets`,
`snapshot`, `finalize_episode_artifacts`, and `close`. The physical target loop
lives only in `execute_targets`; it returns requested/executed horizon, final
observation ID, proposals, terminal flags, and per-step audit references.
Direct Python and HTTP use this same core. A later Runtime adapter only translates
contracts, as `robocasa_current.py` does; it must not duplicate the step loop.

Execution-tool protocol (read-only and episode-control handling is specified in
section 5): `prepare(arguments, approved_context) -> PreparedTool`, then
`next_targets(approved_context) -> bounded target batch | done`. Preparation
validates the whole EEF/gripper plan before motion; Zeva produces at most one
validated inference chunk at a time. Only the session core executes batches.
Tools have no environment handle. The Zeva adapter alone receives a private
policy-observation accessor; motion tools receive only command state and public
inputs. Gripper/EEF recovery must work even if the Zeva server is unavailable.

Reuse the ownership pattern of RoboCasa's proposal-only actor and LIBERO's
bounded tool/result accounting. Keep adaptive tool choice in the agent/episode
driver; do not reuse or modify `RecoveryController`'s ordered-step advancement
for adaptive ARX recovery. Version one exposes a small ARX recovery-context
record rather than pretending a tool completion means recovery completion.

Map service behavior to Runtime concepts: request-ID conflict detection,
per-session serialization, operation status, cancellation, and outcome-unknown.
Use the existing public error/state patterns as reference, with an explicit
adapter if Runtime is later used. Do not change the general Runtime API or
OperationRegistry. SQLite persistence is ARX-local; the existing Runtime registry
only guarantees idempotency within its process lifetime.

## 4. Public API and strict envelopes

Provide loopback HTTP initially, with per-episode bearer capabilities and request
size/time limits. Bind externally only with an explicitly configured secure
transport. The agent capability grants access to one episode; administrative
reset, episode creation, package selection, and seeds use a separate harness
capability. No arbitrary file, shell, Python, scene editing, or reset tools.

Endpoints:

```text
GET  /v1/episodes/{id}/catalog
GET  /v1/episodes/{id}/observation
GET  /v1/episodes/{id}/events?after={sequence}   # bounded long poll, <=30 s
GET  /v1/episodes/{id}/images/{content_id}
POST /v1/episodes/{id}/operations
GET  /v1/episodes/{id}/operations/{request_id}
POST /v1/episodes/{id}/operations/{request_id}/cancel
```

The image endpoint serves only registered public image IDs in this episode,
not filesystem paths. Tool schemas are exported from the same validators used
by the server. Reject unknown keys, wrong types, NaN/Infinity, booleans supplied
as numeric integers, malformed IDs, and excess payload sizes.

Example operation request:

```json
{
  "schema_version": "arx.tool.request.v1",
  "request_id": "decision-12",
  "decision_ref": "deployment-decision-12",
  "observation_id": "obs-42",
  "control_epoch": 3,
  "tool": "arx.move_eef",
  "arguments": {
    "delta_xyz_m": [0.002, 0.0, -0.003],
    "delta_rotvec_rad": [0.0, 0.0, 0.0],
    "frame": "tool",
    "speed_m_s": 0.01
  },
  "evidence_ids": ["rgb-right-42"],
  "reason": "Current RGB supports a small adjustment within visible free space."
}
```

The episode binding supplies package/catalog identity server-side. For agent
requests the runner persists the validated decision and registers `decision_ref`
with its invocation/evidence digest before submission (following Role1's
persist-before-effect pattern). The service rejects unknown decision references.
Nominal runner calls use harness-registered decision records with source=runner;
they are never attributed to an LLM. This is an internal harness interface, not
a public tool for agents to register arbitrary authority.

An accepted POST returns HTTP 202 with an operation ID and status URL. A completed duplicate
returns its existing result. A validation rejection returns a typed error and
is recorded as attempt feedback even though no operation is dispatched.

Result fields:

```text
schema_version = arx.tool.result.v1
request_id, operation_id, event_sequence
status = accepted | running | completed | interrupted | rejected |
         failed | cancelled | unknown
tool, observation_id_before, observation_id_after, control_epoch
executed_steps, planned_steps (nullable for VLA until inferred)
write_certainty = none | known_partial | completed | unknown
error = null | {code, phase, retry_class, public_message}
critic_event_ids[], result: object|null, budget_remaining{}
```

`completed` means execution completed, not recovery succeeded. Results expose
only typed measurements. Full exceptions and stack traces remain private;
sanitized error codes and useful contract diagnostics are public.

## 5. Frozen version-one tool catalog

| Tool | Arguments and fixed bounds | Meaning |
| --- | --- | --- |
| `arx.zeva` | `max_chunks`: integer 1..4; optional current `reentry_token` | Fresh inference and bounded execution, interruptible after every physical action |
| `arx.hold` | `steps`: integer 1..15 | Execute last committed command; physics advances |
| `arx.set_gripper` | `opening`: finite 0..1; `max_steps`: integer 1..60 | Right gripper only, preserving other command channels |
| `arx.move_eef` | Translation norm <=0.01 m; rotation norm <=0.1 rad; frame `world` or `tool`; speed in (0,0.03] m/s | Relative right-arm command-space IK, max 60 physical actions including settling |
| `arx.review_reentry` | `observation_ids`: 1..30 unique ordered IDs already exposed in this episode | Read-only frozen observation-derived reentry assessment |
| `arx.finish` | `reason`: nonempty string | Close the trial; never asserts success |

GET observation replaces a reobserve tool. Version one does not expose generic
joint targets or geometry queries. Visual geometry may be used inside a reviewed
reentry implementation; adding a public geometry tool requires a catalog version.

For gripper mapping, derive raw hardware targets from the calibrated mapping
and task offsets; do not copy the example's `-3.4 * opening` constant blindly.
Tests must check open/closed endpoints through the actual mapping and offsets.
For EEF motion, plan all waypoints before any write. Use <=1 mm translation
waypoints, bounded angular increments and a fixed settling allocation. Reject
unreachable, joint-limit, or over-budget plans before execution. Finishing the
command plan is not proof the robot reached the predicted TCP.

Tool numeric bounds are gateway-owned. A learning package may reduce them, never
expand them. Static joint limits must come from a reviewed robot-only calibration
artifact. Do not hand the recovery planner an entire scene XML containing object
locations. These tools are not collision planners. Preserve existing action
filters and locks; any added actuator envelope must be validated and versioned.

### General tool registration interface

Implement this interface in ARX-local `tools.py` and `contracts.py`. Follow
LIBERO's schema + handler registration and RoboCasa's immutable metadata/catalog
patterns. Do not extend shared `ToolSpec`, whose namespaces currently exclude
`arx.*`, or alter either existing robot's dispatcher.

```python
@dataclass(frozen=True)
class ArxToolSpec:
    name: str                          # arx.*; unique
    version: int                       # positive contract version
    description: str
    kind: Literal["read_only", "execution", "episode_control"]
    capabilities: tuple[str, ...]
    input_schema: dict                 # JSON Schema 2020-12, object root
    output_schema: dict                # schema of tool-specific public result
    allowed_states: tuple[str, ...]
    requires_reobservation: bool
    requirements: tuple[str, ...]       # named, trusted dependency bindings

class ArxToolRegistry:
    def register(self, spec: ArxToolSpec, handler: ToolHandler) -> None: ...
    def freeze(self) -> FrozenToolCatalog: ...
    def describe(self) -> dict: ...
    def resolve(self, name: str) -> RegisteredTool: ...  # service only
```

Register trusted handlers at service startup; reject duplicate names, unknown
kinds, incompatible handler protocols, invalid schemas, unresolved dependencies,
and absent descriptions/capabilities. Deep-copy and freeze schema/metadata values
so callers cannot mutate the catalog through a nested dictionary. After `freeze`,
registration/replacement is forbidden until a new service/package configuration
is loaded between episodes. No registration endpoint or learner-supplied import
path is exposed. Capability labels describe behavior; they do not grant permission.

`describe()` returns `schema_version=arx.tool.catalog.v1`, sorted public specs,
and `catalog_sha256`. Hash canonical public specs, including input/output schemas,
versions, categories, requirements, allowed states, and operational metadata.
Bind actual handler code/configuration hashes separately in the frozen runtime
manifest. Neither callable names, credentials, nor service URLs are public.
External-service versus local implementation is a private dependency choice,
not a side-effect category. In particular, Zeva uses `CosmosEdgeClient`, not
RoboCasa's HTTP service adapter.

Registration example (schema constants are defined below):

```python
registry.register(
    ArxToolSpec(
        name="arx.hold", version=1, description="Hold the committed command.",
        kind="execution", capabilities=("motion.hold",),
        input_schema=HOLD_INPUT, output_schema=EXECUTION_OUTPUT,
        allowed_states=("INTERRUPTED", "RECOVERING"),
        requires_reobservation=True, requirements=("command_state",),
    ),
    HoldPlanner(),
)
catalog = registry.freeze()
```

Task/package tool allowlists and current recovery authorization further restrict
`allowed_states`; a catalog entry is not sufficient authorization. Version-one
state sets: Zeva in `READY/RUNNING_NOMINAL` and, with valid reentry authorization,
`INTERRUPTED/RECOVERING`; hold/gripper/EEF and review in
`INTERRUPTED/RECOVERING`; finish in every idle live state and idempotently in
`ENDED`. `EXECUTION_UNCERTAIN` permits status/read-only snapshot and harness
closure only. No new tool can bypass that global restriction.

### Basic schemas and public results

All argument/result object schemas use `additionalProperties: false`. Unknown
arguments are errors, not silently dropped. Numeric validators additionally reject
nonfinite values and booleans where numeric values are required. Validate JSON
Schema and semantic constraints before execution. JSON Schema `default` is only
an annotation: the server applies the explicit defaults listed below after
validation, journals effective arguments, and hashes the original request for
idempotency. A retry must use the original request bytes/values, not reconstructed
effective arguments.

`HOLD_INPUT`, a basic execution input:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 15}},
  "required": ["steps"],
  "additionalProperties": false
}
```

`REVIEW_INPUT`, a basic read-only input:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "observation_ids": {
      "type": "array", "minItems": 1, "maxItems": 30, "uniqueItems": true,
      "items": {"type": "string", "minLength": 1}
    }
  },
  "required": ["observation_ids"],
  "additionalProperties": false
}
```

Review additionally requires episode-owned, previously exposed IDs ordered by
step, ending at the request's current observation. No arbitrary path/URL input.
The common operation result from section 4 adds a required `result` field:
`null` before a result is available, otherwise a value validated against the
registered output schema. Keep errors in the common `error` field. Use this validated `result` for tool-specific public measurements.

Execution result schema (`EXECUTION_OUTPUT`):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "completion": {
      "enum": ["plan_exhausted", "budget_exhausted", "critic_interrupted",
               "environment_ended", "cancelled", "error"]
    },
    "last_committed_step": {"type": "integer", "minimum": 0},
    "command_target_reached": {"type": ["boolean", "null"]},
    "physical_arrival_verified": {"const": false}
  },
  "required": ["completion", "last_committed_step", "command_target_reached",
               "physical_arrival_verified"],
  "additionalProperties": false
}
```

`command_target_reached` concerns committed command-space convergence only, never
measured arrival or grasp success; use null when no such criterion applies.
`executed_steps` in the envelope is authoritative, including zero-step interruption.
No-op completion is valid for an already-satisfied command target; record zero
steps and true command-target status. A critic interruption never counts as plan
completion. Budget exhaustion yields operation `completed` with
`completion=budget_exhausted`; that means the bounded call ended, not its target
was reached. Recovery remains active.

Read-only review result is `{assessment, reentry_token}`: `assessment` must
validate against `arx.reentry.assessment.v1` in the critic plan; token is a string
only for eligible/current assessments, otherwise null. Reject additional keys at
both levels. The gateway, not the read-only handler, creates the token after
checking observation/epoch freshness. Review records `executed_steps=0`,
`write_certainty=none`, and identical before/after observation IDs. Journal/token
bookkeeping does not imply a physics write. Finish returns
`{closed: true, finalization: "complete" | "incomplete"}`; finalization errors
remain explicit and cannot publish a valid completed trajectory.

Per-tool argument contracts beyond the examples:

| Tool | Required arguments | Optional defaults and additional validation |
| --- | --- | --- |
| `arx.zeva` | `max_chunks` | `reentry_token` omitted in nominal mode; required in recovery, nonempty string; no instruction/endpoint/seed override |
| `arx.hold` | `steps` | No defaults; always attempts that many physical holds unless interrupted/terminal/budget-limited |
| `arx.set_gripper` | `opening`, `max_steps` | No defaults; command completion tolerance is frozen in handler config and reported in catalog requirements/config identity |
| `arx.move_eef` | `delta_xyz_m` | `delta_rotvec_rad=[0,0,0]`, `frame="tool"`, `speed_m_s=0.01`; vectors exactly three finite numbers; enforce norm bounds from section 5 |
| `arx.review_reentry` | `observation_ids` | No defaults; assessment is not supplied by the agent |
| `arx.finish` | `reason` | Nonempty bounded string, max 2048 characters |

Use the bounds in section 5 in the generated schemas. Semantic validators enforce
vector norms, freshness, command feasibility, and remaining episode budget.
Gripper repeats its target until command tolerance or `max_steps`; EEF executes
its prevalidated waypoints/settling plan. Neither checks hidden object/contact
state to declare completion. Record truncation by the episode budget explicitly.

### Handler protocols and dispatch

```python
class ReadOnlyHandler(Protocol):
    def inspect(self, arguments, context: PublicReadContext) -> dict: ...

class ExecutionHandler(Protocol):
    def prepare(self, arguments, context: ApprovedToolContext) -> PreparedTool: ...

class PreparedTool(Protocol):
    def next_targets(self, context: ApprovedToolContext) -> TargetBatch | None: ...
    def on_commit(self, feedback: ApprovedExecutionFeedback) -> None: ...
```

Execution handlers produce plans, not side effects. `TargetBatch` is a nonempty,
finite float32 `[N,14]` array plus internal provenance; `None` means exhausted.
The session core validates each full batch before its first write, enforces tool
and episode horizon limits, calls the existing action processor only through
`env.step`, and reports actual commits. Feedback includes committed command,
executed count, public observation identity, and interruption/terminal status;
no privileged evaluator data. Zeva's approved context alone includes the trusted
policy-observation accessor. Read-only contexts contain only permitted public
frames/calibration and cannot call inference or stepping.

Dispatch by registered kind after common identity/decision/authorization checks:

1. `read_only`: capture an immutable public snapshot, invoke `inspect`, validate
   output, and publish without calling `execute_targets`. Version one serializes
   review with active operations; cached GET observation/status remain available.
   Reject token publication if epoch/observation changed while review ran.
2. `execution`: prepare without writes, persist admission/acknowledgement, then
   feed target batches through the sole session-core execution loop. The core
   constructs execution results from commits; do not trust a planner's claimed
   executed count. Validate public output before publication. If output validation
   fails after motion, retain known execution evidence and return an error;
   never label it a pre-write rejection or automatically retry.
3. `episode_control`: dispatch only through a fixed service-owned mapping
   (`arx.finish` initially). Cancel/finalize using existing lifecycle rules;
   never send an empty target batch to pretend finish is a motion primitive.

GET observation is the public read-only snapshot interface, not a separately
registered model tool in version one. If an agent framework needs an observation
function, expose a client wrapper around that GET with the same public schema;
do not create another environment observation path. Agent-visible tool descriptions/schemas are generated from the registry. Under
the version-one Role1-style invocation policy, motion tools are choices in the
structured decision, not executable model SDK functions. The runner dispatches
the selected tool after validation/persistence and supplies the common envelope
and decision reference. Only read-only image/evidence tools run inside the model
invocation; gateway execution feedback is supplied to the next fresh decision.

Registration/dispatch tests must cover duplicate/frozen registration, schema
mutation, schema/catalog hash changes, mismatched handler kinds, unknown arguments,
default normalization, read-only zero writes, finish without stepping, unauthorized
state/tool use, bounded target batches, post-write result-validation faults, and
agreement between model-visible schemas and server validation.

## 6. Zeva tool and action scheduling

Reuse `CosmosEdgeClient.predict`, `prepare_model_state`, model contract loading,
and the runner's three camera ordering. Preserve Task7 semantics: 32 absolute
14D targets, first `task.execution_steps` (currently 16) executed per chunk,
left-state preparation, immutable task instruction, and harness-owned inference
seed. The agent cannot select endpoint, checkpoint, seed, instruction, or scales.

On every `arx.zeva` invocation infer from the current observation; never reuse
unexecuted targets from an interrupted call. Within a call infer again after
each executed prefix. Inference failure advances no physics. Validate the entire
returned chunk before accepting any target.

For execution tools only, the executor follows this ordering (read-only and
episode-control tools use section 5 dispatch):

```python
validate_request_and_registered_decision(request)
prepared = tool.prepare(arguments, approved_context)  # no physics or inference
persist_operation_and_authorized_recovery_acknowledgement(request)
require_dispatch_allowed()  # uses suppression only for authorized recovery
for raw_target in bounded_targets(prepared):                  # Zeva plan can fetch bounded new chunks
    require_live_owner_and_budget()
    require_current_assessment_or_reset_baseline()  # no repeated evaluation
    if must_interrupt():
        discard_plan_and_record_interruption()
        break
    journal.step_intent(raw_target)
    commit = backend.step(raw_target)    # sole processor/filter invocation
    journal.step_commit(commit)
    update_command_and_observation(commit)
    assess_new_observation_and_filter_trigger_rules()  # includes final tool action
    if commit.environment_ended:
        close_episode()
        break
publish_public_result()
```

Check critic state before inference as well so a latched failure does not waste
a VLA call. An interrupt derived from the final observation takes precedence
over ordinary tool completion. Terminal closure takes precedence over recovery;
retain any coincident critic event for offline analysis only.

## 7. Control state and reentry

States: `READY`, `RUNNING_NOMINAL`, `INTERRUPTED`, `RECOVERING`,
`EXECUTION_UNCERTAIN`, `ENDED`. An independent active-operation flag enforces
one motion/inference operation in flight. Reads return immutable snapshots.

- The deployment runner starts nominal execution by calling `arx.zeva`; after
  ordinary bounded completion it may call again without an LLM turn.
- A critic interrupt invalidates the remaining tool plan, increments the control
  epoch, and publishes an interrupt event. The episode driver, not the gateway
  core/service, makes a fresh planner invocation for each decision with the frozen
  skill, current public evidence, recovery context, and bounded explicit history.
  There is no episode-long model conversation; the runner owns continuity.
- After validating the first recovery motion, persist acknowledgement of the
  pending trigger rule IDs and transition `INTERRUPTED` to `RECOVERING` BEFORE
  the action gate and any physical write. Failed validation does not acknowledge.
  This follows LIBERO `begin_recovery_step`; no detector re-evaluation or history
  reset occurs. Further decisions may adapt to fresh observations.
- While interrupted/recovering, `arx.zeva` requires a reentry token. Review
  applies the frozen observation-only policy described in the critic plan.
- Tokens bind episode, package, observation, control epoch, recovery ID, and
  assessment. Any motion or new interrupt invalidates them. Successful admission
  consumes the token and starts a fresh Zeva call.
- Budget exhaustion or agent loss closes as unsuccessful if outcome and execution
  are known; uncertainty closes infrastructure-invalid. No unsupervised fallback.

Reentry checks must be executable predicates or validated assessment results;
prose in the skill cannot mint a token. `finish` is always available when idle,
including after invalid recovery requests. While a call is active, cancel first
and wait for its final status. A separate admin stop can latch closure at the
next action boundary.

Expose `RecoveryContext` in every public snapshot/result:
`recovery_id`, selected binding/skill entrypoint, original triggering rule IDs,
latest proposal IDs, current permitted tools, remaining recovery steps/decisions,
last execution status, and reentry check failures/token availability. One active
recovery is allowed. A different unsuppressed rule stops the current tool and
closes with `RECOVERY_ESCALATION_REQUIRED`; no implicit nested recovery or
suppression expansion. Retain full feedback for later improvement.

Follow the critic plan's firing-event semantics: no inferred active/cleared
incidents and no automatic interrupt on diagnostic `unknown`. A specific
observation-quality rule may fire and have a normal recovery binding. Gateway
state retains pending recovery even when subsequent observations have no firing.
Lifecycle changes do not re-evaluate the cached image; control epochs invalidate
stale commands/tokens while critic results remain tied to physical observations.

## 8. Persistence, retries, and faults

Use a SQLite journal with WAL and FULL synchronous mode, plus content-addressed
image/artifact files. Unique key: `(episode_id, request_id)`. Store canonical
request digest, validation result, state transitions, physical-step intents and
commits, and immutable final results. Persist files before referencing them.

Under one executor admission lock, check duplicate identity before stale-state
validation. Same ID + same digest returns existing status without execution;
same ID + different digest returns `REQUEST_ID_CONFLICT`. Check observation ID,
epoch, state, bounds, and budgets atomically before accepting new work.

There is no atomic transaction between SQLite and MuJoCo. A crash between a step
intent and its durable commit means execution is unknown. Never promise exactly
once across simulator crashes and never replay that target. Mark the old episode
lost; run any authorized retry as a fresh isolated attempt with a new reset.
After a mere client disconnect, the live executor may finish its bounded call;
the client queries the same request ID to reconcile it.

Cancellation is cooperative at action boundaries, not a rollback. Record the
actual executed prefix. Timeout during inference can safely report no writes;
timeout during `env.step` cannot be assumed safe. If physics may have advanced,
enter `EXECUTION_UNCERTAIN` and block all further motion. Critic timeout holds
the simulation and becomes a typed error, never implicit acceptance.

### Worker liveness and finalization

Use a service parent and one serialized environment worker process per episode.
Construct/render/step/close MuJoCo in that worker. The parent owns durable request
status, a monotonic watchdog, and an admin lease, so status reads remain responsive
if native stepping/rendering hangs. Frozen configuration must specify operation,
critic, idle-agent, and worker-shutdown deadlines; do not invent runtime defaults
from the model's response speed. The harness renews its lease/heartbeat during
long agent inference, following `RolloutSession` and runner heartbeat patterns.

A client disconnect does not cancel a bounded operation. After it completes,
remain paused if recovery/agent review is required. On lease expiry, request
cooperative cancellation at the next action boundary and finalize. If the worker
misses its shutdown deadline, terminate the worker, mark the session lost and
uncommitted writes unknown, retain partial artifacts, and require a new attempt.
Never restart physics in place from the journal. Learner-authored feature code
runs in the isolated worker specified in the critic plan; it is never imported
into the environment/service process. The critic adapter enforces IPC observation
identity, output validation and deadlines. On worker failure block motion, close
the attempt, and return critic-error feedback; completed physics commits remain
known. Isolation capability is a preflight requirement, not a deferred extension.
The environment watchdog independently handles native stepping/rendering hangs.

Finalize videos/audit streams before publishing a completed episode, separately
from environment teardown, as in RoboCasa `finalize_episode_artifacts`. On a
forced worker kill, mark incomplete artifacts and do not publish a valid completed
trajectory. Bounded finalization failure must not erase the operation journal.

## 9. Recording and implementation sequence

Record public observations/images, tool requests/results, critic assessments,
exposure logs, raw/processed targets, VLA chunk metadata, and private evaluator
traces in distinct namespaces. Generate `EpisodeRecord` only after final
validation; record every rejected/partial attempt in the separate feedback
ledger from the learning plan. Do not return raw camera-runner `audit.json`,
`trajectory.npz`, or `diagnostic_trace.jsonl` to deployment.

Implement in reviewable stages:

1. Add contracts, catalog, command ledger, direct backend, and transport-independent
   session core. Reuse the existing client/environment classes without refactoring
   the validated camera runner in the first implementation.
2. Serialized tool executor, journal, idempotency, and public HTTP/client API.
3. Observation/critic barrier and interruption state machine.
4. Bounded gripper/EEF tools and reentry policy integration.
5. Deployment runner, fault injection, and final artifact publication.

Add proposed `scripts/deployment/serve_arx_gateway.py` with the running guide's
`--scene --mapping --task --contract --host --port` inputs (name Zeva endpoint
flags `--zeva-host/--zeva-port` to distinguish the new `--listen-port`), plus
`--package --output --seed`. OSMesa remains the documented backend. Episode
budgets come from the frozen harness configuration, not agent arguments.

Acceptance: baseline parity with fake Zeva; 1/15 s per accepted policy action;
no processor/physics mutation on rejection; no simultaneous tools; fresh
inference after recovery; cancellation prefix accuracy; duplicate/conflicting
IDs; crash-between-intent-and-commit handling; nested private-data exclusion;
gripper mapping endpoints; joint/IK bounds; terminal interruption; and complete
output publication. Then run the documented live checkpoint smoke test through
the gateway with no critic before enabling recovery.

## 10. Additive implementation and final review checklist

First implementation adds `robots/arx/gateway/`, the new serve/episode entrypoint,
and ARX tests. Existing Zeva runner, LIBERO/RoboCasa paths, shared TemporalCritic,
RecoveryController, and Runtime contracts remain unchanged. A read-only command
accessor in `ArxMujocoEnv` is the only anticipated existing-class addition; verify
it does not change reset/step outputs or processor behavior. No learning campaign
refactor is required to run a supplied frozen package through this gateway. The
critic isolation launcher and package-code loader are required additive modules.

Pass owned/read-only RGB buffers over bounded local IPC to the isolated feature
worker, not HTTP image round trips.
Persist original observation frames once, expose content IDs, and retain all
exposed frames for the episode's bounded lifetime. Critic history is independently
bounded; event windows reference retained images. Record critic/inference latency
and wall-time slowdown; simulation stepping waits for assessments rather than
claiming real-time 15 Hz wall-clock execution. Final cleanup follows the frozen
artifact-retention policy, not an active agent's request.

Before implementation review is closed, confirm:

- One target loop serves every tool; the service and runner cannot step physics.
- Reset initializes critic baselines; only new post-step observations advance it.
- Rule-ID suppression is applied before loop interruption and removed on every
  recovery exit; its recurrence limitation is explicitly retained and tested.
- Decision persistence, request idempotency, and physical step commit are distinct;
  no uncertain operation is automatically replayed.
- Mock tests cover first recovery action, additional-rule interruption, diagnostic
  unknown without a proposal, stale reentry, and no-progress/zero-step results.
- Fault tests cover client loss, long agent inference, critic/renderer hang,
  lease expiry, terminal closure, partial video, and process death.
- Direct baseline parity and existing ARX control/client/environment tests pass
  before a live supervised trial; existing shared backend behavior is unchanged.

This is an implementation-ready additive path, not authorization to weaken a
failed check. Unknown incident recurrence and perceptual uncertainty are recorded
limitations; resolving them with a new generic state machine is out of scope.
