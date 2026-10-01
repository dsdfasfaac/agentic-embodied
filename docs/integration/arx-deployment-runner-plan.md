# ARX single-rollout deployment runner

Status: proposed, documentation only. No implementation changes are requested by
this plan. Builds on the implemented gateway/critic and the
[learning orchestration contract](arx-learning-orchestration-plan.md).
[Known issues](arx-gateway-critic-known-issues.md) remain recorded separately;
the runner requirements below describe future implementation, not fixes already
made or prerequisites to perform immediately.

## 1. Scope and entrypoint

Add `scripts/deployment/run_arx_evolution_rollout.py`, invoked as:

```bash
python -m scripts.deployment.run_arx_evolution_rollout \
  --trial-config /absolute/path/trial.json \
  --output /absolute/path/new_attempt
```

One invocation runs exactly one fresh episode and writes one machine-readable
result. The upper-level learner invokes this command through its rollout tool,
waits for completion, and reads the result/artifacts. It does not converse with
the deployment model during the episode. Multiple scenes/seeds are independent
invocations scheduled by the learner's harness. Clustering, candidate editing,
promotion, and restarting failed trials remain outside this script.

V1 connects to an already-running Zeva VLA server, as in the tested guide. The
runner starts the gateway subprocess, which starts MuJoCo and its critic worker.
It owns their cleanup. It does not start a GPU/model server per rollout or kill
the shared VLA service. No interactive console is required.

Use LIBERO `run_evolution_rollout.py` as a reference for nominal/recovery loops,
actor-call budgets, heartbeat, and durable artifacts. Use RoboCasa
`Role1ModelAdapter.decide()` for fresh planner construction, image tools, and
persisted invocations. Reuse patterns, not their environment/action schemas.

## 2. Frozen trial input

Define strict `arx.rollout.trial.v1` JSON, rejecting unknown fields. Resolve paths
before spawning subprocesses. Never interpolate the configuration into a shell
command; launch argument arrays with the configured interpreter.

| Field | Required content |
| --- | --- |
| `schema_version`, `trial_id` | Schema literal and opaque trial identity |
| `mode` | `candidate` or `baseline` |
| `environment` | Absolute `scene`, `mapping`, `task`, `model_contract`, optional `calibration`, and integer `seed` |
| `vla` | `host`, `port`, expected checkpoint/runtime identity where available |
| `gateway` | Python executable, loopback host, harness-assigned exclusive port, runtime-limits file |
| `candidate` | In candidate mode: package path, expected package/contract/catalog/bootstrap hashes, critic runtime-limits file |
| `agent` | In candidate mode: supported planner type, model, reasoning/settings, maximum planner turns, per-call timeout; credential reference, never credential value |
| `runner_limits` | Startup, episode, reconciliation, and shutdown timeouts; heartbeat interval; maximum total tool attempts, agent calls, per-recovery agent calls, and contract retries |
| `evaluation` | Trusted evaluator identifier/configuration, or explicit `none` |

Baseline mode uses `--baseline`, never invokes an agent, and records that no
learned critic was active. Candidate mode requires all candidate identities and
agent settings even for a trial where the critic never fires. Harness-controlled
limits cannot be expanded by candidate text. Runtime idle/lease limits must
accommodate configured agent/evidence/reconciliation deadlines; heartbeat alone
does not disable the gateway's idle-agent timeout.

The harness supplies expected contract/bootstrap identities from its frozen
configuration, not by copying the candidate's declarations. Validate package
integrity/identities with `load_candidate(expected_...)`, stage the validated
bytes in a private immutable attempt directory, and launch the gateway against
that staged copy. Compare the gateway's advertised catalog digest against the
expected catalog before the first physical tool. A startup mismatch yields a
configuration error and cleanup, never a baseline fallback. Treat initial reset
as distinct from admitting physical tool operations.

## 3. Small internal structure

Keep one CLI, with testable helpers rather than embedding gateway logic in it:

- `robots/arx/deployment/runner.py`: trial lifecycle, state dispatch, budgets,
  event collection, decision registration, operation reconciliation, final result.
- `robots/arx/deployment/agent.py`: ARX decision schema and fresh planner adapter;
  public evidence toolkit and fixed system prompt from the learning plan.
- `robots/arx/deployment/contracts.py`: strict trial/result schemas and exit codes.
- `scripts/deployment/run_arx_evolution_rollout.py`: parse, validate, invoke runner.

Reuse `serve_arx_gateway` as a subprocess and `ArxGatewayClient` for public calls.
A harness-only HTTP client calls `/admin/heartbeat`, `/admin/decisions`, and
`/admin/stop`. Keep harness capabilities out of the model payload and toolkit.
Reuse existing gateway tool schemas, catalog, package loader, critic registration,
and worker isolation. Do not write another physics loop or call critics in the
runner. Do not use the current `EpisodeDriver` unchanged: KI-01/02/03 explain why.
This additive runner can leave existing interactive paths intact.

### Agent reuse boundary

`robots/arx/deployment/agent.py` is a thin environment adapter around existing
agent infrastructure. Do not implement a new provider client, model/tool-call
loop, or agent framework. Keep the existing LIBERO and RoboCasa adapters unchanged.

| Existing component | ARX reuse |
| --- | --- |
| `zetta/planner/base.py:build_planner()` and planner implementations | Reuse model invocation and tool-call execution; construct a fresh planner for each decision |
| `zetta/tools/toolkit.py:Toolkit` and `ToolResult` | Reuse tool registration and multimodal image-result handling; register only ARX public evidence readers |
| `zetta/evolution/jsonio.py` | Reuse JSON persistence utilities for invocation and audit artifacts |
| `robots/robocasa/role1_agent.py:Role1ModelAdapter` | Follow its fresh invocation, read-only image inspection, validation, and artifact-writing pattern; do not use the complete adapter unchanged |
| `robots/libero/role1_recovery.py:LiberoRole1RecoveryActor` | Behavioral reference only; its prescribed recovery-step and direct primitive-execution assumptions do not match ARX |

The complete existing adapter is not a drop-in replacement for three reasons:

- RoboCasa decisions contain proposal dispositions, stages, and optionally a
  five-component `direct_action`. ARX decisions select a gateway tool and its
  arguments, bound to the current observation ID and control epoch. Implement
  the ARX schema and validator from the learning plan; do not translate ARX
  operations into artificial RoboCasa actions or weaken its existing validator.
- RoboCasa's image toolkit identifies encoded data URLs by their hashes. ARX
  publishes image IDs and integrity metadata through the gateway. Add a small
  image-reader adapter that resolves authorized published IDs, validates returned
  bytes against their metadata, and returns the existing `_image_bytes` result
  format. Keep gateway image IDs as the decision's evidence references.
- LIBERO's recovery actor expects a frozen recovery `current_step` and executes
  its primitives. The ARX agent chooses among permitted gateway tools using the
  skill and current evidence; the runner performs execution and the gateway
  enforces reentry. Do not call `decide_and_execute()` for ARX motion.

The new adapter's complete responsibility is:

```text
public ARX event + frozen skill + explicit bounded history
    -> fresh existing planner.solve() + read-only evidence Toolkit
    -> strict ARX decision validation and invocation audit
    -> one decision returned to the runner
```

The runner owns request IDs, durable decision registration, gateway submission,
result reconciliation, and continuity between invocations. The fixed ARX system
prompt comes from the learning plan; the learned skill remains event content.
Do not import RoboCasa's system prompt or retain a planner instance across calls.

Use the existing API planner for the first implementation. CLI-based planner
backends currently configure additional environment memory directories; their
filesystem access, built-in tools, and context persistence require a separate
deployment-capsule check before enabling them. Do not assume that restricting
`Toolkit` alone restricts a CLI agent's other capabilities. This is a backend
admission requirement, not a request to rewrite shared planners now.

## 4. Startup and ownership

1. Exclusively create the output directory; persist resolved trial input and
   identity hashes with secrets redacted. Reject reuse of an existing attempt.
2. Validate input, limits, frozen package, skill references, and runtime settings.
3. Launch `serve_arx_gateway` with the existing scene/mapping/task/model-contract,
   calibration, seed, VLA endpoint, limits, and package/baseline options. Give it
   a new child output directory because the service creates its own directory.
   Capture stdout/stderr privately and use OSMesa/cache settings from the guide.
4. Await capability file and authenticated catalog/observation readiness with a
   bounded deadline; verify episode identity, process liveness, and catalog.
   Start harness heartbeats as soon as credentials are available. File existence
   alone is not readiness. A port collision must not attach to another episode.
5. Preserve the reset observation, then enter the loop below. Each subprocess
   failure or startup timeout produces a structured result even before readiness.

The model receives only task language, skill and allowlisted skill references,
public catalog, RGB evidence, critic proposals, recovery context, and budgets.
The runner may read private configuration/evaluation data for orchestration but
must never forward it. Provider toolkits expose no filesystem, shell, HTTP admin,
simulator, or motion tool access. If a provider requires a process sandbox,
launch it with only the same public capsule and audit the exposed capabilities.

## 5. Deployment loop

```text
while episode not terminal and runner deadline/budgets permit:
    reconcile any pending operation using its original request_id
    capture stable gateway snapshot; drain its public events
    if gateway ended or execution uncertain: stop
    if READY or RUNNING_NOMINAL:
        choose arx.zeva(max_chunks=1), source=runner
    elif INTERRUPTED or RECOVERING:
        build public recovery event; invoke fresh agent; validate one decision
    else:
        terminate with explicit unsupported-state error
    persist input/decision; register decision; submit exactly one tool
    await/reconcile final result; persist result and update bounded history
finalize with known outcome and stop owned processes
```

Critics remain online inside gateway action barriers, including during recovery
motion. The runner learns about interruption from the operation result/snapshot
and public events. It does not poll a second critic, ask the learner what to do,
or continue nominal Zeva while waiting for the deployment decision. MuJoCo does
not advance through physical tool calls during agent thinking.

**Complete event handoff.** With no pending operation, capture snapshot sequence
S and request event pages from the last consumed cursor until a short/empty page
or an event beyond S proves collection complete; include only records through S.
Sequences can skip private records. Bound collection by deadline/record budget;
on incomplete evidence stop rather than decide blindly. If a concurrent terminal
transition changes the snapshot, refresh before deciding. Retain the active
recovery's original proposals separately from the rolling event window and
verify all `latest_proposal_ids` are resolved. Fetch image bytes by published ID;
never supply host image paths to the model.

**Fresh decision call.** Follow section 5 of the learning plan exactly: fixed
system contract, fresh planner per call, no resume/thread ID, frozen skill,
current RGB, allowed tools, proposals, current budgets, and the last 16 completed
decision/result records. Keep active recovery evidence outside that truncation.
Use read-only image tools within `planner.solve(...)`. The model returns one
`arx.deployment.decision.v1`; it cannot execute motion during reasoning.
Record malformed outputs/timeouts as attempts as well as valid choices.

Validate event/observation/epoch IDs, catalog argument schema, recovery permissions,
and evidence ownership. Assign unique request/decision IDs, persist the exact
request plus input digest, then register `{request, source, evidence}` through
`/admin/decisions` before submitting it. Automatic nominal choices use
`source=runner`; model choices use `source=agent`. Do not replace an invalid
model action with invented recovery motion.

**Reentry.** The agent first chooses `arx.review_reentry`; a subsequent fresh
call sees its actual result. Only an eligible result with a current token permits
an agent choice of `arx.zeva` carrying that token. After successful gateway
transition to nominal, automatic one-chunk continuation resumes. The current
`AlwaysIneligibleReentry` cannot do this: report its failed checks and allow
bounded further recovery or finish. Do not treat suppressed critic proposals,
absence of a new trigger, or model confidence as clearance. Support the existing
token path so a future trusted reentry policy needs no runner redesign.

## 6. Bounds, errors, and cleanup

Runner counters independently bound total new tool attempts and model calls,
including validation failures. Transport polling/reconciliation of the same
request does not consume another decision attempt. No pending operation permits
another agent call or tool submission. Exhausted runner or gateway budgets cause
orderly finish/stop, not another nominal call; never retry exhaustion as a schema
correction. Track per-recovery counters by recovery ID.

On malformed model output or a known pre-write rejection, permit only the frozen
number of fresh correction calls, including failure feedback. On HTTP timeout,
query the original request ID; a timeout or failed status query is not proof of
zero writes. If status stays unresolved beyond the reconciliation deadline,
cancel/stop using existing endpoints and report execution uncertainty. Never
resubmit physical intent under a new ID. Preserve `executed_steps` and
`write_certainty` from gateway results; do not equate `completed` with task success.

On critic/model/service failure, close the trial with a typed reason. Do not
resume Zeva automatically. For normal finish, use the registered `arx.finish`
path if the gateway is responsive and execution is known; otherwise use harness
stop and bounded process cleanup. On SIGINT/SIGTERM, stop admission, reconcile
within deadline, close the owned gateway/process group, and write an interrupted
result. Never kill the shared VLA service. Cleanup failure and uncertain writes
must remain visible in the final result.

## 7. Result and learner interface

Write `result.json` atomically as `arx.rollout.result.v1` and print one final JSON
line containing its path and attempt ID. Send progress to stderr, not an
interactive question. Proposed exit codes: 0 for a valid completed trial
(including task failure/budget stop), 2 for configuration/preflight failure,
3 for infrastructure/agent failure or execution uncertainty, and 130 for user
interruption. Learners must read the result; exit zero is not task success.

Required result fields:

```text
schema_version, trial_id, attempt_id, episode_id (nullable before startup)
mode, package_sha256, contract/catalog/bootstrap identities
status: completed | configuration_error | infrastructure_error |
        agent_error | execution_uncertain | interrupted
termination_reason: explicit code, including agent_finish, environment_ended,
                    runner_budget, gateway_budget, critic_error, timeout
outcome: {task_success: boolean|null, evaluator_id: string|null,
          recovery_attempted: boolean, reentry_completed: boolean}
counts: {physical_steps, tool_attempts, rejected_tools, agent_calls, recoveries}
last_operation: {request_id, status, write_certainty} | null
artifact_paths, cleanup_status, error (nullable)
```

Keep task outcome and runtime status separate. In v1, `evaluation=none` yields
`task_success=null`; finish, termination, or red-intensity recovery cannot prove
success. A trusted evaluator may later use private state after execution, with
results confined to learner feedback and never the deployment prompt. Preserve
partial artifacts after failure and distinguish failures worth learning from
infrastructure invalidity.

Output layout: resolved trial/identities, frozen candidate, gateway directory
(journal, images, capabilities), public event log, tool requests/results,
per-invocation public inputs/images/model outputs/validation/timing, private
process diagnostics, and final result. Capabilities/provider credentials are
excluded from learner artifact exports. Do not promise video unless a separately
configured recording path actually produces one. No mandatory new gateway
recording implementation is needed for this runner.

The learner calls the same CLI for candidate, parent, and baseline trials through
an approved frozen trial specification. It receives exported evidence, diagnoses
execution errors and unresolved physical failures, and writes a new package for
a new attempt. It never edits the package used by an active episode.

## 8. Implementation and acceptance sequence

1. Add strict input/result schemas and subprocess startup/cleanup; verify baseline
   execution through the existing gateway using a deterministic fake backend.
2. Add complete event collection, independent budgets, same-ID reconciliation,
   and durable request registration. These are runner requirements addressing
   KI-01/02 without silently rewriting existing gateway behavior.
3. Add the fresh ARX planner adapter and public image toolkit; test with scripted
   decisions first, then a configured real provider. Keep shared Role1 code intact.
4. Integrate the sealed RGB feature fixture through the existing isolated worker.
   Run trigger → fresh decision → permitted tool/finish → exported result. With
   the current stop-only binding, finish is the expected supported path; use a
   separate allowed-motion test binding for recovery-action checks.
5. Run the tested MuJoCo scene with an already-running Zeva server and a real agent.
   Require artifacts proving the input was current, motion stopped at the trigger,
   and every recovery operation came from a persisted decision. Report reentry
   as unavailable until a real policy is separately implemented.

Required regression cases: no-trigger rollout makes zero model calls; step-two
trigger discards remaining targets; more than 100 events retains trigger evidence;
invalid decisions terminate within budget; exhausted gateway budget does not
loop; missing images fail without motion; repeated decisions use fresh planners
and explicit history; uncertain submission cannot duplicate motion; additional
critic failure during recovery respects gateway escalation; startup hash mismatch
admits no tool motion; model timeout and SIGTERM clean up; evaluator results stay
out of model inputs. Test future successful reentry with a trusted fake policy,
clearly separate from the live `always_ineligible` limitation. Resolve and rerun
the known HTTP/ASGI test gap before declaring end-to-end readiness.
