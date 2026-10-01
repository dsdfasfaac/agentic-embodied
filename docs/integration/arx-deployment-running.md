# Single-trial deployment runtime

Run from the checkout with an already-running Zeva server:

```bash
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  -m scripts.deployment.run_arx_evolution_rollout \
  --trial-config /absolute/path/trial.json \
  --output /absolute/path/new_attempt
```

One invocation owns one gateway subprocess, its environment/critic workers, and
fresh API planner invocations. It never starts/stops the shared Zeva GPU server.
Output must not exist. SIGINT/SIGTERM stop admission and invoke bounded cleanup.
The last stdout line identifies `result.json` and the attempt ID.

## Trial configuration

All paths must be absolute. Example baseline:

```json
{
  "schema_version": "arx.rollout.trial.v1",
  "trial_id": "baseline-001",
  "mode": "baseline",
  "environment": {
    "scene": "/data4/zhengyikai/Agentic-Embodied/runs/arx_pickup_test_tube_10_new/episode_000000",
    "mapping": "/data4/zhengyikai/Agentic-Embodied/runs/arx_pickup_test_tube_10_new/episode_000000/mapping.json",
    "task": "/data4/zhengyikai/Agentic-Embodied/runs/arx_pickup_test_tube_10_new/episode_000000/task.yaml",
    "model_contract": "/data4/zhengyikai/Agentic-Embodied/robots/arx/manifests/task7_model_a.yaml",
    "seed": 17
  },
  "vla": {"host": "127.0.0.1", "port": 5581, "expected_identity": {}},
  "gateway": {
    "python": "/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python",
    "host": "127.0.0.1",
    "port": 8091,
    "runtime_limits": "/absolute/path/gateway-limits.json"
  },
  "runner_limits": {
    "startup_timeout_s": 60.0,
    "episode_timeout_s": 1800.0,
    "reconciliation_timeout_s": 360.0,
    "shutdown_timeout_s": 15.0,
    "heartbeat_interval_s": 10.0,
    "max_tool_attempts": 100,
    "max_agent_calls": 10,
    "max_recovery_agent_calls": 5,
    "max_contract_retries": 1,
    "max_event_records": 10000
  },
  "evaluation": "none"
}
```

Gateway limits follow `arx-gateway-running.md`. Its idle deadline must exceed the
agent-call plus reconciliation deadline, and its lease must exceed three heartbeat
intervals. Example values are development inputs, not campaign defaults.
The reconciliation deadline must cover expected bounded VLA operation latency.
Current gateway does not attest checkpoint/runtime identity; nonempty
`vla.expected_identity` is explicitly rejected rather than falsely verified.
Only `evaluation="none"` is currently admitted, yielding null task success.

Candidate mode adds these required sections:

```json
{
  "candidate": {
    "package": "/absolute/path/sealed-candidate",
    "package_sha256": "<actual 64-character package digest>",
    "contract_sha256": "<harness-frozen campaign digest>",
    "catalog_sha256": "<expected gateway catalog digest>",
    "bootstrap_sha256": "<ARX adapter bootstrap digest>",
    "critic_runtime_limits": "/absolute/path/critic-limits.json"
  },
  "agent": {
    "planner_type": "api",
    "model": "<provider:model>",
    "reasoning_effort": "high",
    "max_tokens": 4096,
    "max_turns": 4,
    "timeout_s": 120.0,
    "credential_env": "OPENAI_API_KEY"
  }
}
```

Configure the provider's standard credential environment variable before launch;
only its name appears in the trial. The chosen reference must correspond to the
provider understood by existing `build_planner`. No new provider client is added.
CLI planner backends are rejected. Retrieve the exact bootstrap digest with:

```bash
python -c 'from robots.arx.deployment.agent import BOOTSTRAP_SHA256; print(BOOTSTRAP_SHA256)'
```

Seal a candidate against independently supplied campaign/catalog/bootstrap
identities. The example critic fixture's synthetic identities are not a valid
production trial configuration. Package bytes are validated, staged read-only,
and the live advertised catalog must match before admitting a tool.

## Execution and agent adapter

Nominal states automatically select `arx.zeva(max_chunks=1)`. Interrupted/recovering
states invoke a fresh existing API planner with current RGB, frozen skill,
resolved critic triggers, permissions, budgets, and the last 16 decision/results.
Complete public event pagination is drained before the decision; original active
trigger evidence is retained separately. Gateway action barriers continue owning
critic evaluation and physical stepping.

The read-only toolkit exposes `read_arx_image` and `submit_decision`. Image bytes
are checked against the published hash/shape, and current RGB must be inspected.
Decision submission only returns a strict ARX choice; it cannot execute motion.
The runner validates identity, permissions, registered schema and evidence, writes
the exact request/input digest, registers the decision, then submits one operation.
There is never a second submission or agent call while execution is unresolved.
Transport faults query the same request ID; no uncertain physical intent is retried
under a new ID. Reentry requires an actual current review token. The current
always-ineligible policy remains unable to resume Zeva.

Production planner invocations run in disposable processes for bounded timeout
and cleanup. They use the API backend and an explicit evidence-only Toolkit; this
is not a general-purpose hostile-provider filesystem sandbox. Candidate Python
critics retain their separate OS isolation. Shared agent/robot adapters are unchanged.

## Artifacts and status

`result.json` separates runtime status from task outcome. Exit codes: 0 completed
(including budget stop), 2 configuration failure, 3 infrastructure/agent/uncertain,
130 interruption. Always inspect the result. No success is inferred from finish.

Public evidence: `events/`, `tools/`, `invocations/`, `catalog.json`, `reset.json`,
`identities.json`. `private/` contains trial paths, staged package, gateway journal,
capabilities and process diagnostics; do not export it wholesale to the learner.
Invocations record inputs, inspected images, model output, validation and timing.
No video export is promised by this runner.

Tests cover baseline zero-agent behavior, trigger handoff, event pagination,
invalid-attempt budgets, exhausted gateway budgets, fresh planners, image tools,
missing-image failures, model timeout, same-ID transport reconciliation, structured
preflight failure and real localhost HTTP runner-to-worker execution. Full live
Zeva + real-provider candidate acceptance remains pending provider configuration.
