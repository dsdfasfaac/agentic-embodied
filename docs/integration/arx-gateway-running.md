# ARX gateway implementation and running

The gateway is additive under `robots/arx/gateway/`. The camera-chunk runner and
shared rollout/evolution code are unchanged. `EpisodeDriver` delivers a public
snapshot, events, catalog, and previous result to an episode-local `decide`
callback at every tool boundary. It persists the returned decision through a
harness callback before dispatching. Physics remains paused during agent turns.
Set `nominal_continuation=True` only when the frozen harness contract authorizes
bounded runner-driven Zeva continuation between recovery incidents.

The process boundary is `EpisodeWorker`: its factory constructs the backend,
critic, tool registry, and session core in the child process. Only
`ArxSessionCore.execute_targets` steps the backend. The parent serves cached
observations, operation status, images, and events from SQLite while the worker
is busy. Harness heartbeats maintain the lease; they do not extend the separately
configured agent-idle deadline. Hung workers are terminated after the shutdown
deadline, leaving an uncertain attempt that cannot resume or replay motion.

## Baseline server

Use the project's supported Python/runtime environment with its ARX dependencies,
plus Pillow for lossless public PNG storage. Supply all deadlines and budgets in
`--runtime-config`; no episode budgets are inferred from model latency.

```json
{
  "max_steps": 100,
  "max_decisions": 30,
  "max_recoveries": 3,
  "operation_timeout_s": 360.0,
  "critic_timeout_s": 5.0,
  "idle_agent_timeout_s": 600.0,
  "lease_timeout_s": 60.0,
  "shutdown_timeout_s": 10.0
}
```

These are examples for development, not campaign defaults.

```bash
MUJOCO_GL=osmesa python -m scripts.deployment.serve_arx_gateway \
  --scene PREPARED_BUNDLE --mapping MAPPING_JSON --task TASK_MANIFEST \
  --contract MODEL_CONTRACT --runtime-config LIMITS_JSON \
  --output NEW_EPISODE_DIRECTORY --seed 17 --baseline \
  --zeva-host 127.0.0.1 --zeva-port 5581 --listen-port 8091
```

The harness reads the private `capabilities.json` in the output directory.
Give deployment only `agent_capability` and `episode_id`, never the output mount
or harness capability. The listener accepts loopback addresses only.

Public endpoints follow `arx-mujoco-gateway-plan.md`. Harness-only endpoints are
`POST /admin/heartbeat`, `/admin/decisions`, and `/admin/stop`, authenticated with
the separate harness capability. Decision registration accepts exactly
`{request, source: "agent"|"runner", evidence}`. It binds the complete request
and exposure record; deployment cannot register its own authority.

`ArxGatewayClient` implements reads, submission, cancellation, and bounded polling.
After a transport failure it queries the same request ID. A failed reconciliation
must be resolved by that ID; never submit uncertain motion with a new ID.

## Recovery integration

A trusted worker factory supplies an observation-only critic implementing
`reset(observation, images)`, `observe(observation, images)`, and `lifecycle(event)`;
`observe` returns the strict `Assessment` model. Reset initializes only; each
successful step receives one assessment. Diagnostic unknown does not interrupt.
Registered critics include typed feature payloads and use the isolated package
loader described in `arx-critic-running.md`.

Supply validated `RecoveryBinding` objects and a read-only reentry handler.
Its `inspect(args, context)` returns `ReentryAssessment`, never a token. Context
contains only the requested public RGB window and recovery/policy identities.
The core mints and consumes observation/epoch-bound tokens. A new unsuppressed
rule during recovery closes the attempt. Suppression of repeated firings of the
same rule retains the reference approach's documented recurrence limitation.

The default registry includes only configured dependencies. Zeva, hold, gripper,
EEF, review, and finish use strict schemas generated from their validators.
Gripper recovery uses mapping endpoints; the existing processor consumes hardware
commands directly and does not apply the manifest's unused offset metadata.
EEF requires a reviewed robot-only `Calibration`, with six static revolute links,
unit axes/quaternions, joint limits, base transform, and TCP offset. No scene XML
is accepted by the planner. It validates the full bounded IK plan before motion.
Command FK is a prediction; neither motion tool certifies physical arrival.

## Artifacts and current limits

SQLite uses WAL/FULL synchronization. Separate records contain decisions,
request identity/results, raw step intents, processed commits, private measured
state/evaluation, public observation/event streams, and critic results. PNG bytes
are durable before their IDs are published. The HTTP surface serves only public
registered images and explicitly public records. Retained frames are bounded by
the frozen episode step budget.

This delivery provides the gateway and injection points, not the whole learning
pipeline. The CLI supports baseline mode or sealed Python feature-rule packages; see
`arx-critic-running.md`. A production visual critic/reentry policy, provider
capsule isolation, and canonical `EpisodeRecord`/video export are not implemented
here. Finalization currently describes the gateway journal and image artifacts,
not a validated learning trajectory. Live checkpoint parity and supervised
recovery trials remain required before campaign use.

Tests: `python -m pytest tests/test_arx_gateway.py` covers the turn loop, strict
arguments, interrupt/suppression/reentry, private-data exclusion, idempotency,
partial execution, worker hangs, lease expiry, HTTP authority, and image access.
Run socket tests in an environment permitting local sockets.

## Manual commissioning trace

`python -m scripts.deployment.run_arx_gateway_trace --scene BUNDLE --output NEW_DIR`
executes four live Zeva chunks, closes/opens the gripper, holds 2 seconds, then
commands world −Z 10 cm and back, world +X 10 cm and back, and finishes. Each
Cartesian leg uses successive calls below the 1 cm bound, checking command FK
against the leg target. Motion tools are natively available in READY and RUNNING_NOMINAL as well as
recovery states. Both tool completion and critic interruption return control to
the decision loop. Recovery bindings still restrict tools after interruption.

This test uses an explicitly recorded 2,000-step task copy (override with
`--max-steps`), because settling all bounded moves can exceed the scene's 600-step
budget. The source task is unchanged. Static robot calibration is extracted by
the trusted harness into a separate artifact before constructing the planner.
The planner never receives scene XML. EEF plans include up to 15 settling actions
and report committed-command convergence, not measured physical arrival.

`turns.jsonl` contains each request, result, before/after public snapshot, and
predicted TCP. `observations.jsonl` indexes every emitted RGB observation;
`public/images` retains lossless frames. Four MP4 files provide each camera and a
three-camera mosaic. `report.json` records completion or the first blocking error,
including partial-run video on failure.


## Complete three-terminal pipeline

Run from `/data4/zhengyikai/Agentic-Embodied`. These commands reuse the installed
MuJoCo 3.3.1 environment and existing ckpt1 VLA environment. Select an available
GPU before starting Zeva. No separate environment server is necessary: the
gateway constructs and owns MuJoCo in its child process.

Terminal 1 — VLA:

```bash
CUDA_VISIBLE_DEVICES=2 \
ZEVA_OUTPUT_DIR=/data4/zhengyikai/Agentic-Embodied/runs/zeva_gateway_server \
bash scripts/deployment/start_zeva_arx_task7_server.sh
```

Wait for `Server is ready and listening on tcp://127.0.0.1:5581`.

Prepare limits once (the existing scene task permits 600 steps):

```bash
cat > /tmp/arx-gateway-limits.json <<'JSON'
{"max_steps":600,"max_decisions":100,"max_recoveries":3,
 "operation_timeout_s":360.0,"critic_timeout_s":5.0,
 "idle_agent_timeout_s":1800.0,"lease_timeout_s":60.0,
 "shutdown_timeout_s":10.0}
JSON
```

Terminal 2 — gateway and environment, using the robot-only calibration retained
from the verified trace of this same scene:

```bash
MUJOCO_GL=osmesa XDG_CACHE_HOME=/tmp/zetta-arx-cache \
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  -m scripts.deployment.serve_arx_gateway \
  --scene runs/arx_pickup_test_tube_10_new/episode_000000 \
  --mapping runs/arx_pickup_test_tube_10_new/episode_000000/mapping.json \
  --task runs/arx_pickup_test_tube_10_new/episode_000000/task.yaml \
  --contract robots/arx/manifests/task7_model_a.yaml \
  --calibration runs/arx_gateway_trace_episode000000_attempt2/robot_calibration.json \
  --runtime-config /tmp/arx-gateway-limits.json \
  --output runs/arx_gateway_interactive_001 --seed 17 --baseline \
  --zeva-host 127.0.0.1 --zeva-port 5581 --listen-port 8091
```

Use a new output directory on each launch. Connect the console within the
60-second lease. `--baseline` disables critic interventions; it does not disable
tool selection after completed calls. To reproduce the full long trace, use its
saved `task.json` and set `max_steps` to 2000 in the limits file.

Terminal 3 — interactive HTTP harness client:

```bash
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  -m scripts.deployment.interact_arx_gateway \
  --capabilities runs/arx_gateway_interactive_001/capabilities.json \
  --url http://127.0.0.1:8091 --log runs/arx_gateway_interactive_001/console.jsonl
```

Enter one JSON tool choice at each prompt:

```json
{"tool":"arx.zeva","arguments":{"max_chunks":4}}
{"tool":"arx.set_gripper","arguments":{"opening":0.0,"max_steps":60}}
{"tool":"arx.set_gripper","arguments":{"opening":1.0,"max_steps":60}}
{"tool":"arx.hold","arguments":{"steps":15}}
{"tool":"arx.hold","arguments":{"steps":15}}
{"tool":"arx.move_eef","arguments":{"delta_xyz_m":[0.0,0.0,-0.01],"frame":"world","speed_m_s":0.03}}
{"tool":"arx.finish","arguments":{"reason":"Finished interactive test"}}
```

The console prints the catalog schemas, observation image IDs, previous result,
and event trigger. Its harness side registers decisions and renews the lease;
all actual tool submissions use the public HTTP API. This console has harness
credentials and is intended for trusted interactive testing. A deployment model
should receive only public inputs through an adapter, never these credentials.

Execution path:
`catalog schema → decision → registered ToolRequest → POST operations → worker →
core envelope/state/argument validation → handler.prepare → execute_targets →
backend.step → observation → critic → result → next decision`.
The previous recorded 50-call trace used this same path from registered request
through core execution, but bypassed HTTP and used scripted choices rather than
an LLM. Its journal has 50 decisions, 50 operations, 50 effective-argument records,
and 1083 paired physical-step intents/commits.
