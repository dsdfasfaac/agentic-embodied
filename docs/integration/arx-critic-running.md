# Candidate critic registration and deployment

Implemented in `robots/arx/critics/`. The learner writes a sealed candidate
package; the harness alone registers it. Registration does not import learner
code. `ArxCriticRegistry.register(manifest, config, feature_schema, package_root)`
validates the submitted files, then `preflight()` runs import/reset/extract in a
disposable OS-isolated worker. `freeze()` starts fresh live workers and prevents
further registration. `describe()` exposes manifest identities and the aggregate
digest. Multiple critics are supported with globally unique critic/rule IDs.

`tests/fixtures/arx_candidate_rgb_v1/` is a complete sealed example. It computes
mean front-camera red intensity, with a two-observation dwell above 0.5. It is
synthetic interface evidence, not a useful physical failure detector.

## Authoring

Copy the fixture to a new draft. Edit `critic/features.py`,
`critic/feature_schema.json`, `critic/config.json`, skill, bindings, and evidence.
Feature code implements `reset(observation, images, config)`,
`extract(observation, images)`, and `close()`. Output is exactly the declared
feature names, each `{valid: bool, value: scalar|null}`. Invalid evidence uses null.
Only declared cameras are supplied as owned read-only uint8 RGB arrays. The
critic sees public observation metadata, never measured state, commands, task
success, reward, environment handles, or simulator files.

Call `seal_candidate(draft_path, metadata)` to compute code/config/schema hashes
and the complete package file inventory. Metadata follows `PackageManifest` in
`contracts.py`/`packages.py`; the fixture shows all required fields. Supply actual
campaign contract, bootstrap, and tool-catalog digests for a campaign. The loader
accepts expected identity arguments for harness enforcement. The standalone CLI
currently validates file bindings but does not certify campaign compatibility or
promotion. Extra/unlisted files, symlinks, path escapes, bad hashes, undeclared
features, incompatible thresholds, duplicate IDs, and missing bindings fail.

The initial reentry implementation is the reviewed `always_ineligible` policy
shown in the fixture. It cannot mint tokens. New visual reentry implementations
require a separately registered trusted policy; this delivery does not pretend
the mean-red detector provides recovery clearance.

## Worker isolation

This launcher requires Linux x86-64, user/mount/network/PID namespaces,
`unshare`, `mount`, and libseccomp. It creates a chroot containing only the pinned
Python stdlib, numpy, runtime libraries, one immutable feature code file, and a
trusted IPC worker. Package evidence, scene files, checkout, host home, proc,
credentials, and gateway socket are absent. Mounts are read-only, scratch is
size-bounded tmpfs, memory/CPU/files/descriptors are bounded, and seccomp denies
network, process creation, ptrace, mount/namespace changes, and related escape
syscalls. Access-denial probes must pass before importing the candidate. This is
an OS isolation boundary, not an import allowlist. If unavailable, preflight fails;
there is no unsandboxed fallback. Choose a clean pinned runtime; do not modify it
while running a campaign.

Messages are length-bounded JSON with request/observation identity checks and
owned RGB byte payloads; never pickle. Timeouts, crashes, malformed outputs and
forbidden accesses terminate extraction and block further motion. Committed
physics remains known-partial. CPU budget is cumulative per worker; memory and
scratch limits apply to the entire episode.

## Run replay

```bash
/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python \
  -m scripts.deployment.check_arx_critic \
  --package tests/fixtures/arx_candidate_rgb_v1 \
  --output runs/new_critic_replay
```

Expected `fired_steps: [3]`. Output includes PNG evidence and assessments.
Adding `--scene runs/arx_pickup_test_tube_10_new/episode_000000` runs the same
loader/source against real MuJoCo and deterministic offline policy targets from
the camera-chunk runner. It writes video, observations, journal, requests/results.
The ordinary 0.5 threshold is not guaranteed to fire on a real scene.

A separately sealed test-only `ge 0.0` variant was run successfully at
`runs/arx_critic_mujoco_smoke`: it interrupted at `obs-2`, executed exactly two
physical actions, discarded the remaining targets, and accepted a finish call.
This is a functional test, not evidence of task effectiveness.

## Gateway deployment

Use the VLA/environment commands in `arx-gateway-running.md`. Replace `--baseline`
with `--package PATH --critic-runtime-config LIMITS_JSON`. For example:

```json
{
  "python": "/home/zhenyikai/miniconda3/envs/zetta-mujoco/bin/python",
  "max_history": 60,
  "max_evaluation_ms": 5000,
  "startup_timeout_s": 15.0,
  "memory_bytes": 2147483648,
  "cpu_seconds": 600,
  "scratch_bytes": 1048576,
  "image_width": 320,
  "image_height": 240,
  "max_message_bytes": 2097152
}
```

These are explicit development values, not campaign defaults. The gateway critic
timeout must cover the registered assessment deadline. Each successful step,
including the last action in a call, waits for all assessments. A firing rule
publishes its proposal and increments the control epoch; the result identifies
the current observation and recovery context. The next deployment turn receives
that observation and events. No next action runs while the deployment agent is
thinking. Reset initializes without firing, duplicate reads do not evaluate,
invalid evidence resets only affected rule histories, unknown does not interrupt,
and lifecycle notifications do not reset temporal state. Existing rule suppression,
terminal precedence and reentry authorization remain gateway-owned.

## Reserved PRM path

`FeatureSource`, `PrmRequest`, `PrmResponse`, and `PrmFeatureSource` define the
extension seam. A manifest selecting `source.kind=prm_service` is rejected with
`PRM_SOURCE_NOT_IMPLEMENTED`, never silently replaced by baseline monitoring.
No PRM server/client/model calls are implemented. Future adapters must bind task,
approved target-image hashes, model/preprocessing identity, score semantics and
resource limits under `arx.prm_inputs.v1`. They will return through the same
assessment/proposal barrier; RGB-only candidate code gains no network access.
