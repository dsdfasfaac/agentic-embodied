# Online ARX observation critic format and runtime plan

Status: proposed contract, paired with the [gateway](arx-mujoco-gateway-plan.md)
and [learning orchestration](arx-learning-orchestration-plan.md). The critic
listens to new environment observations, identifies problems, and reports
typed assessments. It cannot execute tools or choose replacement actions.

## 1. Runtime model

The gateway emits `ObservationPublished` after reset and every successful
physical `env.step`, regardless of whether Zeva, gripper, EEF, or hold caused
the step. The critic consumes these in order. The gateway waits for the matching
assessment before allowing the next physical step. MuJoCo is paused while
waiting; no background stepping is permitted.

This is an online observation listener with a synchronous execution barrier,
not a post-episode video evaluator and not an asynchronous monitor whose result
may arrive after the next motion. The critic reports a failure observed at step
t and can prevent step t+1; it does not claim to undo step t.

Use reset RGB to initialize visual baselines, but do not advance temporal rules
or emit learned failure proposals at reset. This follows the references'
first-post-step evaluation convention. Preflight validates required cameras and
configuration before motion. Repeated HTTP reads of the same observation do
not re-evaluate the critic or advance dwell/cooldown. Record assessments at the
last step even if the tool has no remaining targets. Terminal observations may
be recorded/analyzed but must never cause new recovery motion.

## 2. Inputs and allowed evidence

Version-one input profile is `arx.rgb_only.v1`:

```text
CriticObservation
  schema_version = arx.critic.observation.v1
  episode_nonce                 opaque, not a seed or scene identifier
  observation_id, step_index, simulation_time_s
  cameras: front_rgb, left_rgb, right_rgb
    each: content_id, sha256, width, height, encoding
  lifecycle: reset | nominal | recovery | reentry
  event_sequence
```

Actual pixels are delivered through read-only image handles/payloads. The
critic receives all synchronized public cameras but declares which it uses.
It may maintain its own bounded history. Lifecycle is trusted scheduling
metadata, not a target label. No action vectors or commanded/measured poses in
this initial critic profile. Recovery tools may independently use command
history as specified in the gateway plan.

Forbidden: environment handle, qpos/qvel/ctrl, measured joints, object or EEF
ground-truth pose, contact, reward, evaluator progress/success, scene XML,
reset state, seed, filenames identifying prior outcomes, private tool telemetry,
and offline reference trajectories. Do not rename privileged fields into
`visual.*`. Every derived feature must name its actual allowed source.

Static camera calibration can be an explicitly declared versioned dependency
for a visual geometry detector. Supply a camera-only calibration artifact, never
the full scene. Image-derived object position is an estimate and must include
validity/uncertainty, not ground-truth semantics.

## 3. Critic artifact format

Each deployment package contains a critic manifest:

```json
{
  "schema_version": "arx.critic.manifest.v1",
  "critic_id": "acquisition-monitor",
  "input_profile": "arx.rgb_only.v1",
  "implementation": "feature_rules",
  "source": {"kind": "package_python", "api_version": "arx.features.v1"},
  "entrypoint": "features.py:FeatureExtractor",
  "code_sha256": "<64 lowercase hex characters>",
  "config_sha256": "<64 lowercase hex characters>",
  "feature_schema_sha256": "<64 lowercase hex characters>",
  "cameras": ["front_rgb", "right_rgb"],
  "history_limit": 60,
  "evaluation_timeout_ms": 5000,
  "failure_modes": ["target_left_on_fixture"],
  "recovery_bindings": {"target_left_on_fixture": "recover-acquisition"}
}
```

Values above are illustrative; timeout and history bounds are fixed by the
harness before evaluation. All manifests and configuration use strict schemas,
reject unknown versions, and hash canonical JSON plus exact artifact bytes.

Version one supports `feature_rules`: learner-authored Python feature extraction
plus JSON rules evaluated by the unchanged TemporalCritic. There is no prerequisite
catalog of implemented visual features. Freeze the observation ABI, output schema,
dependency/runtime limits, and registration mechanism, not the feature names.
The learner supplies both feature declarations and executable implementations.

Separate two interfaces: a `FeatureSource` produces typed values and the trusted
`ObservationCritic` adapter evaluates rules and constructs assessments. Initial
source kind is `package_python`; reserve `prm_service` for a later implementation
(section 10). Unknown/unimplemented source kinds fail registration; no fallback.
This separation permits a fixed future model-backed critic without learning code.

The RGB example's `PhaseAwareRgbCritic` is a functional reference for phase
tracking, visual closure evidence, and uncertainty. Port behavior into a reviewed
extractor/detector; do not depend on its relative `sys.path` modifications,
session classes, or hardcoded scene without declared compatibility checks.

Python ABI:

```python
class ObservationCritic:
    def reset(self, config: CriticConfig, initial_observation: CriticObservation,
              images: Mapping[str, ReadOnlyRgb]) -> None: ...
    def observe(self, observation: CriticObservation,
                images: Mapping[str, ReadOnlyRgb]) -> CriticAssessment: ...
    def lifecycle(self, event: CriticLifecycleEvent) -> None: ...
```

`reset` resets episode history exactly once. Lifecycle events are sequenced and
journaled: `recovery_started`, `reentry_accepted`, `episode_closed`. In version one these are
audit/scheduling notifications; they do not reset TemporalCritic or re-evaluate
the cached frame. Any later phase-reset behavior requires a separately tested
detector version. Dispatch acknowledgement changes only gateway state.
Replay must use the same observation and lifecycle stream. No dependence on
wall-clock time, random seeds from the episode, or unseen future observations.

## 4. Assessment and failure event schemas

Return exactly one assessment for each post-step observation; reset records a
baseline-initialized marker instead of a firing assessment:

```json
{
  "schema_version": "arx.critic.assessment.v1",
  "critic_id": "acquisition-monitor",
  "observation_id": "obs-42",
  "step_index": 42,
  "status": "failure",
  "features": {
    "visual.target_supported": {"valid": true, "value": true},
    "visual.departure_count": {"valid": true, "value": 6}
  },
  "events": [{
    "detector_id": "failed-acquisition",
    "failure_mode": "target_left_on_fixture",
    "rule_id": "failed-acquisition",
    "evidence_observation_ids": ["obs-36", "obs-42"],
    "reason_code": "closure_then_departure_target_stationary",
    "summary": "Visual acquisition attempt followed by departure while target remains supported.",
    "limitations": ["RGB does not establish finger contact."],
    "proposal": "interrupt"
  }]
}
```

Assessment `status`: `clear | unknown | failure`. Events are firing proposals,
matching existing `TemporalCritic.evaluate()` semantics; there is no
`active/cleared` incident lifecycle in version one. The gateway assigns durable
proposal event IDs from episode,
critic digest, observation ID, and event index; the agent cannot supply them.
Validate that evidence IDs refer only to current/prior frames actually provided
to this critic. Store bounded textual summaries as untrusted evidence, not
instructions. No tool names, replacement targets, or success claims in events.

- `clear`: no currently applicable detected failure, not proof of task success.
- `unknown`: required visual evidence is invalid/occluded/ambiguous.
- `failure`: one or more rules fired on this observation; takes precedence over
  diagnostic unknown features in the aggregate status.
- Crash/timeout/invalid schema/missing required camera: a separate
  `CRITIC_EXECUTION_ERROR` from the runner, never a fabricated assessment.

An empty proposal list means no rule fired, not that a previous failure cleared.
Keep the pending recovery in gateway/controller state, as the existing runners
do. Never derive recovery completion from silence or cooldown expiry.

## 5. Feature and temporal-rule specification

Learner-authored feature schema entries contain:

```text
name, scalar_type, units, source_cameras, extractor_version,
validity_condition, sampling = one_per_observation,
history_window, phase_reset_policy, value_range
```

Start with visibility/identity confidence, image-relative target/rack displacement,
wrist target area change, visual finger opening/closing cues, and temporal
stability. Frozen-camera detection is only a diagnostic cue: an unchanged image
can be legitimate when physics is paused or the scene is static. Do not classify
it as acquisition failure without an applicable visual/contextual condition.

Reuse `zetta/evolution/critic.py:TemporalCritic` for supported scalar rules:
`lt/le/gt/ge/eq/ne/stagnant`, activation predicates, dwell, and cooldown. Its
existing temporal state advances per evaluate call; the adapter ensures one
evaluation per distinct observation. Add strict operator/threshold type checks
at package validation; dataclass annotations alone do not validate JSON enums.

Keep the shared evaluator unchanged. Use a narrow ARX adapter:

- Optional visual uncertainty must not be represented as zero, NaN, or a
  missing dictionary key. The feature layer returns explicit validity.
- Evaluate only rules whose full required feature set is valid. Reset that
  rule's dwell/history/cooldown on invalid evidence, and emit `unknown` for it.
  Use one existing `TemporalCritic` instance per rule and its public `reset()`
  method for invalid evidence. This preserves valid-rule behavior without
  modifying the shared evaluator or accessing its private state.
- Configuration errors such as an unknown feature fail package preflight.
  Runtime invalidity due to occlusion is a normal perception outcome.
- Phase-state detection belongs in the feature/detector layer; do not force
  arbitrary state machines into opaque prose or an `eval()` expression.

The package feature schema is authoritative for that package. Validate every rule
against it and every runtime output against its names/types; do not accept
undeclared outputs or references. Feature schema hashes are package-specific,
not proof of membership in a preimplemented registry. Never derive the online
schema from private trajectory fields. The harness's frozen input policy still
forbids privileged reads regardless of how a learner names a feature.

### Feature source ABI and registration

```python
class FeatureSource:
    def reset(self, observation, images, config) -> None: ...
    def extract(self, observation, images) -> dict: ...
    def close(self) -> None: ...

class ArxCriticRegistry:
    def register(self, manifest, config, feature_schema, package_root) -> None: ...
    def freeze(self) -> FrozenCritic: ...
    def describe(self) -> dict: ...
```

`images` maps declared camera names to owned read-only uint8 HWC arrays;
`observation` contains only section 2 public metadata. `extract` returns exactly
the declared features as `{name: {valid: bool, value: scalar|null}}`; invalid
values use null and reset only affected rule evaluators. Valid numeric values
must be finite, match declared types/ranges, and cannot be bools. Feature
extraction has no motion or gateway capability. Lifecycle events remain audit-only
in this initial source ABI; reset occurs once per episode.

The learner writes `critic/features.py`, `critic/feature_schema.json`, manifest,
config, and RGB replay tests. Manifest adds
`source: {kind: "package_python", api_version: "arx.features.v1"}` and config
contains `extractor_config`, `rules`, and `rule_failure_modes`. The entrypoint is
relative to `critic/`; resolve paths without escapes/symlinks. `code_sha256` hashes
features.py in the single-file initial contract. Imports are limited to the
pinned runtime's stdlib and numpy; additional package code/dependencies require
an explicitly versioned admission contract. Never pip-install learner requests.

The harness validates artifact hashes, duplicate IDs, schema/rule compatibility,
failure-mode/recovery bindings, and resource budgets before creating workers.
Do NOT import learner code in the gateway process, including during preflight.
Run import/reset/replay tests and live extraction in a sandboxed worker with
only read-only package code, the pinned Python runtime, bounded scratch space,
and an observation IPC channel. No simulator files, project checkout, credentials,
network, environment handle, or gateway socket. Apply OS-enforced filesystem and
network restrictions, memory/CPU/process limits, and the manifest wall deadline.
A plain subprocess or Python import allowlist alone is insufficient. Require a
configured isolation launcher that passes access-denial tests; if unavailable,
fail preflight rather than silently running unsandboxed.

Worker protocol is length-bounded messages `RESET`, `EXTRACT`, `CLOSE`, with
request ID and observation ID echoed in responses. Transport image arrays via
owned buffers/read-only shared memory; cap bytes from configured camera shapes.
Reject stale/out-of-order responses and oversized outputs. Timeout, crash, invalid
output, or denied access blocks further motion and produces critic error feedback.
Terminate the failed feature worker and close the attempt; do not reset its history
and continue the same episode. Known completed physical steps remain known;
a feature-worker failure does not itself make those writes uncertain.

Freeze manifests/config/schema/code hashes into one critic digest before reset.
No registration or code replacement during an episode. The learner submits files;
only the harness can register them. Return firing proposals through the existing
session-core barrier; this adds no new incident lifecycle or unknown-interrupt rule.

## 6. Interruption, uncertainty, and recovery monitoring

Each failure mode declares a machine-readable policy in `recovery_bindings.json`:

```text
failure_mode, skill_entrypoint, allowed_tools[],
max_recovery_steps, max_agent_decisions,
monitor_policy = recovery_local | stop_all_motion,
reentry_policy_id
```

Follow LIBERO's rule-ID suppression and pending-recovery convention, without
adding an incident tracker. On fired proposals, retain their rule IDs and select
a matching recovery binding deterministically by sorted binding ID. If no binding
matches, stop with a controller error. Version one has one active recovery.

Before the first validated recovery motion, acknowledge the pending proposals
and install suppression for the selected recovery's original triggering rule
IDs. Persist this control transition before any write, as the equivalent of
`begin_recovery_step()`. Continue evaluating and logging all rules. Filter the
suppressed rule IDs BEFORE deciding whether to stop the physical-action loop;
LIBERO's post-RPC filtering would be too late for a multi-action local tool.
Never suppress gateway hard bounds or a `stop_all_motion` binding.

A different unsuppressed rule interrupts the current tool and preserves the
active recovery context. Do not switch recovery or expand suppression implicitly.
Version one closes with `RECOVERY_ESCALATION_REQUIRED` after exposing the new
proposal and last observation; nested recovery is deferred. A `stop_all_motion`
proposal permits only read-only review and finish; a validated clearance/reentry
policy may permit a fresh Zeva call. Restore suppression on every recovery exit,
error, cancellation that ends recovery, and terminal closure. Ordinary partial
tool cancellation preserves recovery, not nominal execution permission.

Unknown features are diagnostic in version one; they do not automatically stop
motion or create a reserved incident. This follows the references' explicit
proposal-driven interruption. Activation guards encode rule applicability.
A detector may explicitly fire an observation-quality rule, with its own dwell,
activation conditions, and recovery binding, when evidence warrants stopping.
A `failure` assessment must contain a firing proposal; `clear`/`unknown` must
not contain one. Missing required camera, malformed output, and critic timeout
remain execution errors and block further motion.

### Known limitations / possible bugs retained from the reference approach

- TemporalCritic firing events do not describe persistent failure or clearance.
  Repeated firings and genuinely new occurrences of the same rule cannot be
  distinguished. Rule-ID suppression may hide a recurrence during recovery.
  Bound recovery duration, record every filtered firing, and cover recurrence
  in replay/live tests; do not add incident lifecycle inference in this change.
- The references do not define general perceptual `unknown` semantics. Ignoring
  diagnostic unknowns can miss an occluded failure; turning every unknown into
  an interrupt can prevent normal approach/warmup. Keep explicit guarded rules,
  report unknown rates and missed failures, and revisit only with evidence.
- A false guard/invalid feature resets temporal history; intermittent visibility
  can delay a trigger. Include that case in detector replay validation.

## 7. Reentry assessment

Use a separate `ReentryAssessment` implementation with the same observation-only
input boundary. The gateway invokes it through `arx.review_reentry` using recent
episode frames, available static calibration, and, only if declared by this
reentry profile, predicted command kinematics. It cannot query private state.

```text
schema_version = arx.reentry.assessment.v1
recovery_id, observation_id, policy_id
status = eligible | ineligible | unknown
checks[] = {check_id, status: pass|fail|unknown, evidence_ids[], reason_code}
```

For pickup, first implement configurable visual checks for target identity,
supported target, open fingers, visible approach corridor, stable observation,
and appropriate pregrasp phase. Any learned numerical region is an explicitly
versioned hypothesis requiring live validation, not a guaranteed Zeva initiation
region. An optional model-based check must have a frozen prompt/model and
validated structured response; it does not accept deployment self-certification.

Only the gateway turns `eligible` with all mandatory checks passed into a
single-use reentry token. Ineligible/unknown gives feedback and no token.
An agent's prose assertion, critic cooldown, or completed motion cannot clear
the latch on its own. New motion invalidates the assessment and token.

## 8. Validation and learner-facing contract

The learner receives the exact input/output schemas, feature-source ABI,
implementation ABI, bounded runtime profile, and example allowed observations.
It must deliver deterministic replay fixtures including clear, failure, unknown,
and recovery/reentry sequences. No online dependence on training filenames,
seed IDs, absolute simulator coordinates, or private evaluation labels.

Tests before live deployment:

1. Ordered reset/step/lifecycle stream; duplicate observation deduplication;
   same observation and lifecycle input produces identical outputs on replay;
   reset initializes but does not fire; lifecycle notifications do not evaluate
   the same frame twice.
2. Identical public pixels with altered hidden object/contact state produces
   identical assessments. Separately test actual process/file/network isolation.
3. Occlusion, wrong identity, camera mismatch, nonfinite feature values, missing
   required frames, and exhausted history yield the specified unknown/error.
4. Interrupt before next step, including after final chunk action; no stale
   assessment authorizes motion; timeout leaves physics paused.
5. Trigger-rule acknowledgement allows bounded recovery while other rules still
   interrupt; suppression is restored on all exit paths; reentry cannot reuse stale tokens.
6. Detector replay measures failure coverage, onset/detection delay, success-control
   false interrupts, unknown rate, and remaining recovery horizon. Offline labels
   may use simulator truth; detector inputs may not.

Replay assesses detection, not counterfactual recovery success. Live fresh-context
trials are mandatory for recovery effectiveness and promotion.

## 9. Example learner candidate and functional registration test

Draft this complete payload fixture during implementation at
`tests/fixtures/arx_candidate_rgb_v1/`. These are proposed files, not installed
artifacts. Its purpose is registration/execution/observation handoff, NOT a useful
physical-failure detector. It uses one clean feature: mean front-camera red-channel
intensity in [0,1]. The learner writes its implementation, declaration, and rule.

```text
manifest.json                      generated/sealed package manifest
critic/manifest.json
critic/features.py
critic/feature_schema.json
critic/config.json
skill/SKILL.md
recovery_bindings.json
reentry/manifest.json
reentry/config.json
evidence/claims.json
tests/replay_cases.json
```

`critic/features.py` (actual proposed learner-authored code):

```python
import numpy as np

class FeatureExtractor:
    def reset(self, observation, images, config):
        if config:
            raise ValueError("this fixture accepts no extractor config")

    def extract(self, observation, images):
        image = images["front_rgb"]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected uint8 HWC RGB")
        value = float(image[:, :, 0].mean(dtype=np.float64) / 255.0)
        return {"visual.mean_red": {"valid": True, "value": value}}

    def close(self):
        pass
```

`critic/feature_schema.json`:

```json
{
  "schema_version": "arx.features.schema.v1",
  "features": [{
    "name": "visual.mean_red", "scalar_type": "number", "units": "fraction",
    "source_cameras": ["front_rgb"], "extractor_version": "mean-red-v1",
    "validity_condition": "configured uint8 RGB frame available",
    "sampling": "one_per_observation", "history_window": 1,
    "phase_reset_policy": "episode_reset_only", "value_range": [0.0, 1.0]
  }]
}
```

`critic/manifest.json` (replace hash placeholders when sealing):

```json
{
  "schema_version": "arx.critic.manifest.v1",
  "critic_id": "mean-red-monitor",
  "input_profile": "arx.rgb_only.v1",
  "implementation": "feature_rules",
  "source": {"kind": "package_python", "api_version": "arx.features.v1"},
  "entrypoint": "features.py:FeatureExtractor",
  "code_sha256": "<computed>", "config_sha256": "<computed>",
  "feature_schema_sha256": "<computed>",
  "cameras": ["front_rgb"], "history_limit": 2,
  "evaluation_timeout_ms": 5000,
  "failure_modes": ["smoke_red_high"],
  "recovery_bindings": {"smoke_red_high": "smoke-stop"}
}
```

`critic/config.json`:

```json
{
  "schema_version": "arx.critic.config.v1",
  "extractor_config": {},
  "rule_failure_modes": {"red-high": "smoke_red_high"},
  "rules": [{
    "rule_id": "red-high", "title": "Functional RGB threshold trigger",
    "feature": "visual.mean_red", "operator": "gt", "threshold": 0.5,
    "dwell_steps": 2, "cooldown_steps": 0, "proposal": "interrupt",
    "activation_conditions": [], "evidence_ids": ["fixture-red-sequence"]
  }]
}
```

`recovery_bindings.json` declares one `smoke-stop` binding for `smoke_red_high`,
`skill_entrypoint="smoke-stop"`, `allowed_tools=["arx.finish"]`,
`max_recovery_steps=0`, `max_agent_decisions=1`,
`monitor_policy="stop_all_motion"`, `reentry_policy_id="smoke-never-resume"`.
For this fixture, admit zero motion budget for stop-only bindings. Reentry
manifest/config binds a reviewed built-in `always_ineligible` assessment policy;
it issues no token and adds no learned behavior. It is test-only, not a shortcut
for production recovery. The skill contents are:

```text
# Smoke stop
This is a functional interface test, not a grasp-failure detector.
When red-high fires, inspect the supplied current front RGB. Select arx.finish
with reason "RGB critic functional test completed" using the current event,
observation and epoch. Do not move, resume Zeva, query private state, or assert
physical task success. Return the fixed ARX decision JSON format.
```

`evidence/claims.json` identifies the synthetic sequence as test evidence, never
as learned task effectiveness. `tests/replay_cases.json` describes three 320x240
RGB sequences: reset+low/low (no fire); reset+low/high/high/low (fire at step 3);
reset+high/low/high (no fire). Low is RGB [0,0,0], high is [255,0,0]. Other
required policy cameras may remain black. The fixture builder creates PNGs,
content IDs and their hashes in the test evidence store; replay metadata is
not mounted as online critic input. A separate malformed-frame case expects
`CRITIC_EXECUTION_ERROR`, not an unknown success or implicit acceptance.

Seal `manifest.json` using the learning plan's complete package schema: package ID
`arx-candidate-rgb-v1`, generation 0, null parent, changed components critic/skill/
bindings, fixture-only hypothesis/validation plan, contract/catalog/bootstrap
hashes, package `feature_schema_sha256`, and every payload file's path/hash/media
type. Do not omit `files` or allow unresolved hashes. No preexisting extractor ID
is needed: the manifest binds the code file submitted by the learner.

Required end-to-end test:

1. Load/seal package and run the real code in the isolation worker. Assert the
   values on low/high images are 0/1 and code has no access to private mounts.
2. Register and freeze through the production package loader/critic interface.
   Reset initializes without firing. A fake backend executes at least four
   pending targets and returns low/high/high/low post-step images.
3. Assert exactly three targets execute; the real extractor and TemporalCritic
   fire `red-high` at step 3 and the fourth target is discarded. Do not mock
   feature outputs in this integration test.
4. Assert the interrupted operation/event contains `obs-3`, its front PNG ID/hash,
   the rule/failure mode, actual executed count, epoch and stop-only recovery
   context. Retrieve that PNG through the public image endpoint and verify bytes.
5. With no agent attached, poll observation/status and assert no further physics
   advances. Then use a fake fresh-decision adapter returning `arx.finish`;
   validate/persist the decision and close. No real LLM is required for this test.
6. Reject modified code/config hashes, undeclared feature output, wrong scalar
   types, path escapes, missing bindings, worker timeout and forbidden reads.
   Critic errors block further motion and preserve known executed-step records.

After this deterministic test, use the same loader and gateway with real ARX
MuJoCo and the offline-test policy targets. The 0.5 threshold may not fire on the
real scene; report no-fire honestly. For a guaranteed functional trigger, seal a
SEPARATE explicitly test-only package with `operator="ge", threshold=0.0` and
unchanged dwell=2: every valid RGB frame qualifies, so it fires after two
nonterminal steps. This still executes learner code on actual RGB and uses no
step-count/privileged feature. It must never enter task-effectiveness gates or
promotion. Verify the returned observation is the actual second post-step render.
A prior terminal event takes precedence and invalidates that trigger expectation.

## 10. Future PRM critic extension (not implemented in this change)

Keep feature production separate from rule evaluation and interruption. A future
`FeatureSource` kind `prm_service` accepts the immutable task instruction, current
RGB, and configured target RGB and returns `{score, valid, model_version}` tied
to observation/request IDs. It does not require a learning agent or package-local
Python. A hand-authored frozen critic manifest/config can select it through the
same registration path. Initial code only defines the source interface and rejects
unimplemented kinds; do not add a server, network client, or PRM model now.

The future trusted service adapter runs outside the learner-code sandbox. Its
endpoint/credentials are harness-owned; bind model/version, prompt/preprocessing,
score direction/range, selected cameras, target-image content hashes and task
identity in the manifest. Target images must be approved public goal references,
not future frames from the same evaluation episode or simulator pose renderings
constructed using hidden target state. This requires an explicit new input profile
`arx.prm_inputs.v1`; do not silently expand `arx.rgb_only.v1`.

A configurable score-history adapter can then expose `progress.net_drop` and
`progress.consecutive_drops`. For higher-is-better scores s[t], define a meaningful
drop as s[t-1]-s[t] > epsilon. Fire only after k consecutive meaningful drops AND
s[t-k]-s[t] >= delta. Freeze epsilon, k, delta, score scale and sampling interval;
use existing guards/dwell/cooldown or a dedicated reviewed temporal adapter. A
lower-is-better model must explicitly normalize its output first. Warmup or
invalid scores do not synthesize zeros or failures; reset the affected history
and report diagnostic unknown. Transport/timeout/schema failures are execution
errors. Repeated reads of one observation never extend the score history.

Future acceptance tests: rising/flat/noisy scores do not fire; sustained large
decline does; tiny declines and insufficient history do not; bad/stale responses
block correctly; target/model hashes bind replay evidence. Model calls may not
be deterministic, so archive request/response scores for deterministic temporal
replay rather than promising identical re-inference. Sampling and latency must
preserve the gateway barrier: no unchecked actions run while an assessment is
pending. PRM events use the SAME assessment/proposal/recovery interface, without
changes to tool execution, fresh-decision invocation, or incident semantics.

## 11. Reference compatibility and implementation boundary

Use `robots/libero/env_server.py:critic_chunk_step` and
`robots/robocasa/session_core.py:execute_chunk` for post-step evaluation and
executed-prefix accounting. Use LIBERO `configure_critic`,
`begin_recovery_step`, and `suppress_recovery_rules` for freeze/acknowledgement
semantics. Do not copy their privileged feature extraction. Do not change
`zetta/evolution/critic.py`, `rollout_runtime/backends/libero_critic.py`, or either
existing backend to add ARX behavior. Put adapters and visual extraction under
`robots/arx/`; use existing public evaluator methods and add ARX-specific tests.
The robot-specific reentry check is an additive policy, not a change to the
shared `RecoveryController` or an inferred incident-clearance algorithm.
