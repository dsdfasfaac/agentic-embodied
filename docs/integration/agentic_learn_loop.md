# loop1

# Loop 1: Pure-VLA Rollout Collection with GPU Routing

Atomic 要用这个model:/mnt/mingzhe/Code/robocasa365/Isaac-GR00T/outputs/robocasa_atomic_seen_8gpu/checkpoint-180000

Evaluate the frozen VLA policy on seeds 100–149, one valid rollout per seed.

Composite 要使用xiaomi的官方推理代码和模型，/mnt/mingzhe/Code/Xiamorobotics

以及horizon要和robocasa365的标准对齐，而不是默认的1k

## Mandatory scheduler entry point

Before doing anything else, locate and use the repository-provided scheduler:

```
/Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/rollout_scheduler.py
```

This scheduler is the mandatory entry point for the batch. Do not launch
`/Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/run_pure_vla_loop1.py`
directly through local execution or raw SSH. Every Loop 1 task batch must be
submitted as the command payload of the scheduler:

```
python3 /Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/rollout_scheduler.py run [scheduler arguments] -- [frozen batch command]
```

The first execution action must be a scheduler status probe:

```bash
python3 /Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/rollout_scheduler.py status
```

Then use the scheduler's `run` operation to select a free node/GPU and launch
the complete frozen 50-rollout batch. Always pass `--route-record` and place
the resulting JSON beside the batch artifacts.

If the scheduler file is missing, unreadable, not executable through Python,
or cannot probe the node pool, report an infrastructure blocker. Do not bypass
the scheduler with raw SSH, direct local execution, a manually selected node,
or a replacement scheduling implementation.

## Scheduler constraints

GPU scheduling is an outer infrastructure step only. It must not enter the
policy runtime or alter the policy prompt, policy weights, simulator
configuration, seed list, rollout horizon, or evaluation logic.

Before starting a task batch:

1. Use
`/Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/rollout_scheduler.py status`
to inspect the default pool:
`4090_wuwen`, `4090_wuwen2`, `4090_wuwen3`, `4090_wuwen4`,
`4090_wuwen5`, `4090_wuwen6`, `4090_wuwen7`, and `4090_wuwen8`.
2. Treat a GPU as available only when the scheduler reports it below the
frozen memory and utilization thresholds. Unreachable nodes and GPUs with
unknown metrics are not available.
3. Submit one complete 50-rollout task batch through
`/Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/rollout_scheduler.py run`.
If several task batches are requested, submit each task as a separate
scheduler request so they can occupy different free nodes.
4. Once routed, keep that task batch on the selected node/GPU. Do not migrate
a valid in-progress rollout merely because a different GPU becomes idle.
5. Save the scheduler route record beside the batch artifacts. The record must
include the selected node, GPU index, probe snapshot, exact remote command,
scheduler version, and return code.
6. If no GPU is free, wait and probe again. Do not replace the task or seed,
relax the frozen protocol, or invoke a planner/recovery path.
7. A scheduler, SSH, node, or service-start failure is an infrastructure
failure. Preserve any artifacts already produced and resubmit the exact
same task/seed request. Never substitute another seed.

Mandatory launch form (replace paths and endpoints with the deployment's
frozen values, but do not remove the scheduler wrapper):

```bash
python3 /Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/rollout_scheduler.py run \
  --label CloseToasterOvenDoor-loop1 \
  --wait-s 1200 \
  --route-record /artifact/root/scheduler_route.json \
  -- \
  python3 /ABSOLUTE/PATH/ON/THE/REMOTE/NODE/scripts/evolution/run_pure_vla_loop1.py \
  --repo-root /remote/repo \
  --robocasa-root /remote/robocasa \
  --env-endpoint <http://127.0.0.1:18766> \
  --vla-endpoint <http://127.0.0.1:18080> \
  --output-root /artifact/root
```

Use only the VLA policy during each rollout. Do not invoke an Agent, tool,
planner, runtime critic, recovery, CAP, or manually designed fallback. The GPU
scheduler is not a policy tool: it only selects the machine before the frozen
rollout command starts.

Path rule: the scheduler command above is executed on the controller machine
and therefore uses the absolute local path
`/Users/huangmingzhe/Documents/ChatGPT/atomic/work/close_toaster_worktree/scripts/evolution/rollout_scheduler.py`.
The command after the scheduler's `--` is executed on the selected remote
node, so `/ABSOLUTE/PATH/ON/THE/REMOTE/NODE` must be replaced with the actual
absolute checkout path on that node. Do not use a relative path for either
script. The `--repo-root`, `--robocasa-root`, `--env-endpoint`,
`--vla-endpoint`, and `--output-root` values must also be absolute or fully
qualified deployment paths appropriate for the selected remote node.

Do not modify the policy, policy prompt, simulator configuration, or evaluation
logic during the batch.

For every seed, save:

- seed and initial state;
- policy, prompt, and simulator versions;
- success, reward, and final task progress;
- completed milestones and first missing milestone;
- state and action trajectory;
- object-relative simulator state;
- keyframes and rollout video;
- collision, contact, and grasp signals when available;
- termination reason;
- inference, simulator, and infrastructure errors.

Preserve both successful and failed trajectories. Do not overwrite or delete
failed runs.

If a rollout is invalid because of infrastructure, simulator, logging, or model
service failure:

1. preserve the invalid attempt;
2. record the failure reason;
3. rerun the same seed;
4. do not replace it with another seed.

After all 50 valid rollouts:

1. Report overall SR, reward, final-progress, and milestone-completion
statistics.
2. Index successful trajectories as nominal VLA references using:
    - task and milestone;
    - policy version;
    - object-relative state;
    - local trajectory window.
3. Produce a failure manifest containing every failed seed and its artifact
paths.
4. Do not cluster failures, infer root causes, select repair candidates, or
modify the VLA.

Output:

- frozen batch configuration;
- scheduler route record;
- per-seed result table;
- aggregate SR and milestone statistics;
- artifact paths for every rollout;
- successful-trajectory reference index;
- failed-seed manifest for Loop 2.

Loop 1 is complete only when all seeds 100–149 have valid saved rollouts, all
artifacts are readable, and the scheduler route record is readable.

Loop 2 Stage 1 is responsible for clustering the failed seeds, selecting
representative failures, diagnosing causal mechanisms, and producing repair
candidates.

# LOOP2 STAGE1

Loop-2 Stage 1: Failure Clustering and Causal Diagnosis

You are a failure-clustering and causal-diagnosis agent, not a patch generator.

Your goal is to:

1. cluster the failed seeds produced by Loop 1;
2. select representative failures;
3. identify the earliest causal mechanism for each major cluster; and
4. produce testable repair candidates for Stage 2.

Do not implement or validate repairs in this stage.

Inputs:

- frozen Loop-1 batch manifest;
- failed-seed manifest and rollout artifacts;
- successful-trajectory reference index; and
- milestone and reference-progress definition.

Version control:

- Do not mix rollouts from different policy, Skill, prompt, tool, critic, or
simulator versions unless explicitly analyzing a version difference.
- For every conclusion, identify the artifact and version that support it.

Single-view video policy:

- When a video contains multiple camera views, select one primary view before
inspection. Prefer the view that most clearly shows the active end-effector,
manipulated object, and relevant contact region; otherwise use the default
or first available view.
- Use only this same primary view for the entire analysis of a rollout. Do not
switch between camera views, combine evidence from different views, or ask
for multi-view agreement.
- Use the same view-selection rule for failed and successful reference videos
whenever comparison is needed.
- If the selected view is missing, corrupted, occluded, or insufficient to
support a claim, report the limitation explicitly and rely on simulator
state, trajectory data, logs, and critic outputs. Do not silently switch to
another view.
- Video is supporting evidence only. It does not replace simulator state,
trajectory data, or critic outputs.

Failure clustering:

Before diagnosing individual failures, build a structured record for every
failed seed containing:

- first missing milestone;
- first observable divergence;
- task phase;
- active tool, if present;
- runtime-critic state, if present;
- object-relative simulator state;
- contact, collision, grasp, and stability signals;
- proposal and Agent-decision pattern, if present;
- tool-switch history, if present;
- local trajectory behavior; and
- visible failure signature from the selected-view keyframes or downsampled
video.

Cluster failures primarily by:

- first missing milestone;
- first observable divergence;
- execution phase and active tool;
- object-relative state near the divergence;
- runtime-critic signature;
- contact, collision, or grasp signature;
- proposal, Agent-decision, and tool-switch pattern; and
- local trajectory behavior.

Do not cluster failures solely by:

- final reward;
- final image;
- seed number;
- absolute world coordinates;
- final failure description; or
- an unvalidated causal explanation.

Assign each failure a primary cluster based on its earliest divergence. Record
secondary tags when it matches additional signatures. Keep unexplained or
low-similarity failures as explicit outliers instead of forcing them into an
existing cluster.

For each major cluster, select:

- one medoid seed representing the typical failure; and
- one boundary seed when the cluster contains meaningful variation.

For pure-VLA rollouts, mark critic, proposal, Agent, and tool fields as absent.
Do not fabricate missing evidence.

Diagnosis order:

Diagnose the representative seed of each cluster in this order:

1. Check simulation success, reward, and final task progress.
2. Compare actual progress with the reference progress and identify the first
missing milestone.
3. Explicitly inspect the runtime critic.
4. Retrieve matched successful VLA references using task and milestone, policy
or Skill version, object-relative simulator state, critic state if
available, and active tool if available.
5. Compare failed and successful runs around the first divergence using
structured state, critic outputs, tool history, proposals and Agent
decisions, keyframes, and local trajectory windows.
6. Inspect the selected primary video view by default at one frame every
0.5 seconds. Use it only to check phase transitions, contact and collision,
grasp acquisition or loss, object motion, and visible task progress.
7. Inspect a short high-rate or time-aligned clip from this same view only when
the previous evidence is insufficient or indicates perception/contact
ambiguity.

Successful-reference use:

Successful VLA trajectories are nominal references, not recovery
demonstrations. Use them to identify:

- where the failed run leaves the successful state distribution;
- the expected state, critic, and milestone transitions; and
- a possible state where the nominal VLA policy can resume.

Do not copy actions from a successful trajectory into an unmatched failure
state. If no sufficiently similar successful reference exists, report
reference-unavailable instead of using an irrelevant trajectory.

Runtime-critic check:

Always report whether the rollout has a runtime critic. If present, verify its
phase, collision, contact, grasp-stability, deviation, and interruption
outputs. Compare each output against simulator state and classify it as
correct, false-positive, false-negative, delayed, or unstable.

If the critic is absent, determine whether missing online detection
contributed to the failure. Propose adding a critic only when supported by
evidence.

Causal diagnosis:

- Separate observed symptoms from inferred causes.
- When the cause is ambiguous, compare at least two falsifiable hypotheses.
- Perform at least one check that distinguishes the leading hypotheses.
- Use tools, logs, simulator state, critic outputs, matched-success references,
trajectory data, and the selected-view video as needed.
- Every system claim must reference observed evidence.
- Do not treat parameter correlation as causal evidence.
- Do not assume that the phase where the task finally fails is the phase that
caused the failure.

Identify the earliest incorrect layer:

- evaluation or progress;
- runtime critic;
- state representation;
- planning or control;
- recovery; or
- parameter, only as a last resort.

Cluster consistency check:

After diagnosing the representative seed:

1. Check whether the same first divergence, causal signal, and predicted
mechanism appear in the other seeds assigned to the cluster.
2. If the mechanism explains the cluster members, keep the cluster and
produce one shared repair candidate.
3. If different members have different earliest mechanisms, split the cluster
and diagnose the new representatives separately.
4. Do not assume that visually similar final failures share the same cause.

Output for the complete failure set:

- cluster IDs and observable definitions;
- seed list and frequency for each cluster;
- representative and boundary seeds;
- outlier seeds; and
- evidence and artifact paths.

Output for each diagnosed cluster:

- first missing milestone;
- first divergence;
- matched-success comparison or reference-unavailable status;
- runtime-critic verdict;
- selected-view downsampled-video evidence summary, including the view used;
- root-cause layer and observed evidence;
- competing hypotheses and discriminating check;
- testable repair candidates;
- expected critic, state, progress, and trajectory changes;
- proposed nominal VLA re-entry state, if recovery is required; and
- cluster-consistency result.

Stage 2 receives one repair candidate per validated failure mechanism, not one
independent repair per seed.

Stage 2 implements the candidate and validates it first on the representative
seed, then on the remaining seeds in the same cluster using same-seed,
closed-loop trajectory replay.

# Loop-2 Stage 2: Critic-Guided Harness Repair

You are a harness repair and validation agent.

The VLA remains the default policy for normal execution. The harness supplements
the VLA by detecting execution failures and enabling evidence-driven recovery.
Implement the minimal repair identified by Stage 1 and validate it on the
original failure.

Repair order:

1. If Stage 1 identifies missing, incorrect, delayed, or unstable online
detection, create or modify the runtime critic.
2. The runtime critic runs during execution and continuously checks:
    - task phase and milestone progress;
    - object presence and motion;
    - contact and collision;
    - grasp acquisition and stability;
    - trajectory deviation;
    - conditions requiring recovery or interruption.
3. The critic must report evidence and emit structured proposals. It must not
select actions, switch tools, recover, terminate, or choose fallbacks.
4. Define recovery playbooks for the validated failure mechanism. Each playbook
must specify:
    - triggering evidence;
    - preconditions;
    - candidate recovery actions;
    - expected success signal;
    - failure or escalation condition;
    - nominal VLA re-entry state.

    Example playbooks:

    - object lost or grasp unstable → retreat, regrasp, or restage;
    - target progress stalled → inspect state, retry, restage, or regenerate plan;
    - collision or blocked motion → stop the current attempt and propose retreat,
    replanning, or restaging.

    Playbooks are references for the Agent, not autonomous fallbacks.


Proposal and decision protocol:

The multimodal Role1 Agent is the only component allowed to approve or modify:

- tool switching;
- recovery or restaging;Loop-2 Stage 2: Critic-Guided Harness Repair

You are a harness repair and validation agent.

The VLA remains the default policy for normal execution. The harness supplements
the VLA by detecting execution failures and enabling evidence-driven recovery.
Implement the minimal repair identified by Stage 1 and validate it on the
original failure.

Repair order:

1. If Stage 1 identifies missing, incorrect, delayed, or unstable online
detection, create or modify the runtime critic.
2. The runtime critic runs during execution and continuously checks:
    - task phase and milestone progress;
    - object presence and motion;
    - contact and collision;
    - grasp acquisition and stability;
    - trajectory deviation;
    - conditions requiring recovery or interruption.
3. The critic must report evidence and emit structured proposals. It must not
select actions, switch tools, recover, terminate, or choose fallbacks.
4. Define recovery playbooks for the validated failure mechanism. Each playbook
must specify:
    - triggering evidence;
    - preconditions;
    - candidate recovery actions;
    - expected success signal;
    - failure or escalation condition;
    - nominal VLA re-entry state.

    Example playbooks:

    - object lost or grasp unstable → retreat, regrasp, or restage;
    - target progress stalled → inspect state, retry, restage, or regenerate plan;
    - collision or blocked motion → stop the current attempt and propose retreat,
    replanning, or restaging.

    Playbooks are references for the Agent, not autonomous fallbacks.


Proposal and decision protocol:

The multimodal Role1 Agent is the only component allowed to approve or modify:

- tool switching;
- recovery or restaging;
- plan regeneration;
- replacement of a rejected action;
- interruption or termination;
- returning control from recovery to the VLA.

Critics, monitors, planners, tools, and safety shields may only emit proposals.
A safety shield may block an unsafe action, but must return the rejected action
and rejection reasons as a proposal.

The runtime critic may run at high frequency, but the Agent is event-driven.
Invoke the Agent only for a new or materially changed proposal. Consolidate
repeated equivalent proposals.

Each proposal must include:

- current visual observation;
- simulator and task state;
- task progress and milestone state;
- runtime-critic evidence;
- recent trajectory and action history;
- active tool and tool-switch history;
- relevant recovery playbooks;
- proposed change and supporting evidence.

Recovery and VLA re-entry:

Recovery should return control to the VLA at the earliest validated nominal
re-entry state, not after a fixed number of steps or a fixed duration.

A valid re-entry state must satisfy:

- the original failure condition has cleared;
- the recovery post-condition is satisfied;
- robot, object, and contact states are stable;
- the current task phase is known;
- the VLA initiation condition is satisfied;
- the state is consistent with successful VLA references when available;
- the same failure is not expected to immediately recur.

When these conditions are satisfied, the critic emits a
resume-VLA proposal. Only the Agent may accept it.

Do not hand control back during unstable contact, object motion, incomplete
grasp acquisition, or an unvalidated state outside the VLA operating
distribution.

If no safe VLA re-entry exists, recovery may complete the current milestone.
Recovery may complete the entire task only when explicitly approved by the Agent.
If this happens repeatedly, treat the recovery as a possible new primitive or
as evidence of insufficient VLA coverage.

Validation:

1. Reproduce the original failure using the same seed and initial state.
2. If the critic was added or modified, replay the original trajectory through
the new critic and verify the predicted signal.
3. Rerun the repaired Skill in closed loop from the same state or from before
the first divergence.
4. Compare original, repaired, and reference runs using:
    - task progress and milestones;
    - runtime-critic outputs;
    - proposals and Agent decisions;
    - tool switches;
    - recovery behavior;
    - VLA re-entry state;
    - trajectory behavior;
    - final result.
5. Verify that the repaired VLA resumes nominal progress after handoff and that
the original proposal does not immediately recur.
6. Do not validate a critic, policy, controller, or recovery using only old
recorded actions.

Stage 2 passes only when:

- the original failure is fixed;
- the predicted critic signal changes as expected;
- recovery reaches a validated re-entry state;
- the Agent explicitly makes every recovery, tool-switch, fallback,
termination, and resume-VLA decision;
- the repaired closed-loop trajectory reaches the expected task milestone.

Otherwise, return to Stage 1 and revise the diagnosis or repair.

- plan regeneration;
- replacement of a rejected action;
- interruption or termination;
- returning control from recovery to the VLA.

Critics, monitors, planners, tools, and safety shields may only emit proposals.
A safety shield may block an unsafe action, but must return the rejected action
and rejection reasons as a proposal.

The runtime critic may run at high frequency, but the Agent is event-driven.
Invoke the Agent only for a new or materially changed proposal. Consolidate
repeated equivalent proposals.

Each proposal must include:

- current visual observation;
- simulator and task state;
- task progress and milestone state;
- runtime-critic evidence;
- recent trajectory and action history;
- active tool and tool-switch history;
- relevant recovery playbooks;
- proposed change and supporting evidence.

Recovery and VLA re-entry:

Recovery should return control to the VLA at the earliest validated nominal
re-entry state, not after a fixed number of steps or a fixed duration.

A valid re-entry state must satisfy:

- the original failure condition has cleared;
- the recovery post-condition is satisfied;
- robot, object, and contact states are stable;
- the current task phase is known;
- the VLA initiation condition is satisfied;
- the state is consistent with successful VLA references when available;
- the same failure is not expected to immediately recur.

When these conditions are satisfied, the critic emits a
resume-VLA proposal. Only the Agent may accept it.

Do not hand control back during unstable contact, object motion, incomplete
grasp acquisition, or an unvalidated state outside the VLA operating
distribution.

If no safe VLA re-entry exists, recovery may complete the current milestone.
Recovery may complete the entire task only when explicitly approved by the Agent.
If this happens repeatedly, treat the recovery as a possible new primitive or
as evidence of insufficient VLA coverage.

Validation:

1. Reproduce the original failure using the same seed and initial state.
2. If the critic was added or modified, replay the original trajectory through
the new critic and verify the predicted signal.
3. Rerun the repaired Skill in closed loop from the same state or from before
the first divergence.
4. Compare original, repaired, and reference runs using:
    - task progress and milestones;
    - runtime-critic outputs;
    - proposals and Agent decisions;
    - tool switches;
    - recovery behavior;
    - VLA re-entry state;
    - trajectory behavior;
    - final result.
5. Verify that the repaired VLA resumes nominal progress after handoff and that
the original proposal does not immediately recur.
6. Do not validate a critic, policy, controller, or recovery using only old
recorded actions.

Stage 2 passes only when:

- the original failure is fixed;
- the predicted critic signal changes as expected;
- recovery reaches a validated re-entry state;
- the Agent explicitly makes every recovery, tool-switch, fallback,
termination, and resume-VLA decision;
- the repaired closed-loop trajectory reaches the expected task milestone.

Otherwise, return to Stage 1 and revise the diagnosis or repair.

Loop-3: Repair Generalization, Merge, and Skill Packaging

You are the Loop-3 generalization and skill-packaging agent.

Input:

- locally validated Stage-2 repairs for N historical failure seeds;
- the original and repaired rollout evidence for those seeds;
- the frozen pre-merge Skill;
- held-out seeds 0–9.

Your goal is to consolidate the seed-level repairs into the smallest
mechanism-level repair, validate it on all historical failure seeds and held-out
seeds 0–9, and package the result as a complete executable Skill.

Do not simply concatenate seed-specific patches.

Repair consolidation:

1. Compare the Stage-2 repairs and identify:
    - shared failure mechanism;
    - shared runtime-critic signals;
    - common recovery requirements;
    - compatible and conflicting tool, parameter, and planning changes.
2. Generalize repairs using task-state invariants rather than seed IDs, absolute
poses, fixed step counts, timeouts, or unexplained thresholds.
3. Preserve the VLA as the default policy. Use the harness only for:
    - runtime failure detection;
    - evidence and proposal generation;
    - Agent-approved recovery;
    - restaging and return to a validated VLA re-entry state.
4. Runtime critics, tools, monitors, planners, and safety shields may only emit
evidence or proposals. The multimodal Role1 Agent must explicitly approve
tool switching, recovery, fallback, replanning, VLA re-entry, or termination.
5. Resolve conflicting repairs through replay or ablation. Prefer the smallest
repair that explains and fixes all validated failures without reducing the
valid operating range of the Skill.
6. Keep numeric parameters only when they are derived from a stable invariant
and supported by sensitivity analysis. Do not preserve seed-specific values.

Skill packaging:

Create the final Skill under `loop3/` with:

- `SKILL.md`
- execution workflow;
- task phases and milestones;
- runtime-critic behavior;
- proposal and Agent-decision protocol;
- tool selection rules;
- recovery and VLA re-entry logic;
- validation and artifact requirements.
- `tools/`
- executable tools and runtime critics;
- clear inputs, outputs, preconditions, post-conditions, and failure signals;
- no autonomous tool switching or fallback behavior.
- `params/`
- justified configuration and parameter defaults;
- invariant and valid operating range for every numeric value;
- no seed-specific parameters.
- `plans/`
- nominal execution plan;
- recovery playbooks;
- triggering evidence, preconditions, candidate actions, expected success

```
signals, escalation conditions, and VLA re-entry states.
```

The Skill must be self-contained so that another Agent can execute it without
access to the original Loop-2 conversation.

Validation:

A. Historical failure seeds

1. Use all N seeds repaired in Stage 2.
2. Reproduce the original failures using the frozen pre-merge Skill.
3. Run the merged Skill in closed loop from the same initial states.
4. Confirm that every historical failure is repaired.
5. Compare progress, critic outputs, proposals, Agent decisions, tool switches,
recovery behavior, VLA re-entry, trajectory, and final result.

B. Critic replay

If a runtime critic was added or modified:

1. Replay the original trajectories through the merged critic.
2. Confirm that the expected failure signal is detected correctly.
3. Check for false positives, false negatives, delayed signals, unstable
proposals, and repeated equivalent proposals.

Critic replay is diagnostic evidence only. It does not replace closed-loop
rollout validation.

C. Held-out generalization

1. Freeze the merged Skill before inspecting held-out results.
2. Run full closed-loop rollouts on seeds 0–9 from reset.
3. Do not add seed-specific fixes, parameters, or exceptions during evaluation.
4. Record success, reward, milestones, first divergence, critic outputs,
proposals, Agent decisions, tool switches, recovery, trajectory, video, and
final result.
5. Compare with the frozen pre-merge Skill on seeds 0–9 when baseline results
are available.

Seeds 0–9 are evaluation-only. If their results are used to modify the Skill,
they become development seeds and a new unseen held-out set must be selected for
final validation.

Loop-3 pass criteria:

The merged repair passes only when:

- all N historical failure seeds succeed in closed-loop replay;
- the predicted critic and recovery signals change as expected;
- every recovery, tool switch, fallback, VLA re-entry, and termination decision

is made explicitly by the Agent;
• the same failure does not immediately recur after VLA re-entry;

- seeds 0–9 show no systematic regression relative to the frozen baseline;
- the repair does not depend on seed IDs or unexplained hard-coded values;
- another Agent can execute the packaged Skill directly.

If a historical failure remains, return the corresponding repair to Loop 2.

If a new mechanism appears on seeds 0–9, preserve the current Skill and evidence,
send the new failure to Loop 2, and use a new held-out set after any subsequent
modification.

Output:

- complete `loop3/SKILL.md`;
- merged `tools/`, `params/`, and `plans/`;
- historical-seed replay report;
- held-out seeds 0–9 rollout report;
- before/after comparison;
- remaining limitations and validated operating range;
- final versioned Skill only after all pass criteria are satisfied.
