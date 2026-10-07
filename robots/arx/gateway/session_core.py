# Copyright (c) 2026 Zetta Contributors
"""Transport-independent single-owner execution and observation barrier."""

from __future__ import annotations

import threading
import time
import uuid
import dataclasses
from copy import deepcopy

import numpy as np

from .contracts import (
    Assessment,
    GatewayError,
    ReentryAssessment,
    ToolRequest,
    ToolResult,
    digest,
)
from .public import ImageStore
from .tools import ApprovedToolContext


class BaselineMonitor:
    """Harness-selected no-intervention mode; never an agent tool."""

    def reset(self, observation, images):
        pass

    def observe(self, observation, images):
        return Assessment(
            critic_id="baseline",
            observation_id=observation["observation_id"],
            step_index=observation["step_index"],
            status="clear",
        )

    def lifecycle(self, event):
        pass


class ArxSessionCore:
    def __init__(
        self,
        *,
        episode_id,
        backend,
        registry,
        journal,
        output,
        limits,
        critic,
        bindings=(),
        package_sha256="baseline",
        cancel_requested=lambda: False,
        phase_changed=lambda phase: None,
        privileged=False,
        programs=None,
    ):
        self.episode_id, self.backend, self.registry = episode_id, backend, registry
        self.journal, self.limits = journal, limits
        self.images = ImageStore(output / "public" / "images")
        self.critic, self.bindings, self.package_sha256 = (
            critic,
            tuple(bindings),
            package_sha256,
        )
        self.cancel_requested, self.phase_changed = cancel_requested, phase_changed
        self.privileged = bool(privileged)
        self.programs = programs or {}
        self.program_cursor = 0
        self.program_token = None
        self.program_outputs = {}
        self.read_samples = 0
        self.state, self.epoch, self.step_index = "READY", 0, 0
        self.decisions, self.incidents = 0, 0
        self.recovery, self.binding, self.token = None, None, None
        self.suppressed = set()
        self.observations, self.frame_history = {}, {}
        self.assessment, self.current, self.commit = None, None, None
        self.closed = False
        self.owner = threading.get_ident()
        self.active = False
        self.sequence = 0
        for binding in self.bindings:
            for name in binding.allowed_tools:
                self.registry.resolve(name)

    def _owned(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("session core is executor-owned")

    def _record(self, kind, value, *, public=False):
        self.sequence = self.journal.record(kind, value, public=public)
        return self.sequence

    def _lifecycle(self, name):
        event = {
            "kind": name,
            "observation_id": self.current["observation_id"],
            "control_epoch": self.epoch,
        }
        self._record("critic_lifecycle", event, public=True)
        self.critic.lifecycle(deepcopy(event))

    def reset(self):
        self._owned()
        if self.current is not None or self.journal.snapshot() is not None:
            raise RuntimeError("episodes cannot be reset or resumed")
        self.phase_changed("reset")
        if hasattr(self.backend, "set_event_sink"):
            self.backend.set_event_sink(
                lambda kind, payload: self._record(kind, payload)
            )
        self.commit = self.backend.reset()
        self._publish(self.commit, lifecycle="reset")
        self.phase_changed("critic")
        started = time.monotonic()
        self.critic.reset(self._critic_observation(), self._images())
        if time.monotonic() - started > self.limits.critic_timeout_s:
            raise GatewayError("CRITIC_EXECUTION_ERROR")
        self._record_feature_evidence()
        self._record(
            "baseline_initialized",
            {"observation_id": self.current["observation_id"]},
            public=True,
        )
        self.phase_changed("idle")
        self._save()
        return self.snapshot()

    def _images(self):
        # Independent read-only copies prevent a trusted buggy detector from
        # mutating policy pixels or the retained evidence.
        result = {}
        for key, value in self.frame_history[self.current["observation_id"]].items():
            copy = value.copy()
            copy.flags.writeable = False
            result[key] = copy
        return result

    def _record_feature_evidence(self):
        evidence = getattr(self.critic, "last_feature_evidence", None)
        if evidence is not None:
            if evidence.get("observation_id") != self.current["observation_id"]:
                raise GatewayError("CRITIC_EXECUTION_ERROR")
            self._record("real_feature_evidence", deepcopy(evidence))

    def _publish(self, commit, *, lifecycle, observation_id=None):
        references, images = self.images.publish(commit.policy.images)
        self.current = {
            "schema_version": "arx.public.observation.v1",
            "episode_nonce": self.episode_id,
            "observation_id": observation_id or f"obs-{self.step_index}",
            "step_index": self.step_index,
            "simulation_time_s": commit.simulation_time_s,
            "lifecycle": lifecycle,
            "cameras": references,
        }
        if commit.hardware is not None:
            self.current["clock_domain"] = "host_monotonic_ns"
            self.current["hardware"] = deepcopy(commit.hardware.observation)
            self.current["hardware"]["arrival_verified"] = commit.hardware.arrival_verified
            self.current["hardware"]["command_receipt"] = deepcopy(commit.hardware.command_receipt)
        if self.privileged and commit.privileged is not None:
            privileged = deepcopy(dataclasses.asdict(commit.privileged))
            self.current["privileged"] = privileged
            self._record("PrivilegedObservation", {
                "observation_id": self.current["observation_id"],
                "step_index": self.step_index,
                "privileged": privileged,
            }, public=False)
        self.current["event_sequence"] = self._record(
            "ObservationPublished", deepcopy(self.current), public=True
        )
        self.observations[self.current["observation_id"]] = deepcopy(self.current)
        self.frame_history[self.current["observation_id"]] = {
            **images,
            **(commit.hardware.feature_frames or {} if commit.hardware is not None else {}),
        }

    def snapshot(self):
        return {
            "observation": deepcopy(self.current),
            "state": self.state,
            "control_epoch": self.epoch,
            "event_sequence": self.sequence,
            "environment_ended": self.closed,
            "recovery_context": deepcopy(self.recovery),
            "budget_remaining": self._budgets(),
        }

    def _save(self):
        self.journal.save_snapshot(self.snapshot())

    def _budgets(self):
        return {
            "steps": max(0, self.limits.max_steps - self.step_index),
            "decisions": max(0, self.limits.max_decisions - self.decisions),
            "recoveries": max(0, self.limits.max_recoveries - self.incidents),
        }

    def _context(self):
        command = self.commit.command.copy()
        command.flags.writeable = False
        return ApprovedToolContext(command, deepcopy(self.current), self._images())

    def _critic_observation(self):
        observation = deepcopy(self.current)
        if self.privileged and self.commit.privileged is not None:
            p = self.commit.privileged
            observation.update({
                "privileged.selected.target_gripper_distance_m": p.selected["target_gripper_distance_m"],
                "privileged.interaction.gripper_closed": p.interaction["gripper_closed"],
                "privileged.interaction.grasped": p.interaction["grasped"],
                "privileged.interaction.progress": p.interaction["progress"],
            })
        return observation

    def policy_observation(self):
        from .backend import PolicyObservation

        return PolicyObservation(
            {k: v.copy() for k, v in self.commit.policy.images.items()},
            self.commit.policy.state.copy(),
        )

    def _result(self, request):
        return {
            "schema_version": "arx.tool.result.v1",
            "request_id": request.request_id,
            "operation_id": request.request_id,
            "event_sequence": self.sequence,
            "status": "accepted",
            "tool": request.tool,
            "observation_id_before": request.observation_id,
            "observation_id_after": self.current["observation_id"],
            "control_epoch": self.epoch,
            "executed_steps": 0,
            "planned_steps": None,
            "write_certainty": "none",
            "error": None,
            "critic_event_ids": [],
            "result": None,
            "budget_remaining": self._budgets(),
            "recovery_context": deepcopy(self.recovery),
        }

    def _validate(self, request):
        self.journal.check_decision(request)
        if self.active:
            raise GatewayError("OPERATION_IN_FLIGHT")
        if (
            request.observation_id != self.current["observation_id"]
            or request.control_epoch != self.epoch
        ):
            raise GatewayError("STALE_OBSERVATION")
        if self.state == "EXECUTION_UNCERTAIN":
            raise GatewayError("EXECUTION_UNCERTAIN")
        entry = self.registry.resolve(request.tool)
        if self.state not in entry.spec.allowed_states:
            raise GatewayError("TOOL_NOT_AUTHORIZED")
        args = entry.spec.input_model.model_validate(request.arguments)
        if self.recovery and self.programs and request.tool != "arx.finish":
            from robots.arx.deployment.bundle_program import resolve_call

            program = self.programs.get(self.recovery["binding_id"])
            if program is None or self.program_cursor >= len(program.calls):
                raise GatewayError("BUNDLE_PLAN_EXHAUSTED")
            expected = program.calls[self.program_cursor]
            values = resolve_call(expected, self.current["observation_id"], self.program_token,
                                  self.program_outputs)
            if request.tool != expected.tool or args.model_dump() != entry.spec.input_model.model_validate(values).model_dump():
                raise GatewayError("BUNDLE_STEP_MISMATCH")
        known = set(self.observations)
        known.update(
            ref["content_id"]
            for obs in self.observations.values()
            for ref in obs["cameras"].values()
        )
        if not set(request.evidence_ids) <= known:
            raise GatewayError("INVALID_EVIDENCE")
        if request.tool != "arx.finish" and self.decisions >= self.limits.max_decisions:
            raise GatewayError("DECISION_BUDGET_EXHAUSTED")
        if self.recovery and request.tool != "arx.finish":
            if request.tool not in self.recovery["permitted_tools"]:
                raise GatewayError("TOOL_NOT_AUTHORIZED")
            if self.recovery["remaining_decisions"] <= 0:
                raise GatewayError("RECOVERY_BUDGET_EXHAUSTED")
            if request.tool == "arx.zeva":
                if not self.token or args.reentry_token != self.token["value"]:
                    raise GatewayError("REENTRY_REQUIRED")
                if (
                    self.token["observation_id"] != request.observation_id
                    or self.token["epoch"] != self.epoch
                ):
                    raise GatewayError("STALE_REENTRY")
            elif entry.spec.kind == "execution":
                if self.binding.monitor_policy == "stop_all_motion":
                    raise GatewayError("STOP_ALL_MOTION")
                if self.recovery["remaining_steps"] <= 0:
                    raise GatewayError("RECOVERY_BUDGET_EXHAUSTED")
        if request.tool == "arx.review_reentry":
            ids = args.observation_ids
            if (
                any(x not in self.observations for x in ids)
                or ids[-1] != request.observation_id
            ):
                raise GatewayError("INVALID_REVIEW_WINDOW")
            steps = [self.observations[x]["step_index"] for x in ids]
            if steps != sorted(steps):
                raise GatewayError("INVALID_REVIEW_WINDOW")
        return entry, args

    def execute(self, request: ToolRequest):
        self._owned()
        duplicate = self.journal.duplicate(request)
        if duplicate is not None:
            return duplicate
        result = self._result(request)
        try:
            entry, args = self._validate(request)
            fresh_reentry = request.tool == "arx.review_reentry" and self.commit.hardware is not None
            fresh_gripper = request.tool == "arx.set_gripper" and self.commit.hardware is not None
            if request.tool in {"arx.propose_grasp", "arx.review_grasp", "arx.execute_grasp"} or fresh_reentry or fresh_gripper:
                self.phase_changed("observation")
                # Fresh sensor acquisition sends no hold command and consumes no physical steps.
                self.commit = self.backend.observe()
                self.read_samples += 1
                self._publish(self.commit, lifecycle="grasp-observation",
                              observation_id=f"obs-{self.step_index}-read-{self.read_samples}")
                self._retain_grasp_sensors()
                if fresh_reentry:
                    requested = list(args.observation_ids)
                    args = args.model_copy(update={"observation_ids": [*requested[:-1], self.current["observation_id"]]})
                    self._record("reentry_observation_refreshed", {
                        "requested_observation_ids": requested,
                        "review_observation_ids": args.observation_ids,
                        "step": self.step_index, "robot_commands_sent": False,
                    }, public=True)
                self._save()
            prepared = (
                entry.handler.prepare(args, self._context())
                if entry.spec.kind == "execution"
                else None
            )
            if prepared is not None and request.tool in {"arx.move_eef", "arx.execute_grasp"}:
                remaining = self.limits.max_steps - self.step_index
                if self.recovery:
                    remaining = min(remaining, self.recovery["remaining_steps"])
                if prepared.limit > remaining:
                    raise GatewayError("PLAN_EXCEEDS_BUDGET")
        except Exception as exc:
            result.update(status="rejected", error=self._error(exc, "validation"),
                          observation_id_after=self.current["observation_id"],
                          control_epoch=self.epoch)
            self.journal.admit(request, result)
            self._record("attempt_rejected", result, public=True)
            self.journal.update(result, final=True)
            return result
        self.journal.admit(request, result)
        self._record(
            "effective_arguments",
            {"request_id": request.request_id, "arguments": args.model_dump()},
        )
        if prepared is not None and hasattr(prepared, "admission_evidence"):
            self._record("motion_admission_evidence", {
                "request_id": request.request_id, "tool": request.tool,
                "evidence": prepared.admission_evidence,
            }, public=True)
        self.active = True
        self.decisions += 1
        bundle_call = (self.programs[self.recovery["binding_id"]].calls[self.program_cursor]
                       if self.recovery and self.programs and request.tool != "arx.finish"
                       else None)
        result["status"] = "running"
        self.journal.update(result)
        try:
            if self.recovery:
                self.recovery["remaining_decisions"] -= 1
            if entry.spec.kind == "episode_control":
                self.close()
                result.update(
                    status="completed",
                    result={"closed": True, "finalization": "complete"},
                )
            elif entry.spec.kind == "read_only":
                if request.tool == "arx.review_reentry":
                    self._review(entry.handler, args, result)
                else:
                    self.phase_changed("review")
                    value = entry.handler.inspect(args, {
                        "observation": deepcopy(self.current), "images": self._images(),
                        "command": self.commit.command.copy(),
                    })
                    result.update(status="completed", result=value)
            else:
                self._admit_motion(request)
                evidence = getattr(self.critic, "last_feature_evidence", None) or {}
                if (evidence.get("feature_observation", {}).get("status") == "unknown"
                        and self.commit.hardware is not None):
                    task_success = self._assess(result, terminal=False)
                    unavailable, task_success = self._reacquire_observation(result, task_success)
                else:
                    unavailable, task_success = None, False
                if unavailable:
                    result.update(status="cancelled" if unavailable == "cancelled" else "interrupted", result={
                        "completion": unavailable, "last_committed_step": self.step_index,
                        "command_target_reached": False, "physical_arrival_verified": False})
                elif task_success:
                    self.close()
                    result.update(status="completed", result={
                        "completion": "task_success", "last_committed_step": self.step_index,
                        "command_target_reached": None, "physical_arrival_verified": self.step_index > 0})
                else:
                    self.execute_targets(prepared, request, result)
            if result["result"] is not None:
                entry.spec.output_model.model_validate(result["result"])
            if bundle_call is not None and result["status"] == "completed":
                from robots.arx.deployment.bundle_program import verify_call_result, retain_tool_outputs

                try:
                    next_token = verify_call_result(
                        bundle_call, result, real=self.commit.hardware is not None
                    )
                except ValueError as exc:
                    self._record("bundle_step_failed", {
                        "tool": request.tool, "output": deepcopy(result["result"]),
                        "reason": str(exc), "observation_id": self.current["observation_id"],
                    })
                    self.close()
                    raise GatewayError("BUNDLE_STEP_FAILED") from exc
                if next_token:
                    self.program_token = next_token
                if result["result"].get("completion") != "task_success":
                    retain_tool_outputs(bundle_call, result, self.program_outputs)
                self.program_cursor += 1
        except Exception as exc:
            import traceback

            self._record("execution_error", {"traceback": traceback.format_exc()})
            result["result"] = None
            result.update(
                status="unknown" if self.state == "EXECUTION_UNCERTAIN" else "failed",
                error=self._error(exc, "execution"),
            )
            if self.state == "EXECUTION_UNCERTAIN":
                result["write_certainty"] = "unknown"
            elif result["executed_steps"]:
                result["write_certainty"] = "known_partial"
            self.suppressed.clear()
            self.token = None
            try:
                self.close()
            except Exception:
                self.closed = True
        finally:
            self.active = False
            self.phase_changed("idle")
            if self.recovery:
                self.recovery["last_execution_status"] = result["status"]
            result.update(
                observation_id_after=self.current["observation_id"],
                control_epoch=self.epoch,
                budget_remaining=self._budgets(),
                recovery_context=deepcopy(self.recovery),
            )
            result["event_sequence"] = self._record("tool_result", result, public=True)
            ToolResult.model_validate(result)
            self.journal.update(result, final=True)
            self._save()
        return result

    @staticmethod
    def _error(exc, phase):
        return {
            "code": exc.code
            if isinstance(exc, GatewayError)
            else "VALIDATION_ERROR"
            if phase == "validation"
            else "EXECUTION_ERROR",
            "phase": phase,
            "retry_class": "correct_request" if phase == "validation" else "never",
            "public_message": str(exc)
            if isinstance(exc, GatewayError)
            else "Contract validation failed"
            if phase == "validation"
            else "Execution failed; consult harness diagnostics",
        }

    def _retain_grasp_sensors(self):
        """Keep the original aligned metric depth needed to replay proposal reviews."""
        from .public import atomic_write
        import io
        import hashlib

        frames = self._images()
        stream = io.BytesIO()
        np.savez_compressed(stream, **frames)
        data = stream.getvalue()
        # ImageStore already owns this episode's public output directory.
        root = self.images.root.parent.parent
        target = root / "grasp-sensors" / (self.current["observation_id"] + ".npz")
        atomic_write(target, data)
        self._record("grasp_sensor_evidence", {
            "observation_id": self.current["observation_id"],
            "path": str(target.relative_to(root)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "depth_units": "mm", "observation": deepcopy(self.current),
        })

    def _admit_motion(self, request):
        if self.recovery and request.tool == "arx.zeva":
            self._record(
                "reentry_accepted",
                {"token": self.token, "recovery_id": self.recovery["recovery_id"]},
            )
            self.suppressed.clear()
            self.token = None
            self._lifecycle("reentry_accepted")
            self.recovery, self.binding = None, None
            self.state = "RUNNING_NOMINAL"
            self.epoch += 1
        elif self.recovery and self.state == "INTERRUPTED":
            self._record("recovery_acknowledged", deepcopy(self.recovery))
            self.suppressed = set(self.recovery["triggering_rule_ids"])
            self.state = "RECOVERING"
            self._lifecycle("recovery_started")
        elif not self.recovery:
            self.state = "RUNNING_NOMINAL"
        self._save()

    def execute_targets(self, prepared, request, result):
        """The sole physical target loop. Every committed step crosses the critic barrier."""
        completion = "plan_exhausted"
        hardware_steps = 0
        verified_steps = 0
        result["planned_steps"] = prepared.planned_steps
        while True:
            completion = self._gate(result)
            if completion:
                break
            self.phase_changed(
                "inference" if request.tool == "arx.zeva" else "planning"
            )
            batch = prepared.next_targets(self._context())
            if batch is None:
                completion = "plan_exhausted"
                break
            batch = np.asarray(batch, dtype=np.float32)
            if (
                batch.ndim != 2
                or batch.shape[1] != 14
                or len(batch) < 1
                or not np.isfinite(batch).all()
            ):
                raise GatewayError("INVALID_TARGET_BATCH")
            if result["executed_steps"] + len(batch) > prepared.limit:
                raise GatewayError("TOOL_HORIZON_EXCEEDED")
            for raw_target in batch:
                completion = self._gate(result)
                if completion:
                    break
                self.token = None
                if self.recovery:
                    self.recovery["reentry_token_available"] = False
                self._record(
                    "step_intent",
                    {
                        "request_id": request.request_id,
                        "step": self.step_index + 1,
                        "target": raw_target.tolist(),
                    },
                )
                self.phase_changed("step")
                try:
                    commit = self.backend.step(raw_target)
                    # A returned step is still uncertain until its commit is durable.
                    self._record(
                        "step_commit",
                        {
                            "request_id": request.request_id,
                            "step": self.step_index + 1,
                            "command": commit.command.tolist(),
                            "environment_ended": commit.environment_ended,
                        },
                    )
                except BaseException:
                    self.state = "EXECUTION_UNCERTAIN"
                    raise
                self.commit = commit
                if commit.hardware is not None:
                    hardware_steps += 1
                    verified_steps += int(commit.hardware.arrival_verified is True)
                self.step_index += 1
                result["executed_steps"] += 1
                result["write_certainty"] = "known_partial"
                if self.recovery:
                    self.recovery["remaining_steps"] -= 1
                self._record(
                    "private_evaluation",
                    {
                        "step": self.step_index,
                        "evaluation": _json_value(commit.private_evaluation),
                        "measured_state": commit.policy.state.tolist(),
                    },
                )
                self._publish(
                    commit, lifecycle="recovery" if self.recovery else "nominal"
                )
                if commit.hardware is not None and self.limits.retain_step_sensors:
                    self._retain_grasp_sensors()
                if commit.hardware is not None and commit.hardware.arrival_verified is not True:
                    self.state = "EXECUTION_UNCERTAIN"
                    self._save()
                    raise GatewayError("PHYSICAL_ARRIVAL_UNVERIFIED")
                prepared.on_commit(self._context())
                task_success = self._assess(result, terminal=commit.environment_ended)
                if not commit.environment_ended and not task_success:
                    wait_completion, task_success = self._reacquire_observation(result, task_success)
                    if wait_completion:
                        completion = wait_completion
                self._save()
                self.journal.update(result)
                if task_success:
                    self.close()
                    completion = "task_success"
                    break
                if commit.environment_ended:
                    self.close()
                    completion = "environment_ended"
                    break
                if completion:
                    break
                if self.state in {"INTERRUPTED", "ENDED"}:
                    completion = "critic_interrupted"
                    break
                if prepared.reached is True:
                    completion = "plan_exhausted"
                    break
            if completion or prepared.reached is True:
                break
        if (
            completion == "budget_exhausted"
            and self.step_index >= self.limits.max_steps
        ):
            self.close()
        if completion == "plan_exhausted" and prepared.reached is False:
            completion = "budget_exhausted"
        status = {"critic_interrupted": "interrupted", "observation_unavailable": "interrupted",
                  "cancelled": "cancelled"}.get(
            completion, "completed"
        )
        result.update(
            status=status,
            result={
                "completion": completion,
                "last_committed_step": self.step_index,
                "command_target_reached": prepared.reached,
                "physical_arrival_verified": bool(hardware_steps and hardware_steps == verified_steps),
            },
        )
        if result["executed_steps"] and status == "completed":
            result["write_certainty"] = "completed"

    def _gate(self, result):
        if self.cancel_requested():
            return "cancelled"
        if self.state in {"INTERRUPTED", "ENDED"}:
            return "critic_interrupted"
        if self.state == "EXECUTION_UNCERTAIN":
            raise GatewayError("EXECUTION_UNCERTAIN")
        if self.step_index >= self.limits.max_steps or (
            self.recovery and self.recovery["remaining_steps"] <= 0
        ):
            return "budget_exhausted"
        return None

    def _reacquire_observation(self, result, task_success=False):
        """Wait on sensors only; the separate controller keeps its last target."""
        def unavailable():
            return (self.commit.hardware is not None and self.assessment is not None
                    and self.assessment.features.get("feature_observation", {}).get("status") == "unknown")
        if not unavailable() or self.assessment.events or task_success:
            return None, task_success
        started = time.monotonic()
        deadline = started + self.limits.observation_reacquire_timeout_s
        self.token = None
        if self.recovery:
            self.recovery["reentry_token_available"] = False
        self._record("observation_wait_started", {
            "observation_id": self.current["observation_id"], "step": self.step_index,
            "timeout_s": self.limits.observation_reacquire_timeout_s,
            "feature_observation": self.assessment.features["feature_observation"],
            "robot_commands_sent": False,
        }, public=True)
        while unavailable() and not self.assessment.events:
            if self.cancel_requested():
                return "cancelled", False
            if time.monotonic() >= deadline:
                self.state = "INTERRUPTED"
                self.epoch += 1
                self._record("interrupt", {
                    "code": "OBSERVATION_UNAVAILABLE", "step": self.step_index,
                    "observation_id": self.current["observation_id"],
                    "feature_observation": self.assessment.features["feature_observation"],
                    "wait_s": time.monotonic() - started,
                    "task_failure": False, "controller_disable_requested": False,
                }, public=True)
                return "observation_unavailable", False
            time.sleep(min(.05, max(0., deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                continue
            self.phase_changed("observation")
            self.commit = self.backend.observe()
            self.read_samples += 1
            self._publish(self.commit, lifecycle="observation-reacquisition",
                          observation_id=f"obs-{self.step_index}-reacquire-{self.read_samples}")
            self._retain_grasp_sensors()
            if (self.step_index and self.commit.hardware.arrival_verified is not True):
                self.state = "EXECUTION_UNCERTAIN"
                raise GatewayError("PHYSICAL_ARRIVAL_UNVERIFIED")
            task_success = bool(self._assess(result, terminal=False))
            self._save()
            if task_success:
                break
        self._record("observation_reacquired", {
            "observation_id": self.current["observation_id"], "step": self.step_index,
            "wait_s": time.monotonic() - started, "robot_commands_sent": False,
        }, public=True)
        return None, task_success

    def _assess(self, result, *, terminal):
        self.phase_changed("critic")
        before = time.monotonic()
        try:
            assessment = self.critic.observe(self._critic_observation(), self._images())
            assessment = Assessment.model_validate(
                assessment.model_dump()
                if isinstance(assessment, Assessment)
                else assessment
            )
            if time.monotonic() - before > self.limits.critic_timeout_s:
                raise ValueError("critic deadline")
            if (
                assessment.observation_id != self.current["observation_id"]
                or assessment.step_index != self.step_index
            ):
                raise ValueError("stale assessment")
            if any(
                not set(event.evidence_observation_ids) <= set(self.observations)
                for event in assessment.events
            ):
                raise ValueError("unknown critic evidence")
        except Exception as exc:
            raise GatewayError("CRITIC_EXECUTION_ERROR") from exc
        self.assessment = assessment
        self._record_feature_evidence()
        self._record(
            "critic_assessment",
            {
                "assessment": assessment.model_dump(),
                "latency_s": time.monotonic() - before,
            },
            public=True,
        )
        if not terminal and hasattr(self.critic, "completion_evidence"):
            try:
                completion = self.critic.completion_evidence()
            except Exception as exc:
                raise GatewayError("CRITIC_EXECUTION_ERROR") from exc
            if completion is not None:
                if completion.get("observation_id") != self.current["observation_id"]:
                    raise GatewayError("CRITIC_EXECUTION_ERROR")
                self._record("task_success", completion, public=True)
                return True
        proposals = []
        for index, event in enumerate(assessment.events):
            event_id = "proposal-" + digest(
                {
                    "episode": self.episode_id,
                    "critic": getattr(self.critic, "sha256", assessment.critic_id),
                    "observation": assessment.observation_id,
                    "index": index,
                }
            )
            proposal = dict(event_id=event_id, **event.model_dump())
            self._record("critic_proposal", proposal, public=True)
            result["critic_event_ids"].append(event_id)
            stop_all = any(
                event.failure_mode in b.failure_modes
                and b.monitor_policy == "stop_all_motion"
                for b in self.bindings
            )
            if event.rule_id not in self.suppressed or stop_all:
                proposals.append(proposal)
            else:
                self._record("suppressed_proposal", proposal, public=True)
        if not proposals or terminal:
            return
        self.token = None
        self.epoch += 1
        if self.recovery:
            self.recovery["latest_proposal_ids"] = [p["event_id"] for p in proposals]
            self._record(
                "interrupt",
                {"code": "RECOVERY_ESCALATION_REQUIRED", "proposals": proposals},
                public=True,
            )
            self.close()
            return
        matches = sorted(
            (
                b
                for b in self.bindings
                if any(p["failure_mode"] in b.failure_modes for p in proposals)
            ),
            key=lambda b: (b.monitor_policy != "stop_all_motion", b.binding_id),
        )
        if not matches or self.incidents >= self.limits.max_recoveries:
            self._record(
                "interrupt",
                {
                    "code": "NO_RECOVERY_BINDING"
                    if not matches
                    else "RECOVERY_BUDGET_EXHAUSTED",
                    "proposals": proposals,
                },
                public=True,
            )
            self.close()
            return
        self.binding = matches[0]
        self.program_cursor = 0
        self.program_token = None
        self.program_outputs = {}
        self.incidents += 1
        self.state = "INTERRUPTED"
        self.recovery = {
            "recovery_id": f"recovery-{self.incidents}",
            "binding_id": self.binding.binding_id,
            "skill_entrypoint": self.binding.skill_entrypoint,
            "triggering_rule_ids": sorted({p["rule_id"] for p in proposals}),
            "latest_proposal_ids": [p["event_id"] for p in proposals],
            "permitted_tools": list(self.binding.allowed_tools),
            "remaining_steps": self.binding.max_recovery_steps,
            "remaining_decisions": self.binding.max_agent_decisions,
            "last_execution_status": "interrupted",
            "reentry_check_failures": [],
            "reentry_token_available": False,
        }
        self._record(
            "interrupt",
            {"proposals": proposals, "recovery_context": self.recovery},
            public=True,
        )

    def _review(self, handler, args, result):
        self.phase_changed("review")
        context = {
            "observations": [
                deepcopy(self.observations[x]) for x in args.observation_ids
            ],
            "images": [
                {k: v.copy() for k, v in self.frame_history[x].items()}
                for x in args.observation_ids
            ],
            "recovery_id": self.recovery["recovery_id"],
            "policy_id": self.binding.reentry_policy_id,
            "tool_outputs": deepcopy(self.program_outputs),
        }
        assessment = handler.inspect(args, context)
        assessment = ReentryAssessment.model_validate(
            assessment.model_dump()
            if isinstance(assessment, ReentryAssessment)
            else assessment
        )
        if (
            assessment.recovery_id != self.recovery["recovery_id"]
            or assessment.observation_id != self.current["observation_id"]
            or assessment.policy_id != self.binding.reentry_policy_id
        ):
            raise GatewayError("STALE_REENTRY")
        if any(
            not set(check.evidence_ids) <= set(args.observation_ids)
            for check in assessment.checks
        ):
            raise GatewayError("INVALID_REENTRY_EVIDENCE")
        eligible = assessment.status == "eligible" and all(
            check.status == "pass" for check in assessment.checks
        )
        self.token = None
        if eligible:
            self.token = {
                "value": uuid.uuid4().hex,
                "observation_id": assessment.observation_id,
                "epoch": self.epoch,
                "episode": self.episode_id,
                "package": self.package_sha256,
                "recovery_id": assessment.recovery_id,
                "assessment": assessment.model_dump(),
            }
            self._record("reentry_token", self.token)
        self.recovery["reentry_token_available"] = eligible
        self.recovery["reentry_check_failures"] = [
            c.model_dump() for c in assessment.checks if c.status != "pass"
        ]
        result.update(
            status="completed",
            result={
                "assessment": assessment.model_dump(),
                "reentry_token": self.token["value"] if self.token else None,
            },
        )

    def finalize_episode_artifacts(self):
        self._record(
            "artifact_finalization",
            {"finalization": "complete", "format": "gateway_journal_v1"},
        )

    def close(self):
        self._owned()
        if self.closed:
            return
        self.suppressed.clear()
        self.token = None
        uncertain = self.state == "EXECUTION_UNCERTAIN"
        self.phase_changed("close")
        try:
            self._lifecycle("episode_closed")
            self.finalize_episode_artifacts()
        finally:
            self.backend.close()
            self.closed = True
            self.state = "EXECUTION_UNCERTAIN" if uncertain else "ENDED"
            self.recovery = None
            self._save()


def _json_value(value):
    """Private recorder conversion; never used for public serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
