# Copyright (c) 2026 Zetta Contributors
"""Single-attempt orchestration. Never steps physics or evaluates a critic."""

from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import httpx

from robots.arx.critics import load_candidate
from robots.arx.gateway.client import ArxGatewayClient
from robots.arx.gateway.contracts import RuntimeLimits, ToolRequest, digest
from zetta.evolution.jsonio import atomic_write_json

from .agent import (
    BOOTSTRAP_SHA256,
    AgentFailure,
    ArxAgentAdapter,
    InvalidDecision,
    validate_decision,
)
from .contracts import Counts, Outcome, RolloutResult


class RunnerError(RuntimeError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


class HarnessClient:
    def __init__(self, url, capability, timeout):
        self.http = httpx.Client(
            base_url=url,
            headers={"Authorization": "Bearer " + capability},
            timeout=timeout,
        )

    def post(self, path, value=None):
        response = self.http.post("/admin/" + path, json=value)
        response.raise_for_status()
        return response.json()

    def register(self, request, source, evidence):
        return self.post(
            "decisions",
            {"request": request.model_dump(), "source": source, "evidence": evidence},
        )

    def heartbeat(self):
        return self.post("heartbeat")

    def stop(self):
        return self.post("stop")

    def close(self):
        self.http.close()


class RolloutRunner:
    def __init__(self, trial, output, *, adapter_factory=ArxAgentAdapter):
        self.trial, self.output = trial, Path(output)
        self.adapter_factory = adapter_factory
        self.attempt_id = uuid.uuid4().hex
        self.client = self.admin = self.process = None
        self.pending = None
        self.last = None
        self.snapshot = None
        self.catalog = None
        self.counts = Counts()
        self.outcome = Outcome()
        self.cursor = 0
        self.proposals = {}
        self.observation_records = {}
        self.active_proposals = {}
        self.images = {}
        self.observations = set()
        self.history = deque(maxlen=16)
        self.recovery_calls = {}
        self.skill = ""
        self.stop_event = threading.Event()
        self.heartbeat_thread = None
        self.heartbeat_failed = False
        self.interrupted = False
        self.deadline = 0
        self.cleanup_status = "not_started"

    def _count(self, name, amount=1):
        self.counts = self.counts.model_copy(
            update={name: getattr(self.counts, name) + amount}
        )

    def _save(self, name, value):
        return atomic_write_json(self.output / name, value)

    def preflight(self):
        t = self.trial
        for path in [
            t.environment.scene,
            t.environment.mapping,
            t.environment.task,
            t.environment.model_contract,
            t.gateway.python,
            t.gateway.runtime_limits,
        ]:
            Path(path).resolve(strict=True)
        limits = RuntimeLimits.model_validate_json(
            Path(t.gateway.runtime_limits).read_text()
        )
        r = t.runner_limits
        if r.heartbeat_interval_s * 3 >= limits.lease_timeout_s:
            raise ValueError("heartbeat interval too large for lease")
        needed = (t.agent.timeout_s if t.agent else 0) + r.reconciliation_timeout_s
        if limits.idle_agent_timeout_s <= needed:
            raise ValueError(
                "gateway idle deadline cannot accommodate agent/reconciliation"
            )
        self.gateway_limits = limits
        self.package_path = None
        if t.candidate:
            c = t.candidate
            if Path(c.package).is_file():
                from zetta.evolution.jsonio import canonical_sha256, read_json
                if canonical_sha256(read_json(Path(c.package))) != c.package_sha256:
                    raise ValueError("structured bundle identity mismatch")
                if not os.environ.get(t.agent.credential_env):
                    raise ValueError("configured provider credential reference is unavailable")
                self.package_path = Path(c.package).resolve()
                self.skill = (
                    "Structured privileged recovery bundle. On the critic interruption, execute "
                    "the recovery steps in order using the exact permitted tools. "
                    "Open the gripper, move the EEF by 2cm, call arx.review_reentry with "
                    "the current observation ID, then call arx.zeva with the returned reentry token."
                )
                self._structured_bundle = True
                self._candidate_sha256 = c.package_sha256
                self._save("private/trial.json", t.model_dump())
                self._save("identities.json", {"package": c.package_sha256, "bootstrap": BOOTSTRAP_SHA256, "trial_sha256": digest(t.model_dump())})
                return
            package = load_candidate(
                Path(c.package),
                expected_contract=c.contract_sha256,
                expected_catalog=c.catalog_sha256,
                expected_bootstrap=c.bootstrap_sha256,
            )
            if (
                package.sha256 != c.package_sha256
                or c.bootstrap_sha256 != BOOTSTRAP_SHA256
            ):
                raise ValueError("package/bootstrap identity mismatch")
            if not os.environ.get(t.agent.credential_env):
                raise ValueError(
                    "configured provider credential reference is unavailable"
                )
            self.package_path = self.output / "private" / "candidate"
            self.package_path.mkdir(parents=True)
            payloads = dict(package.payloads)
            payloads["manifest.json"] = package.manifest_bytes
            skill = []
            for name, content in payloads.items():
                target = self.package_path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(0o400)
                if name == "skill/SKILL.md" or name.startswith("skill/references/"):
                    if len(content) > 65536 or not name.endswith(".md"):
                        raise ValueError("invalid skill reference")
                    skill.append(name + "\n" + content.decode())
            self.skill = "\n\n".join(skill)
            if len(self.skill) > 131072:
                raise ValueError("skill capsule too large")
            for directory in sorted(self.package_path.rglob("*"), reverse=True):
                if directory.is_dir():
                    directory.chmod(0o500)
            self.package_path.chmod(0o500)
        self._save("private/trial.json", t.model_dump())
        self._save(
            "identities.json",
            {
                "package": t.candidate.package_sha256 if t.candidate else None,
                "bootstrap": BOOTSTRAP_SHA256,
                "trial_sha256": digest(t.model_dump()),
            },
        )

    def start(self):
        t = self.trial
        env = t.environment
        args = [
            t.gateway.python,
            "-m",
            "scripts.deployment.serve_arx_gateway",
            "--scene",
            env.scene,
            "--mapping",
            env.mapping,
            "--task",
            env.task,
            "--contract",
            env.model_contract,
            "--output",
            str(self.output / "private/gateway"),
            "--seed",
            str(env.seed),
            "--runtime-config",
            t.gateway.runtime_limits,
            "--zeva-host",
            t.vla.host,
            "--zeva-port",
            str(t.vla.port),
            "--listen-host",
            t.gateway.host,
            "--listen-port",
            str(t.gateway.port),
        ]
        if env.calibration:
            args += ["--calibration", env.calibration]
        if self.package_path:
            args += [
                "--bundle" if getattr(self, "_structured_bundle", False) else "--package",
                str(self.package_path),
            ]
            if not getattr(self, "_structured_bundle", False):
                args += ["--critic-runtime-config", t.candidate.critic_runtime_limits]
        else:
            args += ["--baseline"]
        if env.privileged or getattr(self, "_structured_bundle", False):
            args += ["--privileged"]
        log = (self.output / "private/gateway-process.log").open("wb")
        child_env = os.environ.copy()
        codex_key = child_env.get("CODEX_API_KEY")
        codex_base = child_env.get("CODEX_BASE_URL")
        for key in list(child_env):
            if key.endswith(
                ("API_KEY", "ACCESS_TOKEN", "SECRET_KEY")
            ) or key.startswith("ZETTA_API_"):
                child_env.pop(key, None)
        if t.agent:
            # The rollout contract names the credential reference generically;
            # pydantic-ai's OpenAI provider consumes the standard names.
            if t.agent.credential_env == "CODEX_API_KEY" and codex_key:
                child_env["OPENAI_API_KEY"] = codex_key
                if codex_base:
                    child_env["OPENAI_BASE_URL"] = codex_base
            child_env.pop(t.agent.credential_env, None)
        child_env.update(MUJOCO_GL="osmesa", XDG_CACHE_HOME="/tmp/zetta-arx-cache")
        self.process = subprocess.Popen(
            args,
            cwd=Path(__file__).resolve().parents[3],
            env=child_env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        log.close()
        deadline = min(
            self.deadline, time.monotonic() + t.runner_limits.startup_timeout_s
        )
        credentials = self.output / "private/gateway/capabilities.json"
        while time.monotonic() < deadline:
            if self.interrupted:
                raise RunnerError("interrupted", "signal")
            if self.process.poll() is not None:
                raise RunnerError("infrastructure_error", "gateway_startup_failed")
            try:
                if self.client is None:
                    value = json.loads(credentials.read_text())
                    self.episode_id = value["episode_id"]
                    host = "[::1]" if t.gateway.host == "::1" else t.gateway.host
                    url = f"http://{host}:{t.gateway.port}"
                    self.client = ArxGatewayClient(
                        url,
                        self.episode_id,
                        value["agent_capability"],
                        timeout_s=min(5.0, t.runner_limits.reconciliation_timeout_s),
                    )
                    self.admin = HarnessClient(
                        url,
                        value["harness_capability"],
                        min(5.0, t.runner_limits.shutdown_timeout_s),
                    )
                    self._start_heartbeat()
                self.admin.heartbeat()
                self.catalog = self.client.catalog()
                self.snapshot = self.client.observation()
                if self.snapshot["observation"]["episode_nonce"] != self.episode_id:
                    raise ValueError("episode identity mismatch")
                if (
                    t.candidate
                    and self.catalog["catalog_sha256"] != t.candidate.catalog_sha256
                ):
                    raise RunnerError("configuration_error", "catalog_mismatch")
                self._save("catalog.json", self.catalog)
                self._save("reset.json", self.snapshot)
                return
            except (OSError, ValueError, httpx.HTTPError):
                time.sleep(0.1)
        raise RunnerError("infrastructure_error", "startup_timeout")

    def _start_heartbeat(self):
        def beat():
            while not self.stop_event.wait(
                self.trial.runner_limits.heartbeat_interval_s
            ):
                try:
                    self.admin.heartbeat()
                except Exception:
                    self.heartbeat_failed = True

        self.heartbeat_thread = threading.Thread(target=beat, daemon=True)
        self.heartbeat_thread.start()

    def collect(self):
        deadline = min(
            self.deadline,
            time.monotonic() + self.trial.runner_limits.reconciliation_timeout_s,
        )
        records = []
        while True:
            snapshot = self.client.observation()
            sequence = snapshot["event_sequence"]
            while True:
                if time.monotonic() >= deadline:
                    raise RunnerError("infrastructure_error", "evidence_timeout")
                page = self.client.events(after=self.cursor)
                selected = [x for x in page if x["sequence"] <= sequence]
                if selected and selected[0]["sequence"] <= self.cursor:
                    raise RunnerError("infrastructure_error", "invalid_event_order")
                for item in selected:
                    self.cursor = item["sequence"]
                    records.append(item)
                    if len(records) > self.trial.runner_limits.max_event_records:
                        raise RunnerError("infrastructure_error", "evidence_budget")
                    self._save(f"events/{item['sequence']:09d}.json", item)
                    if item["kind"] == "critic_proposal":
                        self.proposals[item["payload"]["event_id"]] = item["payload"]
                    if item["kind"] == "ObservationPublished":
                        self.observation_records[item["payload"]["observation_id"]] = (
                            item["payload"]
                        )
                        self.observations.add(item["payload"]["observation_id"])
                        for ref in item["payload"]["cameras"].values():
                            self.images[ref["content_id"]] = ref
                if len(page) < 100 or any(x["sequence"] > sequence for x in page):
                    break
                if not selected:
                    raise RunnerError(
                        "infrastructure_error", "event_pagination_stalled"
                    )
            fresh = self.client.observation()
            if fresh["event_sequence"] == sequence:
                break
        self.snapshot = fresh
        obs = fresh["observation"]
        self.observations.add(obs["observation_id"])
        for ref in obs["cameras"].values():
            self.images[ref["content_id"]] = ref
        self.counts = self.counts.model_copy(
            update={"physical_steps": obs["step_index"]}
        )
        recovery = fresh["recovery_context"]
        if recovery and not set(recovery["latest_proposal_ids"]) <= set(self.proposals):
            raise RunnerError("infrastructure_error", "missing_trigger_evidence")
        return records

    def reconcile(self, request, initial=None):
        deadline = min(
            self.deadline,
            time.monotonic() + self.trial.runner_limits.reconciliation_timeout_s,
        )
        result = initial
        while time.monotonic() < deadline:
            if result and result["status"] not in ("accepted", "running"):
                self.pending = None
                self.last = result
                self._save(f"tools/{request.request_id}-result.json", result)
                self.counts = self.counts.model_copy(
                    update={
                        "physical_steps": max(
                            self.counts.physical_steps,
                            result.get("result", {}).get("last_committed_step", 0)
                            if result.get("result")
                            else 0,
                        )
                    }
                )
                return result
            try:
                result = self.client.status(request.request_id)
            except (httpx.HTTPError, TimeoutError):
                result = None
            time.sleep(0.05)
        raise RunnerError("execution_uncertain", "operation_unresolved")

    def dispatch(self, tool, arguments, source, evidence, *, closing=False):
        if self.pending:
            raise RunnerError("execution_uncertain", "pending_operation")
        if not closing:
            self._count("tool_attempts")
        identity = uuid.uuid4().hex
        request = ToolRequest(
            request_id=identity,
            decision_ref="decision-" + identity,
            observation_id=self.snapshot["observation"]["observation_id"],
            control_epoch=self.snapshot["control_epoch"],
            tool=tool,
            arguments=arguments,
            evidence_ids=evidence.get("decision_evidence_ids", []),
            reason=evidence.get("rationale", "Automatic bounded nominal continuation"),
        )
        self._save(
            f"tools/{identity}-request.json",
            {
                "request": request.model_dump(),
                "source": source,
                "input_sha256": digest(evidence),
            },
        )
        self.admin.register(
            request,
            source,
            {
                "input_sha256": digest(evidence),
                "observation_id": request.observation_id,
            },
        )
        self.pending = request
        try:
            result = self.client.submit(request)
        except (httpx.HTTPError, TimeoutError):
            result = None
        result = self.reconcile(request, result)
        if result["status"] == "rejected":
            self._count("rejected_tools")
        if result["status"] == "unknown" or result["write_certainty"] == "unknown":
            raise RunnerError("execution_uncertain", "gateway_unknown")
        self.history.append(
            {
                "decision_id": identity,
                "tool": tool,
                "effective_arguments": arguments,
                "result_status": result["status"],
                "executed_steps": result["executed_steps"],
                "write_certainty": result["write_certainty"],
                "observation_id_after": result["observation_id_after"],
                "result": result.get("result"),
            }
        )
        return result

    def event(self, records, feedback):
        s = self.snapshot
        recovery = s["recovery_context"]
        rid = recovery["recovery_id"]
        if rid not in self.recovery_calls:
            self.recovery_calls[rid] = 0
            self._count("recoveries")
        active = [
            p
            for p in self.proposals.values()
            if p["rule_id"] in recovery["triggering_rule_ids"]
            or p["event_id"] in recovery["latest_proposal_ids"]
        ]
        if rid not in self.active_proposals:
            self.active_proposals[rid] = active
        active = list(
            {p["event_id"]: p for p in self.active_proposals[rid] + active}.values()
        )
        current = list(s["observation"]["cameras"].values())
        metadata = {r["content_id"]: r for r in current}
        # Include image references actually attached to retained proposal observations.
        for proposal in active:
            for observation_id in proposal["evidence_observation_ids"]:
                observation = self.observation_records.get(observation_id)
                if observation is None:
                    raise RunnerError("infrastructure_error", "missing_proposal_frame")
                for ref in observation["cameras"].values():
                    metadata[ref["content_id"]] = ref
        token = None
        if self.history and self.history[-1]["tool"] == "arx.review_reentry":
            token = (self.history[-1]["result"] or {}).get("reentry_token")
        return {
            "schema_version": "arx.deployment.decision_input.v1",
            "event_id": "event-" + uuid.uuid4().hex,
            "observation_id": s["observation"]["observation_id"],
            "control_epoch": s["control_epoch"],
            "package_sha256": self.trial.candidate.package_sha256,
            "bootstrap_sha256": BOOTSTRAP_SHA256,
            "fresh_context": True,
            "resume_thread_id": None,
            "task": self.task,
            "skill_contents_and_required_references": self.skill,
            "public_tool_catalog": self.catalog,
            "current_images": s["observation"]["cameras"],
            "image_metadata": metadata,
            "current_image_ids": [r["content_id"] for r in current],
            "critic_proposals": active,
            "recovery_context": recovery,
            "allowed_tools": list(
                dict.fromkeys(recovery["permitted_tools"] + ["arx.finish"])
            ),
            "remaining_budgets": {
                "gateway": s["budget_remaining"],
                "agent_calls": self.trial.runner_limits.max_agent_calls
                - self.counts.agent_calls,
            },
            "evidence_ids": sorted(
                set(metadata)
                | {s["observation"]["observation_id"]}
                | {oid for p in active for oid in p["evidence_observation_ids"]}
            ),
            "history": list(self.history),
            "feedback": feedback,
            "reentry_token": token,
        }

    def _decide(self, adapter, event):
        self._count("agent_calls")
        rid = self.snapshot["recovery_context"]["recovery_id"]
        self.recovery_calls[rid] += 1
        self.outcome = self.outcome.model_copy(update={"recovery_attempted": True})
        invocation = self.output / f"invocations/{self.counts.agent_calls:06d}"
        # No motion authority exists in this thread. Timeout discards any eventual
        # result; cleanup stops the gateway and no next decision is admitted.
        if self.adapter_factory is ArxAgentAdapter:
            import multiprocessing as mp

            from .agent import planner_process

            payloads = {key: self.client.image(key) for key in event["image_metadata"]}
            context = mp.get_context("spawn")
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=planner_process,
                args=(self.trial.agent, event, payloads, str(invocation), child),
            )
            process.start()
            child.close()
            deadline = min(self.deadline, time.monotonic() + self.trial.agent.timeout_s)
            try:
                while time.monotonic() < deadline:
                    if self.interrupted:
                        raise RunnerError("interrupted", "signal")
                    if parent.poll(0.05):
                        response = parent.recv()
                        if "decision" in response:
                            from .contracts import Decision

                            return Decision.model_validate(response["decision"])
                        if response.get("invalid"):
                            raise InvalidDecision(response["error"])
                        raise AgentFailure(response["error"])
                    if not process.is_alive():
                        raise AgentFailure("planner process lost")
                self._save(
                    f"invocations/{self.counts.agent_calls:06d}-timeout.json",
                    {"reason": "agent_timeout"},
                )
                raise RunnerError("agent_error", "agent_timeout")
            finally:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
                parent.close()
        # Injected test adapters have no provider process; still bound their wait.
        future = concurrent.futures.Future()

        def invoke():
            try:
                payloads = {
                    key: self.client.image(key) for key in event["image_metadata"]
                }
                result = adapter.decide(event, payloads, invocation)
                future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        deadline = min(self.deadline, time.monotonic() + self.trial.agent.timeout_s)
        while time.monotonic() < deadline:
            if self.interrupted:
                raise RunnerError("interrupted", "signal")
            try:
                return future.result(timeout=0.05)
            except concurrent.futures.TimeoutError:
                pass
        self._save(
            f"invocations/{self.counts.agent_calls:06d}-timeout.json",
            {"reason": "agent_timeout"},
        )
        raise RunnerError("agent_error", "agent_timeout")

    def loop(self):
        limits = self.trial.runner_limits
        adapter = self.adapter_factory(self.trial.agent) if self.trial.agent else None
        failures = 0
        feedback = None
        while True:
            if self.interrupted:
                raise RunnerError("interrupted", "signal")
            if time.monotonic() >= self.deadline:
                raise RunnerError("infrastructure_error", "episode_timeout")
            if self.heartbeat_failed:
                raise RunnerError("infrastructure_error", "heartbeat_failed")
            records = self.collect()
            state = self.snapshot["state"]
            if state == "EXECUTION_UNCERTAIN":
                raise RunnerError("execution_uncertain", "gateway_unknown")
            if state == "ENDED":
                for record in reversed(records):
                    if record["kind"] == "interrupt" and record["payload"].get("code"):
                        return record["payload"]["code"].lower()
                return "environment_ended"
            if self.counts.tool_attempts >= limits.max_tool_attempts:
                return "runner_budget"
            if any(
                self.snapshot["budget_remaining"][k] <= 0
                for k in ("steps", "decisions")
            ):
                return "gateway_budget"
            recovery = self.snapshot["recovery_context"]
            if state in ("READY", "RUNNING_NOMINAL"):
                if self.outcome.recovery_attempted:
                    self.outcome = self.outcome.model_copy(
                        update={"reentry_completed": True}
                    )
                tool, arguments, source, evidence = (
                    "arx.zeva",
                    {"max_chunks": 1},
                    "runner",
                    {},
                )
            elif state in ("INTERRUPTED", "RECOVERING") and adapter:
                event = self.event(records, feedback)
                if (
                    self.counts.agent_calls >= limits.max_agent_calls
                    or self.recovery_calls[recovery["recovery_id"]]
                    >= limits.max_recovery_agent_calls
                ):
                    return "runner_budget"
                if recovery["remaining_decisions"] <= 0:
                    return "gateway_budget"
                try:
                    decision = self._decide(adapter, event)
                    decision = validate_decision(decision.model_dump(), event)
                except RunnerError:
                    raise
                except (ValueError, InvalidDecision) as exc:
                    self._count("tool_attempts")
                    failures += 1
                    feedback = {
                        "code": "INVALID_DECISION",
                        "message": type(exc).__name__,
                    }
                    self._save(
                        f"agent-attempts/{self.counts.agent_calls:06d}.json", feedback
                    )
                    if failures > limits.max_contract_retries:
                        raise RunnerError("agent_error", "contract_retries_exhausted")
                    continue
                except Exception as exc:
                    raise RunnerError("agent_error", "agent_failure") from exc
                tool, arguments, source = decision.tool, decision.arguments, "agent"
                evidence = dict(
                    event,
                    decision_evidence_ids=decision.evidence_ids,
                    rationale=decision.rationale,
                )
            else:
                raise RunnerError("infrastructure_error", "unsupported_state")
            result = self.dispatch(tool, arguments, source, evidence)
            if result["status"] == "failed":
                raise RunnerError(
                    "infrastructure_error",
                    "critic_error"
                    if (result["error"] or {}).get("code") == "CRITIC_EXECUTION_ERROR"
                    else "tool_failure",
                )
            if result["status"] == "rejected":
                code = (result["error"] or {}).get("code", "")
                if "BUDGET" in code:
                    return "gateway_budget"
                failures += 1
                feedback = result
                if source == "runner" or failures > limits.max_contract_retries:
                    raise RunnerError("agent_error", "contract_retries_exhausted")
            else:
                failures = 0
                feedback = None
            if tool == "arx.finish" and result["status"] == "completed":
                return "agent_finish"

    def cleanup(self):
        errors = []
        try:
            if self.client and not self.pending:
                self.snapshot = self.client.observation()
                if self.snapshot["state"] not in ("ENDED", "EXECUTION_UNCERTAIN"):
                    # Cleanup gets a bounded window even after the episode deadline.
                    self.deadline = (
                        time.monotonic() + self.trial.runner_limits.shutdown_timeout_s
                    )
                    self.dispatch(
                        "arx.finish",
                        {"reason": "Runner closing attempt"},
                        "runner",
                        {},
                        closing=True,
                    )
            elif self.pending and self.client:
                self.client.cancel(self.pending.request_id)
                self.deadline = (
                    time.monotonic() + self.trial.runner_limits.shutdown_timeout_s
                )
                self.reconcile(self.pending)
        except Exception:
            errors.append("finish_or_cancel_failed")
        try:
            if self.admin:
                self.admin.stop()
        except Exception:
            errors.append("admin_stop_failed")
        self.stop_event.set()
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=self.trial.runner_limits.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
                errors.append("forced_kill")
        if self.client:
            self.client.close()
        if self.admin:
            self.admin.close()
        self.cleanup_status = "complete" if not errors else ",".join(errors)

    def run(self):
        # The shared queue executor creates the attempt directory before
        # launching us so it can attach ``worker.stdout.log``.  Direct
        # invocations may still arrive with a new path, so both cases are
        # supported; immutable attempt identity is enforced by the queue.
        self.output.mkdir(parents=True, exist_ok=True)
        self.output.chmod(0o700)
        (self.output / "private").mkdir(mode=0o700, exist_ok=True)
        self.deadline = time.monotonic() + self.trial.runner_limits.episode_timeout_s
        status, reason, error = "configuration_error", "preflight_failed", None
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(
                    sig, lambda *_: setattr(self, "interrupted", True)
                )
        try:
            self.preflight()
            self.task = json.loads(Path(self.trial.environment.task).read_text())[
                "instruction"
            ]
            status = "infrastructure_error"
            reason = "startup_failed"
            self.start()
            reason = self.loop()
            status = "completed"
        except RunnerError as exc:
            status, reason = exc.status, exc.reason
            error = {"type": type(exc).__name__, "message": reason}
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": "See private diagnostics"}
            self._save("private/error.json", {"message": str(exc)})
        finally:
            try:
                self.cleanup()
            except Exception:
                self.cleanup_status = "cleanup_failed"
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if self.pending:
            status, reason = "execution_uncertain", "operation_unresolved"
        if self.cleanup_status != "complete" and status == "completed":
            status, reason = "infrastructure_error", "cleanup_failed"
        c = self.trial.candidate
        result = RolloutResult(
            trial_id=self.trial.trial_id,
            attempt_id=self.attempt_id,
            episode_id=getattr(self, "episode_id", None),
            mode=self.trial.mode,
            package_sha256=c.package_sha256 if c else None,
            identities={
                "contract": c.contract_sha256 if c else None,
                "catalog": self.catalog["catalog_sha256"] if self.catalog else None,
                "bootstrap": BOOTSTRAP_SHA256,
            },
            status=status,
            termination_reason=reason,
            outcome=self.outcome,
            counts=self.counts,
            last_operation={
                k: self.last[k] for k in ("request_id", "status", "write_certainty")
            }
            if self.last
            else None,
            artifact_paths={
                "events": "events",
                "tools": "tools",
                "invocations": "invocations",
                "identities": "identities.json",
            },
            cleanup_status=self.cleanup_status,
            error=error,
        )
        self._save("result.json", result.model_dump())
        return result
