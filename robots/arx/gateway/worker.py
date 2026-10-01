# Copyright (c) 2026 Zetta Contributors
"""Parent supervision keeps reads responsive when native worker calls hang."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

from .contracts import GatewayError, ToolRequest, digest
from .journal import Journal


def _run(factory, commands, messages, cancel, stop, diagnostics_path=None):
    core = None
    try:
        core = factory(
            lambda: cancel.is_set() or stop.is_set(),
            lambda phase: messages.put(("phase", phase)),
        )
        core.reset()
        messages.put(("ready", core.registry.describe()))
        while not stop.is_set():
            try:
                request = commands.get(timeout=0.1)
            except queue.Empty:
                continue
            result = core.execute(ToolRequest.model_validate(request))
            messages.put(("result", result))
            if core.closed:
                break
        core.close()
        messages.put(("closed", None))
    except BaseException:
        # Full diagnostics stay in the harness-owned worker stderr.
        import traceback
        rendered = traceback.format_exc()
        if diagnostics_path is not None:
            try:
                Path(diagnostics_path).write_text(rendered, encoding="utf-8")
            except Exception:
                pass
        traceback.print_exc()
        messages.put(("fault", None))
    finally:
        if core is not None:
            try:
                core.close()
            finally:
                core.journal.close()


class EpisodeWorker:
    """One process per episode. A restart creates a new episode and journal."""

    def __init__(
        self, *, factory, output: Path, episode_id: str, limits, start_method="spawn"
    ):
        self.output, self.episode_id, self.limits = output, episode_id, limits
        self.journal = Journal(output / "journal.sqlite3")
        if self.journal.snapshot() is not None:
            self.journal.mark_lost()
            raise ValueError("cannot restart an existing physical episode")
        ctx = mp.get_context(start_method)
        self.commands, self.messages = ctx.Queue(), ctx.Queue()
        self.cancel_event, self.stop_event = ctx.Event(), ctx.Event()
        self.process = ctx.Process(
            target=_run,
            args=(
                factory,
                self.commands,
                self.messages,
                self.cancel_event,
                self.stop_event,
                str(output / "worker-error.log"),
            ),
            daemon=False,
        )
        self.lock = threading.RLock()
        self.pending = None
        self.catalog = None
        self.closed = False
        self.phase = "reset"
        self.phase_since = self.last_activity = self.lease = time.monotonic()
        self.operation_since = None
        self.stopping_since = None
        self.process.start()
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def heartbeat(self):
        with self.lock:
            self.lease = time.monotonic()

    def register_decision(self, request, *, source, evidence):
        with self.lock:
            self.journal.register_decision(request, source=source, evidence=evidence)
            self.journal.record("exposure", evidence)

    def submit(self, request):
        with self.lock:
            duplicate = self.journal.duplicate(request)
            if duplicate is not None:
                return duplicate
            if self.pending and self.pending[0].request_id == request.request_id:
                if digest(request.model_dump()) != digest(self.pending[0].model_dump()):
                    raise GatewayError("REQUEST_ID_CONFLICT")
                return self.pending[1].copy()
            if (
                self.closed
                or not self.process.is_alive()
                or self.stopping_since is not None
            ):
                raise GatewayError("EPISODE_CLOSED")
            self.journal.check_decision(request)
            if self.pending:
                raise GatewayError("OPERATION_IN_FLIGHT")
            snapshot = self.journal.snapshot()
            if snapshot is None:
                raise GatewayError("NOT_READY")
            result = {
                "schema_version": "arx.tool.result.v1",
                "request_id": request.request_id,
                "operation_id": request.request_id,
                "status": "accepted",
                "result": None,
                "write_certainty": "none",
                "executed_steps": 0,
                "event_sequence": snapshot["event_sequence"],
                "tool": request.tool,
                "observation_id_before": request.observation_id,
                "observation_id_after": snapshot["observation"]["observation_id"],
                "control_epoch": snapshot["control_epoch"],
                "planned_steps": None,
                "error": None,
                "critic_event_ids": [],
                "budget_remaining": snapshot["budget_remaining"],
                "recovery_context": snapshot["recovery_context"],
            }
            self.journal.record("dispatch_intent", request.model_dump())
            self.pending = (request, result)
            self.operation_since = self.last_activity = time.monotonic()
            self.cancel_event.clear()
            self.commands.put(request.model_dump())
            return result.copy()

    def status(self, request_id):
        with self.lock:
            result = self.journal.status(request_id)
            if result is not None:
                return result
            if self.pending and self.pending[0].request_id == request_id:
                return self.pending[1].copy()
            return None

    def cancel(self, request_id):
        with self.lock:
            if self.pending and self.pending[0].request_id == request_id:
                self.cancel_event.set()
            result = self.status(request_id)
            if result is None:
                raise GatewayError("UNKNOWN_REQUEST")
            return result

    def snapshot(self):
        with self.lock:
            return self.journal.snapshot()

    def events(self, after):
        with self.lock:
            return self.journal.events(after)

    def _lost(self):
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=1)
        if self.pending and self.journal.status(self.pending[0].request_id) is None:
            self.journal.admit(*self.pending)
        self.journal.mark_lost()
        self.closed = True

    def _monitor(self):
        while True:
            try:
                kind, value = self.messages.get(timeout=0.05)
            except queue.Empty:
                kind, value = None, None
            with self.lock:
                now = time.monotonic()
                if kind == "ready":
                    self.catalog = value
                    self.phase = "idle"
                    self.phase_since = self.last_activity = now
                elif kind == "phase":
                    self.phase, self.phase_since = value, now
                elif kind == "result":
                    self.pending = None
                    self.operation_since = None
                    self.last_activity = now
                elif kind == "closed":
                    self.closed = True
                    return
                elif kind == "fault":
                    self._lost()
                    return
                if not self.process.is_alive():
                    # Drain the final queued close message before classifying death.
                    if self.snapshot() and self.snapshot()["state"] == "ENDED":
                        self.closed = True
                    else:
                        self._lost()
                    return
                deadline = (
                    self.operation_since is not None
                    and now - self.operation_since > self.limits.operation_timeout_s
                )
                deadline |= (
                    self.phase == "critic"
                    and now - self.phase_since > self.limits.critic_timeout_s
                )
                deadline |= (
                    self.catalog is None
                    and now - self.last_activity > self.limits.operation_timeout_s
                )
                deadline |= now - self.lease > self.limits.lease_timeout_s
                deadline |= (
                    self.pending is None
                    and now - self.last_activity > self.limits.idle_agent_timeout_s
                )
                if deadline and self.stopping_since is None:
                    self.stopping_since = now
                    self.stop_event.set()
                    self.cancel_event.set()
                    self.journal.record(
                        "watchdog_stop", {"phase": self.phase}, public=True
                    )
                if (
                    self.stopping_since is not None
                    and now - self.stopping_since > self.limits.shutdown_timeout_s
                ):
                    self._lost()
                    return

    def close(self):
        with self.lock:
            if self.stopping_since is None:
                self.stopping_since = time.monotonic()
            self.stop_event.set()
            self.cancel_event.set()
        self.monitor.join(timeout=self.limits.shutdown_timeout_s + 2)
        with self.lock:
            if not self.closed:
                self._lost()
        self.process.join(timeout=1)
