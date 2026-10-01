# Copyright (c) 2026 Zetta Contributors
"""Agent turn orchestration, separate from physical execution and HTTP."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from .contracts import ToolRequest


@dataclass(frozen=True)
class AgentDecision:
    tool: str
    arguments: dict
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()


class EpisodeDriver:
    """Invoke one episode-local agent after each call, or only during recovery.

    `decide(event)` is a trusted provider adapter. It receives only public input
    and returns a tool choice. The harness persists the decision before dispatch.
    The callable retains this episode's conversation; never share it with learning.
    """

    def __init__(
        self,
        client,
        *,
        decide,
        register_decision,
        heartbeat,
        close_attempt,
        heartbeat_interval_s,
        operation_timeout_s,
        nominal_continuation=False,
    ):
        self.client, self.decide = client, decide
        self.register_decision, self.heartbeat, self.close_attempt = (
            register_decision,
            heartbeat,
            close_attempt,
        )
        self.heartbeat_interval_s, self.operation_timeout_s = (
            heartbeat_interval_s,
            operation_timeout_s,
        )
        self.nominal_continuation = nominal_continuation

    def run(self):
        stop = threading.Event()

        def keep_alive():
            while not stop.wait(self.heartbeat_interval_s):
                try:
                    self.heartbeat()
                except Exception:
                    return

        self.heartbeat()
        thread = threading.Thread(target=keep_alive, daemon=True)
        thread.start()
        last_result, sequence = None, 0
        try:
            catalog = self.client.catalog()
            while True:
                snapshot = self.client.observation()
                if snapshot["state"] in {"ENDED", "EXECUTION_UNCERTAIN"}:
                    return snapshot
                events = self.client.events(after=sequence)
                if events:
                    sequence = events[-1]["sequence"]
                event = {
                    "trigger": "critic_interrupt"
                    if snapshot["state"] == "INTERRUPTED"
                    else "tool_finished"
                    if last_result is not None
                    else "episode_started",
                    "snapshot": snapshot,
                    "events": events,
                    "last_result": last_result,
                    "catalog": catalog,
                }
                nominal = self.nominal_continuation and snapshot["state"] in {
                    "READY",
                    "RUNNING_NOMINAL",
                }
                choice = (
                    AgentDecision("arx.zeva", {"max_chunks": 1})
                    if nominal
                    else self.decide(event)
                )
                if not isinstance(choice, AgentDecision):
                    raise TypeError("deployment adapter must return AgentDecision")
                identity = uuid.uuid4().hex
                request = ToolRequest(
                    request_id=identity,
                    decision_ref="decision-" + identity,
                    observation_id=snapshot["observation"]["observation_id"],
                    control_epoch=snapshot["control_epoch"],
                    tool=choice.tool,
                    arguments=choice.arguments,
                    reason=choice.reason,
                    evidence_ids=list(choice.evidence_ids),
                )
                self.register_decision(
                    request, source="runner" if nominal else "agent", evidence=event
                )
                last_result = self.client.submit(request)
                if last_result["status"] in {"accepted", "running"}:
                    last_result = self.client.wait(
                        identity, timeout_s=self.operation_timeout_s
                    )
                if last_result["status"] == "unknown":
                    self.close_attempt()
                    return self.client.observation()
        except BaseException:
            self.close_attempt()
            raise
        finally:
            stop.set()
            thread.join(timeout=self.heartbeat_interval_s + 1)
