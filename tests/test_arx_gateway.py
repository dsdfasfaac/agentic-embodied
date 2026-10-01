# Copyright (c) 2026 Zetta Contributors
"""Gateway invariants using deterministic RGB and private-state traps."""

import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from robots.arx.gateway.backend import PolicyObservation, StepCommit
from robots.arx.gateway.contracts import (
    EefArgs,
    GatewayError,
    GripperArgs,
    HoldArgs,
    RecoveryBinding,
    RuntimeLimits,
    ToolRequest,
)
from robots.arx.gateway.journal import Journal
from robots.arx.gateway.session_core import ArxSessionCore, BaselineMonitor
from robots.arx.gateway.tools import ArxToolRegistry, GripperPlanner, default_registry
from robots.arx.gateway.zeva import ZevaPlanner
from robots.arx.mujoco_mapping import MujocoMapping


def limits(**changes):
    values = {
        "max_steps": 100,
        "max_decisions": 30,
        "max_recoveries": 3,
        "operation_timeout_s": 5.0,
        "critic_timeout_s": 2.0,
        "idle_agent_timeout_s": 5.0,
        "lease_timeout_s": 5.0,
        "shutdown_timeout_s": 0.2,
    }
    return RuntimeLimits(**(values | changes))


class FakeBackend:
    def __init__(self, terminal=999, fail=False, delay=0):
        self.steps, self.terminal, self.fail, self.delay = 0, terminal, fail, delay
        self.command = np.zeros(14, dtype=np.float32)
        self.closed = False

    def commit(self):
        pixels = {
            k: np.full((4, 5, 3), self.steps, np.uint8)
            for k in ("front_rgb", "left_rgb", "right_rgb")
        }
        return StepCommit(
            PolicyObservation(pixels, np.ones(14) * 987),
            self.command.copy(),
            self.steps / 15,
            self.steps >= self.terminal,
            {"secret_contact": True, "reward": 999},
        )

    def reset(self):
        return self.commit()

    def step(self, target):
        self.steps += 1
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("may already have stepped")
        self.command = target.copy()
        return self.commit()

    def close(self):
        self.closed = True


class ScriptCritic(BaselineMonitor):
    def __init__(self, fires=None, unknown=False):
        self.fires, self.unknown = fires or {}, unknown
        self.seen, self.lifecycle_events = [], []

    def reset(self, observation, images):
        assert observation["step_index"] == 0
        assert all(not x.flags.writeable for x in images.values())

    def observe(self, observation, images):
        assert "state" not in observation and "secret_contact" not in str(observation)
        step = observation["step_index"]
        self.seen.append(step)
        rule = self.fires.get(step)
        return {
            "critic_id": "test",
            "observation_id": observation["observation_id"],
            "step_index": step,
            "status": "failure" if rule else "unknown" if self.unknown else "clear",
            "events": [
                {
                    "detector_id": rule,
                    "failure_mode": rule,
                    "rule_id": rule,
                    "evidence_observation_ids": [observation["observation_id"]],
                    "reason_code": "visual",
                    "summary": "Visual failure",
                }
            ]
            if rule
            else [],
        }

    def lifecycle(self, event):
        self.lifecycle_events.append(event["kind"])


class Review:
    def inspect(self, args, context):
        return {
            "recovery_id": context["recovery_id"],
            "observation_id": args.observation_ids[-1],
            "policy_id": context["policy_id"],
            "status": "eligible",
            "checks": [
                {
                    "check_id": "visible",
                    "status": "pass",
                    "evidence_ids": args.observation_ids,
                    "reason_code": "visual",
                }
            ],
        }


def binding():
    return RecoveryBinding(
        binding_id="recover",
        failure_modes=["a", "b"],
        skill_entrypoint="recover",
        allowed_tools=["arx.hold", "arx.zeva", "arx.review_reentry"],
        max_recovery_steps=20,
        max_agent_decisions=10,
        monitor_policy="recovery_local",
        reentry_policy_id="visual",
    )


def make_core(tmp_path, critic=None, backend=None, config=None):
    backend = backend or FakeBackend()
    predictions = []
    core = None

    def predict(obs):
        predictions.append(backend.steps)
        assert obs.state[0] == 987
        return np.repeat((backend.command + 0.01)[None], 32, axis=0)

    zeva = ZevaPlanner(predict, lambda: core.policy_observation(), execution_steps=4)
    registry = default_registry(zeva=zeva, reentry=Review())
    core = ArxSessionCore(
        episode_id="test",
        backend=backend,
        registry=registry,
        journal=Journal(tmp_path / "journal.sqlite3"),
        output=tmp_path,
        limits=config or limits(),
        critic=critic or ScriptCritic(),
        bindings=[binding()],
    )
    core.reset()
    return core, backend, predictions


def call(core, tool="arx.zeva", args=None, identity=None, **changes):
    identity = identity or f"r-{core.decisions}-{len(core.journal.events(0))}"
    request = ToolRequest(
        request_id=identity,
        decision_ref=identity,
        observation_id=core.current["observation_id"],
        control_epoch=core.epoch,
        tool=tool,
        arguments=args or {"max_chunks": 1},
        **changes,
    )
    core.journal.register_decision(request, source="agent", evidence={})
    return core.execute(request), request


def test_baseline_parity_fresh_chunks_and_private_boundary(tmp_path):
    critic = ScriptCritic()
    core, backend, predictions = make_core(tmp_path, critic)
    result, _ = call(core, args={"max_chunks": 2})
    assert result["executed_steps"] == 8
    assert predictions == [0, 4]
    assert critic.seen == list(range(1, 9))
    assert core.current["simulation_time_s"] == pytest.approx(8 / 15)
    assert "987" not in str(core.snapshot()) and "secret_contact" not in str(result)
    assert "reward" not in str(core.journal.events(0))


def test_final_action_interrupt_first_recovery_and_suppression(tmp_path):
    critic = ScriptCritic({4: "a", 5: "a", 6: "a"})
    core, backend, _ = make_core(tmp_path, critic)
    result, _ = call(core)
    assert result["status"] == "interrupted" and result["executed_steps"] == 4
    invalid, _ = call(core, "arx.hold", {"steps": 0})
    assert invalid["status"] == "rejected" and core.state == "INTERRUPTED"
    assert not core.suppressed
    recovered, _ = call(core, "arx.hold", {"steps": 2})
    assert recovered["executed_steps"] == 2 and core.state == "RECOVERING"
    assert critic.seen == [1, 2, 3, 4, 5, 6]
    assert "recovery_started" in critic.lifecycle_events
    assert core.recovery is not None  # silence/firing suppression is not clearance


def test_additional_rule_closes_without_nested_recovery(tmp_path):
    core, backend, _ = make_core(tmp_path, ScriptCritic({1: "a", 3: "b"}))
    call(core)
    result, _ = call(core, "arx.hold", {"steps": 5})
    assert result["executed_steps"] == 2 and core.state == "ENDED"
    assert not core.suppressed
    assert "RECOVERY_ESCALATION_REQUIRED" in str(core.journal.events(0))


def test_unknown_does_not_interrupt_and_terminal_wins(tmp_path):
    core, _, _ = make_core(tmp_path, ScriptCritic(unknown=True))
    assert call(core)[0]["executed_steps"] == 4
    other = tmp_path / "terminal"
    core, _, _ = make_core(other, ScriptCritic({1: "a"}), FakeBackend(terminal=1))
    result, _ = call(core)
    assert result["result"]["completion"] == "environment_ended"
    assert core.state == "ENDED" and core.recovery is None
    assert result["critic_event_ids"]


def test_idempotency_precedes_staleness_and_conflicts(tmp_path):
    core, backend, _ = make_core(tmp_path)
    result, request = call(core)
    assert core.execute(request) == result and backend.steps == 4
    with pytest.raises(GatewayError, match="Request"):
        core.execute(request.model_copy(update={"arguments": {"max_chunks": 2}}))
    stale = request.model_copy(update={"request_id": "new", "decision_ref": "new"})
    core.journal.register_decision(stale, source="agent", evidence={})
    assert core.execute(stale)["error"]["code"] == "STALE_OBSERVATION"
    assert backend.steps == 4


def test_reentry_read_only_token_invalidated_by_motion(tmp_path):
    core, backend, predictions = make_core(tmp_path, ScriptCritic({1: "a"}))
    call(core)
    review, _ = call(
        core, "arx.review_reentry", {"observation_ids": ["obs-0", "obs-1"]}
    )
    token = review["result"]["reentry_token"]
    assert (
        token and review["executed_steps"] == 0 and review["write_certainty"] == "none"
    )
    call(core, "arx.hold", {"steps": 1})
    denied, _ = call(core, args={"max_chunks": 1, "reentry_token": token})
    assert denied["status"] == "rejected" and predictions == [0]
    review, _ = call(core, "arx.review_reentry", {"observation_ids": ["obs-2"]})
    call(
        core, args={"max_chunks": 1, "reentry_token": review["result"]["reentry_token"]}
    )
    assert predictions == [0, 2] and core.recovery is None and not core.suppressed


def test_uncertain_step_blocks_motion_and_no_replay(tmp_path):
    core, backend, _ = make_core(tmp_path, backend=FakeBackend(fail=True))
    result, request = call(core)
    assert result["status"] == "unknown" and result["write_certainty"] == "unknown"
    assert core.state == "EXECUTION_UNCERTAIN"
    assert core.execute(request) == result and backend.steps == 1


def test_cancellation_exact_prefix_and_recovery_preserved(tmp_path):
    core, backend, _ = make_core(tmp_path, ScriptCritic({1: "a"}))
    call(core)
    core.cancel_requested = lambda: backend.steps >= 3
    result, _ = call(core, "arx.hold", {"steps": 10})
    assert result["status"] == "cancelled" and result["executed_steps"] == 2
    assert core.state == "RECOVERING" and core.recovery


def test_budget_and_finish_zero_writes(tmp_path):
    core, backend, _ = make_core(tmp_path, config=limits(max_steps=2))
    result, _ = call(core)
    assert result["result"]["completion"] == "budget_exhausted" and backend.steps == 2
    result, _ = call(core, "arx.finish", {"reason": "done"})
    assert result["executed_steps"] == 0 and result["result"]["closed"]


@pytest.mark.parametrize(
    "model,args",
    [
        (HoldArgs, {"steps": True}),
        (HoldArgs, {"steps": 1, "extra": 0}),
        (GripperArgs, {"opening": float("nan"), "max_steps": 1}),
        (GripperArgs, {"opening": True, "max_steps": 1}),
        (EefArgs, {"delta_xyz_m": [0.01, 0.01, 0.0]}),
        (EefArgs, {"delta_xyz_m": [False, 0.0, 0.0]}),
    ],
)
def test_strict_arguments(model, args):
    with pytest.raises(ValidationError):
        model.model_validate(args)


def test_catalog_freeze_and_deep_copy(tmp_path):
    core, _, _ = make_core(tmp_path)
    registry = core.registry
    first = registry.describe()
    first["tools"][0]["input_schema"]["additionalProperties"] = True
    assert registry.describe() != first
    entry = registry.resolve("arx.hold")
    with pytest.raises(ValueError):
        registry.register(entry.spec, entry.handler)
    fresh = ArxToolRegistry(["command_state"])
    with pytest.raises(ValueError):
        fresh.register(entry.spec, object())
    fresh.register(entry.spec, entry.handler)
    with pytest.raises(ValueError):
        fresh.register(entry.spec, entry.handler)


def test_gripper_mapping_endpoints_and_zero_step_noop(tmp_path):
    from robots.arx.gateway.tools import ApprovedToolContext

    mapping = MujocoMapping(
        tuple((f"j{i}a", f"j{i}b") if i in (6, 13) else (f"j{i}",) for i in range(14)),
        tuple((f"a{i}a", f"a{i}b") if i in (6, 13) else (f"a{i}",) for i in range(14)),
        gripper_hardware_ranges=((-3.4, 0.0), (-2.0, 1.0)),
    )
    planner = GripperPlanner(mapping, command_offsets=(0.0, 0.9))
    context = ApprovedToolContext(np.zeros(14, np.float32), {})
    for opening, expected in [(0.0, 1.0), (1.0, -2.0)]:
        plan = planner.prepare(GripperArgs(opening=opening, max_steps=3), context)
        assert plan.targets[0, 13] == expected
        assert np.all(plan.targets[0, :13] == 0)
    context.command[13] = -2
    plan = planner.prepare(GripperArgs(opening=1.0, max_steps=3), context)
    assert plan.next_targets(context) is None and plan.reached


def test_critic_failure_after_commit_is_known_partial(tmp_path):
    class Broken(ScriptCritic):
        def observe(self, observation, images):
            raise RuntimeError("private path and credentials")

    core, backend, _ = make_core(tmp_path, Broken())
    result, _ = call(core)
    assert result["status"] == "failed" and result["executed_steps"] == 1
    assert result["write_certainty"] == "known_partial"
    assert result["error"]["code"] == "CRITIC_EXECUTION_ERROR"
    assert "credentials" not in str(result)
    assert backend.closed


def test_stop_all_motion_cannot_be_suppressed(tmp_path):
    core, backend, _ = make_core(tmp_path, ScriptCritic({1: "a"}))
    core.bindings = (
        binding().model_copy(update={"monitor_policy": "stop_all_motion"}),
    )
    call(core)
    result, _ = call(core, "arx.hold", {"steps": 1})
    assert result["status"] == "rejected" and backend.steps == 1


def test_full_zeva_chunk_validated_before_prefix(tmp_path):
    core, backend, _ = make_core(tmp_path)

    def bad(obs):
        targets = np.zeros((32, 14), dtype=np.float32)
        targets[-1, -1] = np.nan
        return targets

    core.registry.resolve("arx.zeva").handler.predict = bad
    result, _ = call(core)
    assert backend.steps == 0 and result["write_certainty"] == "none"


def test_episode_driver_delivers_each_result_and_decision(tmp_path):
    from robots.arx.gateway.episode import AgentDecision, EpisodeDriver

    core, backend, _ = make_core(tmp_path, ScriptCritic({2: "a"}))

    class Client:
        def catalog(self):
            return core.registry.describe()

        def observation(self):
            return core.snapshot()

        def events(self, after):
            return core.journal.events(after)

        def submit(self, request):
            return core.execute(request)

    turns = []

    def decide(event):
        turns.append(event)
        assert "secret_contact" not in str(event)
        return [
            AgentDecision("arx.zeva", {"max_chunks": 1}),
            AgentDecision("arx.hold", {"steps": 1}),
            AgentDecision("arx.finish", {"reason": "done"}),
        ][len(turns) - 1]

    driver = EpisodeDriver(
        Client(),
        decide=decide,
        register_decision=core.journal.register_decision,
        heartbeat=lambda: None,
        close_attempt=core.close,
        heartbeat_interval_s=0.1,
        operation_timeout_s=1.0,
    )
    result = driver.run()
    assert result["state"] == "ENDED" and len(turns) == 3
    assert turns[1]["last_result"]["status"] == "interrupted"
    assert turns[2]["snapshot"]["observation"]["step_index"] == 3


class WorkerFactory:
    def __init__(self, output, *, delay=0, config=None):
        self.output, self.delay, self.config = output, delay, config or limits()

    def __call__(self, cancel, phase):
        core, _, _ = make_core_unreset(Path(self.output), self.delay, self.config)
        core.cancel_requested, core.phase_changed = cancel, phase
        return core


def make_core_unreset(output, delay, config):
    backend = FakeBackend(delay=delay)
    core = None
    planner = ZevaPlanner(
        lambda obs: np.zeros((32, 14), np.float32),
        lambda: core.policy_observation(),
        execution_steps=4,
    )
    core = ArxSessionCore(
        episode_id="test",
        backend=backend,
        registry=default_registry(zeva=planner),
        journal=Journal(output / "journal.sqlite3"),
        output=output,
        limits=config,
        critic=BaselineMonitor(),
    )
    return core, backend, None


def await_condition(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition timed out")
        time.sleep(0.02)


async def test_worker_http_auth_decisions_images_and_disconnect(tmp_path):
    import httpx

    from robots.arx.gateway.service import create_app
    from robots.arx.gateway.worker import EpisodeWorker

    worker = EpisodeWorker(
        factory=WorkerFactory(str(tmp_path)),
        output=tmp_path,
        episode_id="test",
        limits=limits(),
    )
    try:
        await_condition(lambda: worker.catalog is not None)
        agent, admin = "a" * 32, "b" * 32
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_app(worker, agent_capability=agent, harness_capability=admin)
            ),
            base_url="http://test",
        ) as http:
            prefix = "/v1/episodes/test"
            assert (await http.get(prefix + "/observation")).status_code == 401
            http.headers["Authorization"] = "Bearer " + agent
            snapshot = (await http.get(prefix + "/observation")).json()
            image_id = snapshot["observation"]["cameras"]["front_rgb"]["content_id"]
            assert (await http.get(prefix + "/images/" + image_id)).headers[
                "content-type"
            ] == "image/png"
            assert (await http.post("/admin/heartbeat")).status_code == 401
            request = ToolRequest(
                request_id="http",
                decision_ref="http",
                observation_id="obs-0",
                control_epoch=0,
                tool="arx.zeva",
                arguments={"max_chunks": 1},
            )
            assert (
                await http.post(prefix + "/operations", json=request.model_dump())
            ).status_code == 409
            worker.register_decision(request, source="agent", evidence=snapshot)
            assert (
                await http.post(prefix + "/operations", json=request.model_dump())
            ).status_code == 202
        # Disconnect does not repeat or cancel a bounded operation.
        await_condition(lambda: worker.status("http")["status"] == "completed")
        assert worker.submit(request)["executed_steps"] == 4
        assert worker.snapshot()["observation"]["step_index"] == 4
    finally:
        worker.close()


def test_worker_hung_step_status_responsive_and_killed(tmp_path):
    from robots.arx.gateway.worker import EpisodeWorker

    config = limits(
        operation_timeout_s=0.7, lease_timeout_s=5.0, shutdown_timeout_s=0.1
    )
    worker = EpisodeWorker(
        factory=WorkerFactory(str(tmp_path), delay=10, config=config),
        output=tmp_path,
        episode_id="test",
        limits=config,
    )
    try:
        await_condition(lambda: worker.catalog is not None)
        request = ToolRequest(
            request_id="hang",
            decision_ref="hang",
            observation_id="obs-0",
            control_epoch=0,
            tool="arx.zeva",
            arguments={"max_chunks": 1},
        )
        worker.register_decision(request, source="agent", evidence={})
        worker.submit(request)
        started = time.monotonic()
        assert worker.snapshot()["observation"]["step_index"] == 0
        assert time.monotonic() - started < 0.2
        await_condition(lambda: worker.closed)
        assert worker.status("hang")["status"] == "unknown"
        assert worker.snapshot()["state"] == "EXECUTION_UNCERTAIN"
        assert not worker.process.is_alive()
    finally:
        worker.close()


def test_worker_lease_expiry_closes_paused_episode(tmp_path):
    from robots.arx.gateway.worker import EpisodeWorker

    config = limits(lease_timeout_s=0.8)
    worker = EpisodeWorker(
        factory=WorkerFactory(str(tmp_path), config=config),
        output=tmp_path,
        episode_id="test",
        limits=config,
    )
    try:
        await_condition(lambda: worker.catalog is not None)
        await_condition(lambda: worker.closed)
        assert worker.snapshot()["state"] == "ENDED"
        assert worker.snapshot()["observation"]["step_index"] == 0
    finally:
        worker.close()


def test_eef_calibration_and_budget_reject_before_motion():
    from robots.arx.gateway.motion import Calibration, CommandKinematics

    calibration = Calibration(
        base_position=[0.0, 0.0, 0.0],
        base_quaternion=[1.0, 0.0, 0.0, 0.0],
        links=[
            {
                "position": [0.1, 0.0, 0.0],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, 1.0],
                "limits": [-1.0, 1.0],
            }
            for _ in range(6)
        ],
        tcp_offset=[0.1, 0.0, 0.0],
    )
    kinematics = CommandKinematics(calibration)
    command = np.zeros(14, np.float32)
    targets = kinematics.plan(command, EefArgs(delta_xyz_m=[0.0, 0.0, 0.0]))
    assert targets.shape == (16, 14)
    with pytest.raises(ValueError, match="60"):
        kinematics.plan(
            command, EefArgs(delta_xyz_m=[0.01, 0.0, 0.0], speed_m_s=0.0001)
        )
    with pytest.raises(ValueError, match="limit"):
        kinematics.fk(np.full(6, 2.0))
    with pytest.raises(ValueError, match="converge"):
        kinematics.plan(command, EefArgs(delta_xyz_m=[0.001, 0.0, 0.0]))


def test_client_transport_failure_queries_same_identity():
    import httpx

    from robots.arx.gateway.client import ArxGatewayClient

    seen = []

    def transport(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            raise httpx.ReadTimeout("lost response", request=request)
        return httpx.Response(200, json={"status": "completed", "executed_steps": 4})

    client = ArxGatewayClient(
        "http://test", "episode", "capability", transport=httpx.MockTransport(transport)
    )
    request = ToolRequest(
        request_id="same",
        decision_ref="decision",
        observation_id="obs-0",
        control_epoch=0,
        tool="arx.zeva",
        arguments={"max_chunks": 1},
    )
    try:
        assert client.submit(request)["executed_steps"] == 4
        assert seen == [
            ("POST", "/v1/episodes/episode/operations"),
            ("GET", "/v1/episodes/episode/operations/same"),
        ]
    finally:
        client.close()


def test_motion_tools_native_after_normal_completion(tmp_path):
    core, backend, _ = make_core(tmp_path)
    result, _ = call(core)
    assert result["status"] == "completed" and core.recovery is None
    result, _ = call(core, "arx.hold", {"steps": 2})
    assert result["status"] == "completed" and result["executed_steps"] == 2
    assert backend.steps == 6 and core.recovery is None


def test_hardware_receipt_and_arrival_are_distinct_and_unverified_halts(tmp_path):
    from dataclasses import replace
    from robots.arx.gateway.backend import HardwareEvidence

    class ReceiptBackend(FakeBackend):
        def __init__(self, arrived):
            super().__init__()
            self.arrived = arrived
            self.sink = None

        def set_event_sink(self, sink):
            self.sink = sink

        def step(self, target):
            self.sink("command_dispatch_started", {"target": target.tolist()})
            self.sink("command_sent", {"status": "ros_publish_returned"})
            self.steps += 1
            self.command = target.copy()
            self.sink(
                "arrival_observed" if self.arrived else "arrival_unverified",
                {"verified": self.arrived},
            )
            return replace(
                self.commit(),
                hardware=HardwareEvidence(
                    observation={"state_monotonic_ns": self.steps},
                    command_receipt={"status": "ros_publish_returned"},
                    arrival_verified=self.arrived,
                ),
            )

    arrived_core, _, _ = make_core(tmp_path / "arrived", backend=ReceiptBackend(True))
    arrived_result, _ = call(arrived_core)
    assert arrived_result["result"]["physical_arrival_verified"] is True
    assert arrived_result["executed_steps"] == 4

    core, backend, _ = make_core(tmp_path / "unverified", backend=ReceiptBackend(False))
    result, _ = call(core)
    assert result["status"] == "unknown"
    assert result["executed_steps"] == 1
    assert backend.steps == 1
    kinds = [
        row[0] for row in core.journal.db.execute(
            "SELECT kind FROM records ORDER BY sequence"
        )
    ]
    assert kinds.index("command_sent") < kinds.index("arrival_unverified")
    assert "ObservationPublished" in kinds
