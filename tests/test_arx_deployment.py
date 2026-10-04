# Copyright (c) 2026 Zetta Contributors
"""Deployment orchestration tests: no duplicate motion or implicit model memory."""

import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from robots.arx.deployment.agent import ArxAgentAdapter
from robots.arx.deployment.contracts import AgentSettings, Decision, RunnerLimits
from robots.arx.deployment.runner import RolloutRunner, RunnerError
from tests.test_arx_gateway import ScriptCritic, make_core


def settings():
    return AgentSettings(
        planner_type="api",
        model="test:model",
        reasoning_effort="low",
        max_tokens=1000,
        max_turns=3,
        timeout_s=1.0,
        credential_env="TEST_KEY",
    )


def runner(tmp_path, core, adapter=None, **changes):
    limits = RunnerLimits(
        startup_timeout_s=2.0,
        episode_timeout_s=20.0,
        reconciliation_timeout_s=1.0,
        shutdown_timeout_s=1.0,
        heartbeat_interval_s=0.1,
        max_tool_attempts=10,
        max_agent_calls=3,
        max_recovery_agent_calls=3,
        max_contract_retries=1,
        **changes,
    )
    trial = SimpleNamespace(
        runner_limits=limits,
        agent=settings() if adapter else None,
        candidate=SimpleNamespace(package_sha256="a" * 64) if adapter else None,
    )
    r = RolloutRunner(trial, tmp_path, adapter_factory=lambda _: adapter)

    class Client:
        def observation(self):
            return core.snapshot()

        def events(self, after):
            return core.journal.events(after)

        def submit(self, request):
            return core.execute(request)

        def status(self, rid):
            return core.journal.status(rid)

        def image(self, rid):
            return (core.images.root / (rid + ".png")).read_bytes()

    class Admin:
        def register(self, request, source, evidence):
            core.journal.register_decision(request, source=source, evidence=evidence)

    r.client, r.admin = Client(), Admin()
    r.catalog = core.registry.describe()
    r.deadline = time.monotonic() + 20
    r.task = "Test task"
    return r


class FinishAdapter:
    def __init__(self):
        self.events = []

    def decide(self, event, images, invocation):
        self.events.append(event)
        return Decision(
            schema_version="arx.deployment.decision.v1",
            event_id=event["event_id"],
            observation_id=event["observation_id"],
            control_epoch=event["control_epoch"],
            tool="arx.finish",
            arguments={"reason": "done"},
            evidence_ids=[event["current_image_ids"][0]],
            rationale="Observed stop",
        )


def test_baseline_never_calls_agent(tmp_path):
    core, backend, _ = make_core(tmp_path / "core")
    core.limits = core.limits.model_copy(update={"max_steps": 8})
    r = runner(tmp_path / "run", core)
    assert r.loop() == "environment_ended"
    assert r.counts.agent_calls == 0 and backend.steps == 8


def test_trigger_handoff_and_finish(tmp_path):
    core, backend, _ = make_core(tmp_path / "core", ScriptCritic({2: "a"}))
    adapter = FinishAdapter()
    r = runner(tmp_path / "run", core, adapter)
    assert r.loop() == "agent_finish"
    assert backend.steps == 2 and len(adapter.events) == 1
    assert adapter.events[0]["critic_proposals"][0]["rule_id"] == "a"
    assert adapter.events[0]["observation_id"] == "obs-2"
    assert r.last["executed_steps"] == 0


def test_pagination_retains_late_trigger(tmp_path):
    core, _, _ = make_core(tmp_path / "core", ScriptCritic({2: "a"}))
    for i in range(130):
        core._record("diagnostic", {"index": i}, public=True)
    core._save()
    adapter = FinishAdapter()
    r = runner(tmp_path / "run", core, adapter)
    assert r.loop() == "agent_finish"
    assert adapter.events[0]["critic_proposals"]
    assert len(list((tmp_path / "run/events").glob("*.json"))) > 100


def test_invalid_agent_is_bounded(tmp_path):
    class Invalid:
        def decide(self, *args):
            raise ValueError("malformed")

    core, backend, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}))
    r = runner(tmp_path / "run", core, Invalid())
    with pytest.raises(RunnerError, match="contract_retries_exhausted"):
        r.loop()
    assert r.counts.agent_calls == 2 and backend.steps == 1
    core.close()


def test_gateway_exhaustion_does_not_retry(tmp_path):
    core, backend, _ = make_core(tmp_path / "core")
    core.limits = core.limits.model_copy(update={"max_decisions": 1})
    r = runner(tmp_path / "run", core)
    assert r.loop() == "gateway_budget"
    assert r.counts.tool_attempts == 1 and backend.steps == 4
    core.close()


def test_uncertain_submit_only_queries_original_id(tmp_path):
    core, backend, _ = make_core(tmp_path / "core")
    r = runner(tmp_path / "run", core)
    r.collect()
    original = r.client.submit
    ids = []

    def submit(request):
        ids.append(request.request_id)
        original(request)
        raise httpx.ReadTimeout("response lost")

    r.client.submit = submit
    result = r.dispatch("arx.zeva", {"max_chunks": 1}, "runner", {})
    assert result["executed_steps"] == 4 and len(ids) == 1 and backend.steps == 4
    core.close()


def test_failed_tool_result_counts_known_partial_step(tmp_path):
    core, _, _ = make_core(tmp_path / "core")
    r = runner(tmp_path / "run", core)
    r.collect()
    observation = r.snapshot["observation"]
    result = {
        "status": "failed", "executed_steps": 1,
        "observation_id_before": observation["observation_id"],
        "result": None, "write_certainty": "known_partial",
    }
    r.reconcile(SimpleNamespace(request_id="partial-step"), result)
    assert r.counts.physical_steps == observation["step_index"] + 1
    core.close()


def test_adapter_fresh_planners_and_public_image_tools(tmp_path):
    from zetta.planner.base import PlannerResult

    core, _, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}))
    r = runner(tmp_path / "run", core, FinishAdapter())
    r.collect()
    r.dispatch("arx.zeva", {"max_chunks": 1}, "runner", {})
    event = r.event(r.collect(), None)
    payloads = {key: r.client.image(key) for key in event["image_metadata"]}
    planners = []

    def factory(*args, **kwargs):
        class Planner:
            def solve(self, **kwargs):
                toolkit = kwargs["toolkit"]
                assert {t["name"] for t in toolkit.get_tools_spec()} == {
                    "read_arx_image",
                    "submit_decision",
                }
                toolkit.execute_tool(
                    "read_arx_image", {"content_id": event["current_image_ids"][0]}
                )
                decision = FinishAdapter().decide(event, payloads, None)
                result = toolkit.execute_tool("submit_decision", decision.model_dump())
                assert result.is_finish
                return PlannerResult()

        planner = Planner()
        planners.append(planner)
        return planner

    adapter = ArxAgentAdapter(settings(), planner_factory=factory)
    for index in range(2):
        adapter.decide(event, payloads, tmp_path / f"invocation-{index}")
    assert planners[0] is not planners[1]
    assert "secret_contact" not in str(event)
    core.close()


def test_missing_images_no_recovery_motion(tmp_path):
    core, backend, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}))
    r = runner(tmp_path / "run", core, FinishAdapter())

    def broken(_):
        raise OSError("missing image")

    r.client.image = broken
    with pytest.raises(RunnerError, match="agent_failure"):
        r.loop()
    assert backend.steps == 1
    core.close()


def test_model_timeout_no_fallback(tmp_path):
    class Slow:
        def decide(self, *args):
            time.sleep(0.4)

    core, backend, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}))
    r = runner(tmp_path / "run", core, Slow())
    r.trial.agent = r.trial.agent.model_copy(update={"timeout_s": 0.05})
    with pytest.raises(RunnerError, match="agent_timeout"):
        r.loop()
    assert backend.steps == 1
    core.close()


def test_full_trial_preflight_failure_writes_result(tmp_path):
    from robots.arx.deployment.contracts import Trial

    base = Path(__file__).resolve().parents[1]
    trial = Trial.model_validate(
        {
            "schema_version": "arx.rollout.trial.v1",
            "trial_id": "test",
            "mode": "baseline",
            "environment": {
                "scene": str(tmp_path),
                "mapping": str(tmp_path / "missing"),
                "task": str(base / "robots/arx/manifests/pickup_test_tube.yaml"),
                "model_contract": str(base / "robots/arx/manifests/task7_model_a.yaml"),
                "seed": 17,
            },
            "vla": {"host": "127.0.0.1", "port": 5581},
            "gateway": {
                "python": "/usr/bin/python3",
                "host": "127.0.0.1",
                "port": 8091,
                "runtime_limits": str(tmp_path / "limits"),
            },
            "runner_limits": {
                "startup_timeout_s": 5.0,
                "episode_timeout_s": 10.0,
                "reconciliation_timeout_s": 30.0,
                "shutdown_timeout_s": 1.0,
                "heartbeat_interval_s": 0.1,
                "max_tool_attempts": 2,
                "max_agent_calls": 2,
                "max_recovery_agent_calls": 2,
                "max_contract_retries": 0,
            },
            "evaluation": "none",
        }
    )
    result = RolloutRunner(trial, tmp_path / "result").run()
    assert result.status == "configuration_error" and result.counts.physical_steps == 0
    assert (tmp_path / "result/result.json").exists()


def test_complete_runner_through_http_worker(tmp_path):
    import socket
    import threading

    import uvicorn

    from robots.arx.deployment.runner import HarnessClient
    from robots.arx.gateway.client import ArxGatewayClient
    from robots.arx.gateway.service import create_app
    from robots.arx.gateway.worker import EpisodeWorker
    from tests.test_arx_gateway import WorkerFactory, await_condition, limits

    config = limits(max_steps=4, lease_timeout_s=10.0, idle_agent_timeout_s=10.0)
    worker = EpisodeWorker(
        factory=WorkerFactory(str(tmp_path / "gateway"), config=config),
        output=tmp_path / "gateway",
        episode_id="test",
        limits=config,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(worker, agent_capability="a" * 32, harness_capability="b" * 32),
            log_level="error",
        )
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        await_condition(lambda: server.started and worker.catalog is not None)
        core, _, _ = make_core(tmp_path / "unused")
        r = runner(tmp_path / "run", core)
        r.client = ArxGatewayClient(f"http://127.0.0.1:{port}", "test", "a" * 32)
        r.admin = HarnessClient(f"http://127.0.0.1:{port}", "b" * 32, 2.0)
        r.catalog = r.client.catalog()
        assert r.loop() == "environment_ended"
        assert r.counts.physical_steps == 4 and r.counts.agent_calls == 0
        r.cleanup()
        core.close()
    finally:
        server.should_exit = True
        thread.join(timeout=3)
        worker.close()
        sock.close()


def test_future_reentry_token_resumes_nominal(tmp_path):
    class Reenter:
        def decide(self, event, images, invocation):
            token = event["reentry_token"]
            tool = "arx.zeva" if token else "arx.review_reentry"
            arguments = (
                {"max_chunks": 1, "reentry_token": token}
                if token
                else {"observation_ids": [event["observation_id"]]}
            )
            return Decision(
                schema_version="arx.deployment.decision.v1",
                event_id=event["event_id"],
                observation_id=event["observation_id"],
                control_epoch=event["control_epoch"],
                tool=tool,
                arguments=arguments,
                evidence_ids=[],
                rationale="Trusted fake review test",
            )

    core, backend, _ = make_core(tmp_path / "core", ScriptCritic({1: "a"}))
    core.limits = core.limits.model_copy(update={"max_steps": 9})
    r = runner(tmp_path / "run", core, Reenter())
    assert r.loop() == "environment_ended"
    assert (
        r.outcome.reentry_completed and r.counts.agent_calls == 2 and backend.steps == 9
    )


def test_signal_stops_admission(tmp_path):
    core, backend, _ = make_core(tmp_path / "core")
    r = runner(tmp_path / "run", core)
    r.interrupted = True
    with pytest.raises(RunnerError, match="signal"):
        r.loop()
    assert backend.steps == 0
    core.close()
