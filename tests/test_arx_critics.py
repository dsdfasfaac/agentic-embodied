# Copyright (c) 2026 Zetta Contributors
"""Real candidate code, OS isolation, temporal replay, and gateway handoff."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from robots.arx.critics import (
    ArxCriticRegistry,
    WorkerLimits,
    load_candidate,
    seal_candidate,
)
from robots.arx.critics.isolation import IsolationError
from robots.arx.gateway.public import ImageStore
from tests.test_arx_gateway import FakeBackend, call, make_core

FIXTURE = Path(__file__).parent / "fixtures/arx_candidate_rgb_v1"


def limits(**changes):
    return WorkerLimits(
        **(
            {
                "python": sys.executable,
                "max_history": 60,
                "max_evaluation_ms": 5000,
                "startup_timeout_s": 10.0,
                "memory_bytes": 2147483648,
                "cpu_seconds": 60,
                "scratch_bytes": 1048576,
                "image_width": 320,
                "image_height": 240,
                "max_message_bytes": 2097152,
            }
            | changes
        )
    )


def register(path=FIXTURE, **kwargs):
    registry = ArxCriticRegistry(limits=limits(**kwargs))
    package = registry.register_package(path)
    return registry, package


def observation(store, step, red):
    images = {
        k: np.zeros((240, 320, 3), np.uint8)
        for k in ("front_rgb", "left_rgb", "right_rgb")
    }
    images["front_rgb"][:, :, 0] = red
    cameras, images = store.publish(images)
    return {
        "schema_version": "arx.public.observation.v1",
        "episode_nonce": "opaque",
        "observation_id": f"obs-{step}",
        "step_index": step,
        "simulation_time_s": step / 15,
        "cameras": cameras,
        "lifecycle": "reset" if step == 0 else "nominal",
        "event_sequence": step,
    }, images


def rewrite(tmp_path, mutate):
    path = tmp_path / "candidate"
    shutil.copytree(FIXTURE, path)
    mutate(path)
    metadata = json.loads((path / "manifest.json").read_text())
    for key in ("files", "feature_schema_sha256"):
        metadata.pop(key)
    seal_candidate(path, metadata)
    return path


@pytest.mark.parametrize(
    "reds,expected",
    [([0, 0, 0], []), ([0, 0, 255, 255, 0], [3]), ([0, 255, 0, 255], [])],
)
def test_real_sandbox_replay_and_duplicate_observation(tmp_path, reds, expected):
    registry, _ = register()
    critic = registry.freeze()
    store = ImageStore(tmp_path / "images")
    try:
        critic.reset(*observation(store, 0, reds[0]))
        fires = []
        for step, red in enumerate(reds[1:], 1):
            obs, images = observation(store, step, red)
            assessment = critic.observe(obs, images)
            assert critic.observe(obs, images) == assessment
            assert (
                assessment.features["mean-red-monitor"]["visual.mean_red"]["value"]
                == red / 255
            )
            if assessment.events:
                fires.append(step)
        assert fires == expected
        with pytest.raises(ValueError, match="frozen"):
            registry.register_package(FIXTURE)
    finally:
        critic.close()


class RGBBackend(FakeBackend):
    def commit(self):
        from robots.arx.gateway.backend import PolicyObservation, StepCommit

        red = {2: 255, 3: 255}.get(self.steps, 0)
        images = {
            k: np.zeros((240, 320, 3), np.uint8)
            for k in ("front_rgb", "left_rgb", "right_rgb")
        }
        images["front_rgb"][:, :, 0] = red
        return StepCommit(
            PolicyObservation(images, np.ones(14) * 987),
            self.command.copy(),
            self.steps / 15,
            False,
            {"secret_contact": self.steps},
        )


def test_real_extractor_interrupts_gateway_before_fourth_target(tmp_path):
    registry, package = register()
    critic = registry.freeze()
    # make_core resets with its baseline monitor; install this frozen critic and
    # initialize its reset baseline before any physical action.
    core, backend, _ = make_core(tmp_path, backend=RGBBackend())
    core.critic = critic
    core.bindings = package.bindings
    critic.reset(core.current, core._images())
    try:
        result, _ = call(core)
        assert result["status"] == "interrupted"
        assert result["executed_steps"] == 3 and backend.steps == 3
        assert result["observation_id_after"] == "obs-3" and core.epoch == 1
        assert core.recovery["permitted_tools"] == ["arx.finish"]
        assert core.recovery["remaining_steps"] == 0
        assert core.current["cameras"]["front_rgb"]["sha256"]
        for _ in range(3):
            core.snapshot()
        assert backend.steps == 3
        result, _ = call(
            core, "arx.finish", {"reason": "RGB critic functional test completed"}
        )
        assert result["executed_steps"] == 0 and core.closed
    finally:
        critic.close()


def test_hashes_paths_and_missing_bindings_rejected(tmp_path):
    path = tmp_path / "candidate"
    shutil.copytree(FIXTURE, path)
    with (path / "critic/features.py").open("a") as f:
        f.write("\n# changed\n")
    with pytest.raises(ValueError, match="hash"):
        load_candidate(path)
    (path / "extra").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        load_candidate(path)


@pytest.mark.parametrize(
    "source",
    [
        'return {"visual.mean_red":{"valid":True,"value":True}}',
        'return {"visual.undeclared":{"valid":True,"value":1.0}}',
        'return {"visual.mean_red":{"valid":True,"value":float("nan")}}',
        'open("/etc/passwd").read(); return {}',
        "import socket; socket.socket(); return {}",
        "while True: pass",
    ],
)
def test_bad_output_forbidden_access_and_timeout_fail_closed(tmp_path, source):
    def mutate(path):
        (path / "critic/features.py").write_text(
            "class FeatureExtractor:\n def reset(self,o,i,c): pass\n def close(self): pass\n def extract(self,o,i):\n  "
            + source
            + "\n"
        )
        m = json.loads((path / "critic/manifest.json").read_text())
        m["evaluation_timeout_ms"] = 300
        (path / "critic/manifest.json").write_text(json.dumps(m))

    path = rewrite(tmp_path, mutate)
    registry, _ = register(path)
    critic = registry.freeze()
    store = ImageStore(tmp_path / "images")
    try:
        critic.reset(*observation(store, 0, 0))
        with pytest.raises((ValueError, IsolationError)):
            critic.observe(*observation(store, 1, 255))
    finally:
        critic.close()


def test_invalid_feature_resets_dwell_only_and_unknown(tmp_path):
    def mutate(path):
        code = (
            (path / "critic/features.py")
            .read_text()
            .replace(
                "value = float(image",
                'if int(image[0, 0, 0]) == 127:\n            return {"visual.mean_red": {"valid": False, "value": None}}\n        value = float(image',
            )
        )
        (path / "critic/features.py").write_text(code)

    registry, _ = register(rewrite(tmp_path, mutate))
    critic = registry.freeze()
    store = ImageStore(tmp_path / "images")
    try:
        critic.reset(*observation(store, 0, 0))
        statuses = [
            critic.observe(*observation(store, i, r)).status
            for i, r in enumerate([255, 127, 255, 255], 1)
        ]
        assert statuses == ["clear", "unknown", "clear", "failure"]
    finally:
        critic.close()


def test_prm_placeholder_fails_registration(tmp_path):
    def mutate(path):
        m = json.loads((path / "critic/manifest.json").read_text())
        m["source"]["kind"] = "prm_service"
        m["input_profile"] = "arx.prm_inputs.v1"
        (path / "critic/manifest.json").write_text(json.dumps(m))

    with pytest.raises(NotImplementedError, match="PRM_SOURCE_NOT_IMPLEMENTED"):
        register(rewrite(tmp_path, mutate))


async def test_registered_critic_http_handoff(tmp_path):
    import hashlib
    import threading

    import httpx

    from robots.arx.gateway.service import create_app

    registry, package = register()
    critic = registry.freeze()
    core, backend, _ = make_core(tmp_path, backend=RGBBackend())
    core.critic = critic
    core.bindings = package.bindings
    critic.reset(core.current, core._images())

    class LocalFacade:
        episode_id = "test"
        output = tmp_path
        lock = threading.RLock()
        journal = core.journal
        catalog = core.registry.describe()
        closed = False

        def snapshot(self):
            return core.snapshot()

        def events(self, after):
            return core.journal.events(after)

        def status(self, identity):
            return core.journal.status(identity)

        def submit(self, request):
            return core.execute(request)

        def register_decision(self, request, **kwargs):
            core.journal.register_decision(request, **kwargs)

    try:
        result, _ = call(core)
        app = create_app(
            LocalFacade(), agent_capability="a" * 32, harness_capability="b" * 32
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + "a" * 32},
        ) as client:
            snapshot = (await client.get("/v1/episodes/test/observation")).json()
            ref = snapshot["observation"]["cameras"]["front_rgb"]
            image = await client.get("/v1/episodes/test/images/" + ref["content_id"])
            assert hashlib.sha256(image.content).hexdigest() == ref["sha256"]
            assert snapshot["observation"]["observation_id"] == "obs-3"
            assert snapshot["state"] == "INTERRUPTED" and backend.steps == 3
    finally:
        core.close()


def test_preflight_and_unknown_feature(tmp_path):
    registry, _ = register()
    registry.preflight()

    def mutate(path):
        config = json.loads((path / "critic/config.json").read_text())
        config["rules"][0]["feature"] = "visual.missing"
        (path / "critic/config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="undeclared"):
        register(rewrite(tmp_path, mutate))


def test_critic_error_keeps_physical_commit_known(tmp_path):
    def mutate(path):
        (path / "critic/features.py").write_text(
            'class FeatureExtractor:\n def reset(self,o,i,c): pass\n def close(self): pass\n def extract(self,o,i): raise RuntimeError("broken")\n'
        )

    registry, package = register(rewrite(tmp_path, mutate))
    critic = registry.freeze()
    core, backend, _ = make_core(tmp_path / "episode", backend=RGBBackend())
    core.critic = critic
    core.bindings = package.bindings
    critic.reset(core.current, core._images())
    result, _ = call(core)
    assert result["error"]["code"] == "CRITIC_EXECUTION_ERROR"
    assert (
        result["executed_steps"] == 1 and result["write_certainty"] == "known_partial"
    )
    assert backend.steps == 1 and core.closed


def test_missing_camera_and_wrong_frame_fail(tmp_path):
    registry, _ = register()
    critic = registry.freeze()
    store = ImageStore(tmp_path / "images")
    try:
        critic.reset(*observation(store, 0, 0))
        obs, images = observation(store, 1, 255)
        images["front_rgb"] = np.zeros((10, 10, 3), np.uint8)
        with pytest.raises(ValueError, match="camera metadata"):
            critic.observe(obs, images)
    finally:
        critic.close()
