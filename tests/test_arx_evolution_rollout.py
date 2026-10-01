import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.deployment.run_arx_evolution_rollout import _campaign_trial
from zetta.evolution.campaign import build_rollout_jobs
from zetta.evolution.queue import SharedHostQueue
from zetta.evolution.store import CampaignStore

ROOT = Path(__file__).resolve().parents[1]


def test_prepared_baseline_trial_uses_external_vla_and_hash_bound_scene(tmp_path):
    destination = tmp_path / "campaign"
    subprocess.run([sys.executable, str(ROOT / "scripts/evolution/prepare_arx_campaign.py"),
                    "--scenes-root", str(ROOT / "runs/arx_pickup_test_tube_10_new"),
                    "--output", str(destination)], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    manifest = json.loads((destination / "manifest.json").read_text())
    command = manifest["runtime"]["rollout_command"]
    assert "{env_endpoint}" not in " ".join(command)
    assert not any("start_zeva" in argument for argument in command)
    args = argparse.Namespace(
        campaign_root=destination, evidence_policy="arx_rgb_public_v1",
        scene_registry=destination / "scene-registry.json", task="pickup_test_tube",
        logical_id="g0000-rollout-000", seed=17, attempt_index=0,
        bundle_sha256="none", baseline_mode="strict_pure_vla", bundle=Path("none"),
        model_contract=ROOT / "robots/arx/manifests/task7_model_a.yaml",
        vla_host="127.0.0.1", vla_port=5581, gateway_port=5582,
        runtime_limits=destination / "runtime-limits.json",
        runner_limits=destination / "runner-limits.json",
        critic_runtime_limits=destination / "critic-runtime-limits.json",
        agent_settings=destination / "agent-settings.json",
    )
    trial = _campaign_trial(args)
    assert trial.mode == "baseline"
    assert trial.vla.port == 5581
    assert trial.environment.seed == 17
    args.seed = 999
    with pytest.raises(ValueError, match="scheduled scene/reset seed"):
        _campaign_trial(args)


def test_prepared_campaign_renders_shared_queue_jobs(tmp_path):
    destination = tmp_path / "campaign"
    subprocess.run([sys.executable, str(ROOT / "scripts/evolution/prepare_arx_campaign.py"),
                    "--scenes-root", str(ROOT / "runs/arx_pickup_test_tube_10_new"),
                    "--output", str(destination)], cwd=ROOT, check=True,
                   capture_output=True, text=True)
    store = CampaignStore(destination)
    store.initialize(store.manifest())
    jobs, blocked = build_rollout_jobs(store=store, queue=SharedHostQueue(tmp_path / "queue"), worker_hosts=("localhost",))
    assert blocked == []
    assert len(jobs) == 10
    for _, job in jobs:
        assert not job.requires_environment_slot
        assert "__ZETTA_ENV_ENDPOINT__" not in job.command
        assert job.command[job.command.index("--result-file") + 1] == job.result_file
