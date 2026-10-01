"""The shared ARX agent resolver must never index private simulator data."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robots.arx.trajectory_recorder import record_public_trajectory
from tests.test_arx_trajectory_recorder import fixture
from zetta.evolution.lifecycle import _agent_artifact_context, resolve_agent_artifact
from zetta.evolution.store import CampaignStore


ROOT = Path(__file__).resolve().parents[1]


def campaign(tmp_path):
    root = tmp_path / "campaign"
    subprocess.run([sys.executable, str(ROOT / "scripts/evolution/prepare_arx_campaign.py"),
                    "--scenes-root", str(ROOT / "runs/arx_pickup_test_tube_10_new"),
                    "--output", str(root)], check=True, cwd=ROOT, capture_output=True)
    attempt = root / "attempts/g0000-rollout-000/attempt-000"
    fixture(attempt)
    index = record_public_trajectory(attempt)
    store = CampaignStore(root)
    store.episodes.append({"episode_id": "episode-public-0", "logical_id": "g0000-rollout-000",
                           "attempt_index": 0, "status": "valid", "success": False,
                           "artifact_index": index})
    return store, attempt


def test_only_public_rgb_and_execution_artifacts_are_readable(tmp_path):
    store, attempt = campaign(tmp_path)
    index, _ = _agent_artifact_context(store)
    assert index["diagnostic_telemetry"] == []
    assert index["artifacts"]
    for item in index["artifacts"]:
        result = resolve_agent_artifact(store.root, item["content_id"])
        if result["kind"] == "file":
            assert "/private/" not in result["path"]
    assert not any("qpos" in str(item) for item in index["artifacts"])


def test_private_artifact_is_rejected_before_prompt(tmp_path):
    store, attempt = campaign(tmp_path)
    row = store.episodes.records()[0]
    row["artifact_index"]["artifacts"].append({
        "path": "private/gateway/journal.sqlite3", "sha256": "0" * 64})
    # Append-only ledgers are authoritative; exercise the validator directly
    # through a small read-only stand-in instead of rewriting campaign history.
    class StandIn:
        root = store.root

        def manifest(self):
            return store.manifest()

        class episodes:
            @staticmethod
            def records():
                return [row]

    with pytest.raises(ValueError, match="private or unsupported"):
        _agent_artifact_context(StandIn())
