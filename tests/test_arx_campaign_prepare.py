"""Fixture-backed validation of the ARX campaign preparer."""
import json
import subprocess
import sys
from pathlib import Path

from zetta.evolution.models import CampaignManifest
from zetta.evolution.store import CampaignStore


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_scene_registry_and_manifest(tmp_path):
    output = tmp_path / "campaign"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/evolution/prepare_arx_campaign.py"),
         "--scenes-root", str(ROOT / "runs/arx_pickup_test_tube_10_new"),
         "--output", str(output)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    manifest = CampaignManifest.from_dict(json.loads((output / "manifest.json").read_text()))
    assert manifest.environment == "arx_mujoco"
    assert manifest.safety_layer.action_contract == "arx_gateway_named_tools_v1"
    assert manifest.runtime["evidence_policy"] == "arx_rgb_public_v1"
    assert manifest.runtime["rollout_requires_environment_slot"] is False
    assert len(manifest.heldout_seeds) == 20
    registry = json.loads((output / "scene-registry.json").read_text())
    assert len(registry["entries"]) == 10
    assert len(registry["heldout"]) == 20
    assert {entry["scene_id"] for entry in registry["heldout"]} == {
        entry["scene_id"] for entry in registry["entries"]
    }
    assert len({entry["reset_seed"] for entry in registry["heldout"]}) == 20
    assert registry["task_contract"]["language"] == "Pick up test tube with the pink label."
    assert all("model_contract" in entry["hashes"] for entry in registry["entries"])
    catalog = json.loads((output / "tool-catalog.json").read_text())
    assert manifest.tool_catalog_sha256 == catalog["catalog_sha256"]
    assert {row["name"] for row in catalog["tools"]} >= {"arx.zeva", "arx.finish"}
    assert CampaignStore(output).manifest() == manifest


def test_external_zeva_endpoint_is_frozen_from_config(tmp_path):
    output = tmp_path / "campaign"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"zeva": {"host": "127.0.0.2", "port": 5583}, "gateway_port": 5584}))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/evolution/prepare_arx_campaign.py"),
         "--scenes-root", str(ROOT / "runs/arx_pickup_test_tube_10_new"),
         "--config", str(config), "--output", str(output)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    manifest = CampaignManifest.from_dict(json.loads((output / "manifest.json").read_text()))
    command = manifest.runtime["rollout_command"]
    assert command[command.index("--vla-host") + 1] == "127.0.0.2"
    assert command[command.index("--vla-port") + 1] == "5583"
    assert command[command.index("--gateway-port") + 1] == "5584"
    assert manifest.runtime["vla_service_mode"] == "external"
    assert manifest.active_bundle_sha256 is None
    assert len(manifest.runtime["task_contract"]["evaluator_contract_sha256"]) == 64
