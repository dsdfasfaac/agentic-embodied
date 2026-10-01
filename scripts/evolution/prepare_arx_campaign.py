#!/usr/bin/env python3
"""Freeze an ARX campaign manifest and scene/reset registry."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zetta.evolution.models import CampaignManifest
from zetta.evolution.stages import CLUSTER_SYSTEM_PROMPT, DIAGNOSIS_SYSTEM_PROMPT, PROPOSAL_SYSTEM_PROMPT
from zetta.evolution.jsonio import atomic_write_json
from robots.arx.evolution_defaults import CANDIDATE_KIND, EVIDENCE_POLICY, SAFETY_LAYER, PRIVILEGED_CANDIDATE_KIND, PRIVILEGED_EVIDENCE_POLICY
from robots.arx.deployment.agent import BOOTSTRAP_SHA256
from robots.arx.gateway.tools import default_registry

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value: object) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


class _CatalogPrepare:
    def prepare(self, args, context):  # pragma: no cover - catalog schema only
        raise RuntimeError("catalog-only handler")


class _CatalogInspect:
    def inspect(self, args, context):  # pragma: no cover - catalog schema only
        raise RuntimeError("catalog-only handler")


def _tool_catalog() -> dict:
    """Freeze the exact named-tool schema configured by the ARX gateway."""
    return default_registry(
        zeva=_CatalogPrepare(), gripper=_CatalogPrepare(),
        eef=_CatalogPrepare(), reentry=_CatalogInspect(),
    ).describe()

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--scenes-root", type=Path, required=True); p.add_argument("--task-name", default="pickup_test_tube"); p.add_argument("--config", type=Path); p.add_argument("--output", type=Path, required=True); p.add_argument("--calibration", type=Path); p.add_argument("--observation-visibility", choices=("rgb_public", "privileged")); p.add_argument("--privileged", action="store_true"); a = p.parse_args()
    if a.observation_visibility == "privileged": a.privileged = True
    if a.observation_visibility == "rgb_public" and a.privileged: p.error("conflicting visibility flags")
    if a.privileged and a.calibration is None: p.error("--calibration is required for privileged mode")
    config = json.loads(a.config.read_text()) if a.config else {}
    candidate_kind = PRIVILEGED_CANDIDATE_KIND if a.privileged else CANDIDATE_KIND
    evidence_policy = PRIVILEGED_EVIDENCE_POLICY if a.privileged else EVIDENCE_POLICY
    observation_visibility = "privileged_agent" if a.privileged else "rgb_public"
    zeva = config.get("zeva", {})
    vla_host, vla_port = str(zeva.get("host", "127.0.0.1")), int(zeva.get("port", 5581))
    gateway_port = int(config.get("gateway_port", 5582))
    root, out = a.scenes_root.resolve(), a.output.resolve()
    if out.exists() and any(out.iterdir()): raise SystemExit(f"output already exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((root / "scenes.json").read_text())
    if not isinstance(raw, list) or not raw: raise SystemExit("scenes.json must contain a non-empty list")
    entries, languages, task_ids, evaluator_ids, schemas = [], set(), set(), set(), set()
    for index, item in enumerate(raw):
        scene = Path(item["scene"]).resolve(); task_path = Path(item["task"]).resolve(); mapping = Path(item.get("mapping", scene / "mapping.json")).resolve(); metadata_path = scene / "metadata.json"; model = scene / "model.mjb"
        for required in (scene, task_path, mapping, metadata_path, model):
            if not required.exists(): raise SystemExit(f"missing scene input: {required}")
        task, metadata = json.loads(task_path.read_text()), json.loads(metadata_path.read_text())
        name, language = task.get("task_name", task.get("name")), task.get("instruction", task.get("language"))
        if name != a.task_name or metadata.get("task_name") != a.task_name: raise SystemExit(f"task identity mismatch in {scene}")
        if not isinstance(language, str) or not language.strip(): raise SystemExit(f"missing task instruction in {task_path}")
        languages.add(language.strip()); task_ids.add(str(task.get("task_id", task.get("name", a.task_name)))); schemas.add(str(task.get("schema_version"))); evaluator_ids.add((task.get("success", {}).get("evaluator"), task.get("failure", {}).get("evaluator")))
        input_paths = {"model": model, "mapping": mapping, "task": task_path, "metadata": metadata_path,
                       "model_contract": Path(__file__).resolve().parents[2] / "robots/arx/manifests/task7_model_a.yaml"}
        for optional in ("scene.xml", "reset_state.npz"):
            if (scene / optional).exists(): input_paths[optional] = scene / optional
        scene_id = f"episode_{int(item.get('episode', index)):06d}"
        if metadata.get("scene_id") != scene_id or any(row["scene_id"] == scene_id for row in entries):
            raise SystemExit(f"duplicate or inconsistent scene identity: {scene_id}")
        entries.append({"scene_id": scene_id, "scene": str(scene), "split": "train", "hashes": {n: digest(x) for n, x in input_paths.items()}})
    if len(languages) != 1 or len(task_ids) != 1 or len(evaluator_ids) != 1 or len(schemas) != 1: raise SystemExit("scene registry contains mixed task/language/evaluator identities")
    language = next(iter(languages)); normalized = " ".join(language.casefold().split()); contract = {"suite": "arx_mujoco_task7", "task": a.task_name, "language": language, "normalized_language": normalized, "language_sha256": canonical({"language": normalized}), "task_ids": sorted(task_ids), "evaluator_ids": list(next(iter(evaluator_ids))), "task_schema": next(iter(schemas)), "schema_version": "arx.task.v1", "evaluator_contract_sha256": digest(Path(__file__).resolve().parents[2] / "robots/arx/evolution_result.py")}; contract["digest"] = canonical(contract)
    for entry in entries: entry["reset_seed"] = 17 + int(entry["scene_id"].split("_")[-1])
    heldout = [{**entry, "split": "heldout", "trial_index": i, "reset_seed": 1000 + i} for i, entry in enumerate(entries * 2)]
    registry = {"schema_version": 1, "task_contract": contract, "entries": entries, "heldout": heldout}
    safety = dict(SAFETY_LAYER)
    rollout_seeds, heldout_seeds = [17 + i for i in range(10)], [1000 + i for i in range(20)]
    runtime_limits = {"max_steps": 600, "max_decisions": 64, "max_recoveries": 4, "operation_timeout_s": 360, "critic_timeout_s": 5, "idle_agent_timeout_s": 600, "lease_timeout_s": 60, "shutdown_timeout_s": 15}
    # A bounded Zeva chunk may contain many synchronous inference calls.  The
    # reconciliation window must cover that work; 15 seconds caused the
    # runner to cancel valid partially-completed operations before the gateway
    # could publish their terminal result.
    runner_limits = {"startup_timeout_s": 180, "episode_timeout_s": 1800, "reconciliation_timeout_s": 300, "shutdown_timeout_s": 30, "heartbeat_interval_s": 5, "max_tool_attempts": 128, "max_agent_calls": 64, "max_recovery_agent_calls": 4, "max_contract_retries": 2}
    agent_settings = {"planner_type": "api", "model": "openai-chat:gpt-5.6-terra", "reasoning_effort": "medium", "max_tokens": 4096, "max_turns": 4, "timeout_s": 300, "credential_env": "CODEX_API_KEY"}
    (out / "runtime-limits.json").write_text(json.dumps(runtime_limits, indent=2) + "\n")
    (out / "runner-limits.json").write_text(json.dumps(runner_limits, indent=2) + "\n")
    (out / "agent-settings.json").write_text(json.dumps(agent_settings, indent=2) + "\n")
    critic_limits = {"python": sys.executable, "max_history": 60,
                     "max_evaluation_ms": 5000, "startup_timeout_s": 10,
                     "memory_bytes": 2147483648, "cpu_seconds": 60,
                     "scratch_bytes": 1048576, "image_width": 320,
                     "image_height": 240, "max_message_bytes": 2097152}
    (out / "critic-runtime-limits.json").write_text(json.dumps(critic_limits, indent=2) + "\n")
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/deployment/run_arx_evolution_rollout.py"), "--campaign-root", "{campaign_root}", "--logical-id", "{logical_id}", "--attempt-index", "{attempt_index}", "--generation", "{generation}", "--task", "{task}", "--seed", "{seed}", "--policy-rng", "{policy_rng}", "--bundle", "{candidate_path}", "--bundle-sha256", "{bundle_sha256}", "--baseline-mode", "{baseline_mode}", "--scene-registry", str(out / "scene-registry.json"), "--vla-host", vla_host, "--vla-port", str(vla_port), "--gateway-port", str(gateway_port), "--runtime-limits", str(out / "runtime-limits.json"), "--runner-limits", str(out / "runner-limits.json"), "--model-contract", str(Path(__file__).resolve().parents[2] / "robots/arx/manifests/task7_model_a.yaml"), "--critic-runtime-limits", str(out / "critic-runtime-limits.json"), "--agent-settings", str(out / "agent-settings.json"), "--output-dir", "{output_dir}", "--result-file", "{result_file}", "--heartbeat-file", "{heartbeat_file}", "--evidence-policy", evidence_policy]
    command[command.index("--gateway-port") + 1] = "{gateway_port}"
    if a.calibration: command += ["--calibration", str(a.calibration.resolve())]
    if a.privileged: command += ["--privileged"]
    atomic_write_json(out / "prompt-contract.json", {"cluster": CLUSTER_SYSTEM_PROMPT, "diagnosis": DIAGNOSIS_SYSTEM_PROMPT, "proposal": PROPOSAL_SYSTEM_PROMPT, "visibility": observation_visibility})
    atomic_write_json(out / "feature-catalog.json", {"schema_version": "arx.privileged.feature_catalog.v1", "features": ["privileged.selected.target_gripper_distance_m", "privileged.interaction.gripper_closed", "privileged.interaction.grasped", "privileged.interaction.progress", "privileged.interaction.lift_m", "privileged.interaction.success"]})
    tool_catalog = _tool_catalog()
    manifest = {"campaign_id": "arx-pickup-test-tube", "environment": "arx_mujoco", "task": a.task_name, "generation": 0, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "prompt_sha256": digest(out / "prompt-contract.json"), "model": "gpt-5.6-terra", "tool_catalog_sha256": tool_catalog["catalog_sha256"], "rollout_seeds": rollout_seeds, "heldout_seeds": heldout_seeds, "policy_rng_by_seed": {str(s): s for s in rollout_seeds + heldout_seeds}, "baseline_mode": "strict_pure_vla", "active_bundle_sha256": None, "parent_bundle_sha256": None, "safety_layer": safety, "expected_rollouts": 10, "expected_heldout": 20, "initial_logical_slots": 1, "continuous_logical_slots": 1, "maximum_logical_slots": 2, "maximum_api_concurrency": 1, "episode_timeout_s": 1800, "no_progress_timeout_s": 600, "max_infrastructure_attempts": 2, "runtime": {"candidate_kind": candidate_kind, "evidence_policy": evidence_policy, "observation_visibility": observation_visibility, "privileged": a.privileged, "calibration": str(a.calibration.resolve()) if a.calibration else None, "prompt_contract": "prompt-contract.json", "feature_catalog": "feature-catalog.json", "rollout_requires_environment_slot": False, "rollout_requires_api": False, "candidate_rollout_requires_api": True, "vla_service_mode": "external", "vla_host": vla_host, "vla_port": vla_port, "vla_maximum_inflight": 1, "task_contract": contract, "deployment_bootstrap_sha256": digest(Path(__file__).resolve().parents[2] / "robots/arx/deployment/runner.py"), "scene_registry": "scene-registry.json", "rollout_command": command, "same_seed_gate_rollout_command": command, "heldout_gate_kind": "heldout_20", "reuse_rollout_parent_evidence": True}}
    manifest["runtime"]["deployment_bootstrap_sha256"] = BOOTSTRAP_SHA256
    manifest["runtime"]["critic_runtime_limits"] = "critic-runtime-limits.json"
    manifest = CampaignManifest.from_dict(manifest).as_dict()
    atomic_write_json(out / "task-contract.json", contract)
    atomic_write_json(out / "tool-catalog.json", tool_catalog)
    atomic_write_json(out / "scene-registry.json", registry)
    atomic_write_json(out / "manifest.json", manifest)
    atomic_write_json(out / "campaign_contract.json", {**manifest, "task_contract": contract, "trial_registry": registry})
    print(out / "manifest.json")
if __name__ == "__main__": main()
