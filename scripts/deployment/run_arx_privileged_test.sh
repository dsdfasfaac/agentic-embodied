#!/usr/bin/env bash
set -euo pipefail

# End-to-end ARX privileged-observation test harness.
# Required environment: conda zetta-mujoco, and the Cosmos/Zeva assets expected
# by start_zeva_arx_task7_server.sh. Set ARX_TEST_ROOT to override the output.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENES="${ARX_SCENES_ROOT:-$ROOT/runs/arx_pickup_test_tube_10_new}"
OUT="${ARX_TEST_ROOT:-$ROOT/runs/arx_privileged_test_$(date +%Y%m%d_%H%M%S)}"
VLA_PORT="${VLA_PORT:-5581}"
GATEWAY_PORT="${GATEWAY_PORT:-5582}"
PYTHON="${ARX_PYTHON:-$HOME/miniconda3/envs/zetta-mujoco/bin/python}"
CALIBRATION="${ARX_CALIBRATION:-$ROOT/runs/arx_gateway_trace_episode000000_attempt1/robot_calibration.json}"

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
[[ -f "$CALIBRATION" ]] || { echo "missing calibration: $CALIBRATION" >&2; exit 2; }
[[ -n "${CODEX_API_KEY:-}" ]] || {
  echo "missing CODEX_API_KEY: direct Role1Agent rollouts require the configured API credential" >&2
  exit 2
}
export OPENAI_API_KEY="$CODEX_API_KEY"
export OPENAI_BASE_URL="${CODEX_BASE_URL:-https://api.openai.com/v1}"
mkdir -p "$OUT"

echo "[1/4] starting Zeva Task7 server on port $VLA_PORT"
HOST=127.0.0.1 PORT="$VLA_PORT" "$ROOT/scripts/deployment/start_zeva_arx_task7_server.sh" \
  >"$OUT/zeva.log" 2>&1 &
VLA_PID=$!
cleanup() { kill "$VLA_PID" 2>/dev/null || true; wait "$VLA_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if "$PYTHON" - "$VLA_PORT" <<'PY' >/dev/null 2>&1
import socket, sys
s=socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1); s.close()
PY
  then break; fi
  kill -0 "$VLA_PID" 2>/dev/null || { cat "$OUT/zeva.log"; exit 3; }
  sleep 1
done

echo "[2/4] freezing privileged campaign"
"$PYTHON" "$ROOT/scripts/evolution/prepare_arx_campaign.py" \
  --scenes-root "$SCENES" --output "$OUT/campaign" --privileged

echo "[3/4] writing CandidateBundle critic/recovery rule"
"$PYTHON" - "$OUT/campaign/bundle.json" <<'PY'
import json, hashlib, sys
out=sys.argv[1]
def sha(v): return hashlib.sha256(v.encode()).hexdigest()
payload={
 "schema_version":1,"candidate_id":"arx-privileged-distance-recovery","generation":0,
 "parent_sha256":None,"diagnosis_sha256":sha("arx privileged smoke test diagnosis"),
 "causal_hypothesis":"closed gripper is far from the selected target",
 "mechanism_change":"open and nudge the end effector toward the gripper before VLA resumes",
 "validation_plan":"run each frozen pickup scene and verify privileged records and recovery events",
 "critic_rules":[{"rule_id":"far_closed_gripper","title":"Closed gripper is far from target",
  "feature":"privileged.selected.target_gripper_distance_m","operator":"gt","threshold":0.10,
  "dwell_steps":1,"cooldown_steps":8,"proposal":"interrupt",
  "evidence_ids":["privileged-observation"],"safety_only":False,
  "activation_conditions":[{"feature":"privileged.interaction.gripper_closed","operator":"eq","threshold":True}]}],
 "recovery_rules":[{"recovery_id":"open_nudge_resume","title":"Open gripper and move toward gripper",
  "trigger_rule_ids":["far_closed_gripper"],"precondition":"gripper is closed and target distance exceeds 0.10m",
  "steps":[{"tool":"arx.set_gripper","parameters":{"opening":1.0,"max_steps":15},"stop_when":"gripper open"},
           {"tool":"arx.move_eef","parameters":{"delta_xyz_m":[0.0,0.0,0.02],"frame":"tool","speed_m_s":0.01},"stop_when":"2cm move committed"},
           {"tool":"arx.review_reentry","parameters":{"observation_ids":["post-recovery"]},"stop_when":"reentry eligible"},
           {"tool":"arx.zeva","parameters":{"max_chunks":1,"reentry_token":"token-from-review"},"stop_when":"VLA resumed"}],
  "safety_constraints":["named tools only","gateway limits apply"],"stop_condition":"steps complete",
  "fallback":"stop_all_motion","evidence_ids":["privileged-observation"]}],
 "tool_plugin":None}
json.dump(payload,open(out,'w'),sort_keys=True,indent=2); print(out)
PY

echo "[4/4] running direct rollouts with run_arx_evolution_rollout.py"
"$PYTHON" - "$OUT/campaign/bundle.json" "$OUT/campaign" "$ROOT" "$VLA_PORT" "$GATEWAY_PORT" <<'PY'
import hashlib, json, pathlib, subprocess, sys
from zetta.evolution.jsonio import canonical_sha256, read_json
bundle=pathlib.Path(sys.argv[1]); campaign=pathlib.Path(sys.argv[2]); root=pathlib.Path(sys.argv[3])
vla_port, gateway_port=int(sys.argv[4]), int(sys.argv[5]); digest=canonical_sha256(read_json(bundle))
for i in range(10):
    ident=f"episode_{i:06d}"; out=campaign/"rollouts"/ident; out.parent.mkdir(parents=True, exist_ok=True)
    seed=17+i
    cmd=[sys.executable,str(root/"scripts/deployment/run_arx_evolution_rollout.py"),"--campaign-root",str(campaign),"--logical-id",ident,"--attempt-index","0","--generation","0","--task","pickup_test_tube","--seed",str(seed),"--policy-rng",str(seed),"--bundle",str(bundle),"--bundle-sha256",digest,"--baseline-mode","strict_pure_vla","--scene-registry",str(campaign/"scene-registry.json"),"--vla-host","127.0.0.1","--vla-port",str(vla_port),"--gateway-port",str(gateway_port+i),"--runtime-limits",str(campaign/"runtime-limits.json"),"--runner-limits",str(campaign/"runner-limits.json"),"--model-contract",str(root/"robots/arx/manifests/task7_model_a.yaml"),"--critic-runtime-limits",str(campaign/"critic-runtime-limits.json"),"--agent-settings",str(campaign/"agent-settings.json"),"--calibration",str(root/"runs/arx_gateway_trace_episode000000_attempt1/robot_calibration.json"),"--output-dir",str(out),"--result-file",str(out/"episode_record.json"),"--heartbeat-file",str(out/"heartbeat.jsonl"),"--evidence-policy","arx_privileged_v1"]
    print("running", ident, flush=True); subprocess.run(cmd, check=True)
print("completed 10 direct rollouts")
PY
echo "Manifest: $OUT/campaign/manifest.json"
echo "Bundle:   $OUT/campaign/bundle.json"
echo "Zeva log: $OUT/zeva.log"
echo
echo "Direct rollout artifacts: $OUT/campaign/rollouts"
