#!/usr/bin/env bash
set -euo pipefail

# Run the chemistry RealData ARX5 model on dodo, beside the robot gateway.
# This checkpoint is not compatible with the older Task7 server launcher.
ACTION="${1:-check}"
BUNDLE="${ARX_REALDATA_BUNDLE:-/home/dodo/chenfu/cosmos3_edge_arx5_inference_bundle}"
CHECKPOINT="${CHECKPOINT:-/mnt/hdd16t/chenfu/cosmos_models/arx5_chemistry_edge_stride2_gbs256_8gpu_2250/iter_000002250_ema_bf16_hf}"
VENV="${VENV:-/home/dodo/chenfu/cosmos-framework-edge-arx5/.venv}"
CONFIG_FILE="${CONFIG_FILE:-$BUNDLE/config/edge_arx5_stride2_runtime/config.dodo.yaml}"
ACTION_STATS_PATH="${ACTION_STATS_PATH:-$BUNDLE/config/edge_arx5_stride2_runtime/action_stats_meanstd.json}"
EDGE_MODEL_PATH="${EDGE_MODEL_PATH:-/mnt/hdd16t/chenfu/assets/Cosmos3-Edge-processor}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/hdd16t/chenfu/assets/Wan2.2_VAE.pth}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5580}"
COSMOS_ENV_FILE=/dev/null

export CHECKPOINT VENV CONFIG_FILE ACTION_STATS_PATH EDGE_MODEL_PATH WAN_VAE_PATH HOST PORT COSMOS_ENV_FILE

if [[ "$ACTION" == check || "$ACTION" == start || "$ACTION" == restart || "$ACTION" == foreground ]]; then
  for path in \
    "$BUNDLE/start_cosmos3_edge_arx5_server.sh" \
    "$BUNDLE/third_party/cosmos-framework/cosmos_framework/scripts/action_policy_server_realdata_arx5_edge.py" \
    "$VENV/bin/python" \
    "$CHECKPOINT/config.json" \
    "$CHECKPOINT/model.safetensors.index.json" \
    "$CONFIG_FILE" \
    "$ACTION_STATS_PATH" \
    "$EDGE_MODEL_PATH/config.json" \
    "$WAN_VAE_PATH"; do
    [[ -e "$path" ]] || { echo "missing ARX RealData model artifact: $path" >&2; exit 2; }
  done
  [[ -x "$VENV/bin/python" ]] || { echo "Cosmos Python is not executable: $VENV/bin/python" >&2; exit 2; }
  python3 - "$CHECKPOINT" "$ACTION_STATS_PATH" "$CONFIG_FILE" <<'PY'
import json
import sys
from pathlib import Path

checkpoint, stats_path, config_path = map(Path, sys.argv[1:])
index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
shards = set(index.get("weight_map", {}).values())
if not shards or any(not (checkpoint / shard).is_file() for shard in shards):
    raise SystemExit("checkpoint index is empty or references missing shards")
stats = json.loads(stats_path.read_text())
if (stats.get("action_dim"), stats.get("normalization"), len(stats.get("channel_names", []))) != (14, "meanstd", 14):
    raise SystemExit("action statistics must describe 14D meanstd actions")
config = config_path.read_text()
if "sample_stride: 2" not in config or "format_prompt_as_json: false" not in config:
    raise SystemExit("runtime config does not match the stride-2 chemistry model")
print(f"ARX RealData model artifacts present: {len(shards)} checkpoint shards")
PY
  if [[ "$ACTION" == check ]]; then
    echo "checkpoint=$CHECKPOINT"
    echo "endpoint=tcp://$HOST:$PORT (not started)"
    exit 0
  fi
fi

case "$ACTION" in
  start|status|stop|restart|tail|foreground)
    exec bash "$BUNDLE/start_cosmos3_edge_arx5_server.sh" "$ACTION"
    ;;
  *)
    echo "usage: $0 {check|start|status|stop|restart|tail|foreground}" >&2
    exit 2
    ;;
esac
