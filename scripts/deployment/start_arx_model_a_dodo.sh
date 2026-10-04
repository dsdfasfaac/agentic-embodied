#!/usr/bin/env bash
set -euo pipefail

# Serve the H100 iter-5000 Task7 Model A export locally on dodo. This starts
# only a GPU model process; it does not open ROS, cameras, or robot control.
ACTION="${1:-status}"
PACKAGE="${ARX_MODEL_A_PACKAGE:-/mnt/hdd16t/chenfu/cosmos_models/arx_model_a_5task_iter5000_20260817}"
RUNTIME="${ARX_MODEL_A_RUNTIME:-$PACKAGE/runtime_dodo}"
CHECKPOINT="${CHECKPOINT:-$RUNTIME/model}"
CONFIG_FILE="${CONFIG_FILE:-$RUNTIME/config.dodo.yaml}"
COSMOS_REPO="${COSMOS_REPO:-/home/dodo/chenfu/cosmos-framework-edge-arx5}"
VENV="${VENV:-$COSMOS_REPO/.venv}"
LOG_DIR="${LOG_DIR:-$RUNTIME/server_logs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5583}"
TORCHDYNAMO_DISABLE=1
LAUNCHER="${ARX_TASK7_LAUNCHER:-/home/dodo/chenfu/inference/start_cosmos3_edge_task7_server.sh}"
export CHECKPOINT CONFIG_FILE COSMOS_REPO VENV LOG_DIR HOST PORT TORCHDYNAMO_DISABLE

if [[ "$ACTION" == start || "$ACTION" == restart ]]; then
  for path in "$RUNTIME/provenance.json" "$CHECKPOINT/config.json" \
              "$CHECKPOINT/model.safetensors.index.json" "$CONFIG_FILE" \
              "$COSMOS_REPO/cosmos_framework/scripts/action_policy_server_arx_task7_edge.py"; do
    [[ -f "$path" ]] || { echo "missing Model A runtime artifact: $path" >&2; exit 2; }
  done
fi
[[ -f "$LAUNCHER" ]] || { echo "missing Task7 launcher: $LAUNCHER" >&2; exit 2; }
exec bash "$LAUNCHER" "$ACTION"
