#!/usr/bin/env bash
set -euo pipefail

# Launch the self-contained Task7 server shipped by cosmos3-edge-arx5-inference.
# This intentionally does not use that bundle's legacy ARX5 launcher.
BUNDLE="${COSMOS_ARX_BUNDLE:-/data4/zhengyikai/cosmos3-edge-arx5-inference}"
FRAMEWORK="$BUNDLE/third_party/cosmos-framework"
PYTHON="${COSMOS_ARX_PYTHON:-$BUNDLE/.venv/bin/python}"
CHECKPOINT="${ZEVA_CHECKPOINT:-/data4/zhengyikai/ckpt1/model}"
CONFIG="${ZEVA_CONFIG:-/data4/zhengyikai/ckpt1/config/config.yaml}"
OUTPUT="${ZEVA_OUTPUT_DIR:-$BUNDLE/runtime/cosmos3_edge_arx_task7}"
EDGE_MODEL_PATH="${EDGE_MODEL_PATH:-$CHECKPOINT}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$BUNDLE/models/Wan2.2_VAE.pth}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5581}"

SERVER="$FRAMEWORK/cosmos_framework/scripts/action_policy_server_arx_task7_edge.py"
for path in "$SERVER" "$CHECKPOINT/model.safetensors.index.json" "$CONFIG" "$EDGE_MODEL_PATH/config.json" "$WAN_VAE_PATH"; do
  [[ -e "$path" ]] || { echo "missing required Task7 inference artifact: $path" >&2; exit 2; }
done
[[ -x "$PYTHON" ]] || { echo "missing Cosmos Python environment: $PYTHON" >&2; exit 2; }
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || {
  echo "no working NVIDIA driver/GPU is visible; refusing to start Cosmos inference" >&2
  exit 2
}

mkdir -p "$OUTPUT"
cd "$FRAMEWORK"
exec env -u PYTHONPATH \
  EDGE_MODEL_PATH="$EDGE_MODEL_PATH" \
  WAN_VAE_PATH="$WAN_VAE_PATH" \
  COSMOS_TRAINING=0 \
  "$PYTHON" -m cosmos_framework.scripts.action_policy_server_arx_task7_edge \
    --checkpoint-path "$CHECKPOINT" \
    --config-file "$CONFIG" \
    --action-normalization auto \
    --output-dir "$OUTPUT" \
    --host "$HOST" --port "$PORT" \
    --seed 42 --deterministic-seed \
    --action-chunk-size 32 --served-action-steps 32 \
    --num-steps 4 --conditioning-fps 15 \
    --guidance 1.0 --shift 10.0 --sigma-max 80.0 \
    --resolution 480 --max-state-echo-error 0.05
