#!/usr/bin/env bash
set -euo pipefail

# Serve the H100 iter-5000 Task7 Model A export locally on dodo. No ROS or
# robot controller is opened. Keep the source package immutable.
ACTION="${1:-status}"
PACKAGE="${ARX_MODEL_A_PACKAGE:-/mnt/hdd16t/chenfu/cosmos_models/arx_model_a_5task_iter5000_20260817}"
EXPECTED_PACKAGE_CHECKSUMS_SHA256="${EXPECTED_PACKAGE_CHECKSUMS_SHA256:-cd8af4d2eff284cc2c139c5951b6e328ad18adb9e03da2f6d667d44637b0c5e6}"
RUNTIME="${ARX_MODEL_A_RUNTIME:-$PACKAGE/runtime_dodo}"
CHECKPOINT="${CHECKPOINT:-$RUNTIME/model}"
CONFIG_FILE="${CONFIG_FILE:-$RUNTIME/config.dodo.yaml}"
COSMOS_REPO="${COSMOS_REPO:-/home/dodo/chenfu/cosmos-framework-edge-arx5}"
PYTHON="${COSMOS_PYTHON:-$COSMOS_REPO/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$RUNTIME/server_logs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5583}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PID_FILE="$LOG_DIR/action_policy_server_task7_${PORT}.pid"
READY_FILE="$LOG_DIR/action_policy_server_task7_${PORT}.ready.json"
LOG_FILE="$LOG_DIR/action_policy_server_task7_${PORT}.log"
OUTPUT_DIR="$LOG_DIR/runtime_$PORT"

is_running() {
  [[ -s "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

check_inputs() {
  for path in "$RUNTIME/provenance.json" "$CHECKPOINT/config.json" \
              "$CHECKPOINT/model.safetensors.index.json" "$CONFIG_FILE" \
              "$COSMOS_REPO/cosmos_framework/scripts/action_policy_server_arx_task7_edge.py"; do
    [[ -f "$path" ]] || { echo "missing Model A runtime artifact: $path" >&2; exit 2; }
  done
  [[ -x "$PYTHON" ]] || { echo "missing Cosmos Python: $PYTHON" >&2; exit 2; }
  actual_checksums_sha="$(sha256sum "$PACKAGE/checksums/SHA256SUMS" | cut -d' ' -f1)"
  [[ "$actual_checksums_sha" == "$EXPECTED_PACKAGE_CHECKSUMS_SHA256" ]] || {
    echo "Model A package checksum-list SHA-256 differs" >&2; exit 2;
  }
  # The export's first checksum entry names the checksum file itself but
  # contains the digest of an empty file. Verify every actual package entry.
  (cd "$PACKAGE" && awk '$2 != "./checksums/SHA256SUMS" {print}' \
    checksums/SHA256SUMS | sha256sum -c - >/dev/null) || {
    echo "Model A package file SHA-256 verification failed" >&2; exit 2;
  }
  env -u LD_LIBRARY_PATH -u PYTHONPATH "$PYTHON" - \
    "$RUNTIME/provenance.json" "$CONFIG_FILE" "$CHECKPOINT/config.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
provenance = json.loads(Path(sys.argv[1]).read_text())
for path, key in ((Path(sys.argv[2]), "runtime_config_sha256"),
                  (Path(sys.argv[3]), "runtime_model_config_sha256")):
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance[key]:
        raise SystemExit(f"Model A generated runtime SHA-256 differs: {path}")
PY
}

case "$ACTION" in
  check)
    check_inputs
    echo "Model A runtime present: $CHECKPOINT"
    ;;
  status)
    if is_running; then
      if [[ -s "$READY_FILE" ]]; then
        echo "ready pid=$(<"$PID_FILE") endpoint=tcp://$HOST:$PORT"
      else
        echo "loading pid=$(<"$PID_FILE") log=$LOG_FILE"
      fi
    else
      echo "not running"
      exit 1
    fi
    ;;
  start)
    check_inputs
    mkdir -p "$OUTPUT_DIR"
    if is_running; then
      "$0" status
      exit 0
    fi
    rm -f "$PID_FILE" "$READY_FILE"
    # An empty LD_LIBRARY_PATH is interpreted as the current directory by
    # glibc and crashes this dodo Python stack during SciPy/Lark imports.
    # Unset it instead. Disable Dynamo to avoid a vision shape tracing bug.
    setsid bash -c '
      set -euo pipefail
      echo $$ > "$1"
      cd "$2"
      # jsonschema imports rfc3987_syntax after Cosmos has initialized other
      # packages; preload its Lark grammar and Diffusers package metadata
      # before Cosmos initializes native extensions (validated on dodo).
      exec env -u PYTHONPATH -u LD_LIBRARY_PATH \
        CUDA_VISIBLE_DEVICES="$3" TORCHDYNAMO_DISABLE=1 COSMOS_TRAINING=0 \
        "$4" -c '\''import rfc3987_syntax, diffusers, runpy; runpy.run_module("cosmos_framework.scripts.action_policy_server_arx_task7_edge", run_name="__main__")'\'' \
          --checkpoint-path "$5" --config-file "$6" \
          --output-dir "$7" --host "$8" --port "$9" \
          --seed 42 --deterministic-seed --action-chunk-size 32 \
          --served-action-steps 32 --num-steps 4 \
          --conditioning-fps 15 --guidance 1.0 --shift 10.0 \
          --sigma-max 80.0 --resolution 480 \
          --max-state-echo-error 0.05 --ready-file "${10}"
    ' model-a-server "$PID_FILE" "$COSMOS_REPO" "$CUDA_VISIBLE_DEVICES" \
      "$PYTHON" "$CHECKPOINT" "$CONFIG_FILE" "$OUTPUT_DIR" \
      "$HOST" "$PORT" "$READY_FILE" >>"$LOG_FILE" 2>&1 </dev/null &
    for ((elapsed=0; elapsed<300; elapsed++)); do
      if [[ -s "$READY_FILE" ]] && is_running; then
        "$0" status
        exit 0
      fi
      if [[ -s "$PID_FILE" ]] && ! is_running; then
        echo "Model A server exited; see $LOG_FILE" >&2
        tail -n 35 "$LOG_FILE" >&2
        exit 1
      fi
      sleep 1
    done
    echo "Model A server not ready after 300 seconds; see $LOG_FILE" >&2
    exit 1
    ;;
  stop)
    if is_running; then
      pid="$(<"$PID_FILE")"
      kill "$pid"
      for _ in {1..30}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
      done
      echo "stopped pid=$pid"
    else
      echo "not running"
    fi
    rm -f "$PID_FILE" "$READY_FILE"
    ;;
  tail)
    tail -n 100 -f "$LOG_FILE"
    ;;
  *)
    echo "usage: $0 {check|start|status|stop|tail}" >&2
    exit 2
    ;;
esac
