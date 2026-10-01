#!/usr/bin/env bash
set -euo pipefail

# Prepare and run a fresh ARX privileged campaign.
#
# Required environment:
#   CODEX_API_KEY       provider credential (never written to campaign files)
# Optional environment:
#   CODEX_BASE_URL      defaults to https://kitcoding.com/v1
#   ARX_SCENES_ROOT     defaults to runs/arx_pickup_test_tube_10_new
#   ARX_CALIBRATION     defaults to the checked-in trace calibration path
#   ARX_CAMPAIGN_ROOT   explicit output directory; otherwise timestamped
#   ARX_WORKERS         worker host list; defaults to the local hostname
#   ARX_MAX_GENERATIONS defaults to 10
#   ARX_START_ZEVA      set to 1 to start the local Zeva server in this process
#   ARX_VLA_GPU         CUDA device list for the local Zeva/VLA server, e.g. 0 or 2
#   ARX_SIM_GPU         CUDA device list for simulation workers, e.g. 1 or 0,1
#   ARX_SIM_GL          MuJoCo backend when ARX_SIM_GPU is set (defaults to egl)
#   ARX_SIM_EGL_DEVICE_ID EGL ordinal inside ARX_SIM_GPU (defaults to 0)
#
# The campaign supervisor and worker stay in the foreground. Stop with Ctrl-C.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ARX_PYTHON:-$ROOT/../miniconda3/envs/zetta-mujoco/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python)"
fi

SCENES="${ARX_SCENES_ROOT:-$ROOT/runs/arx_pickup_test_tube_10_new}"
CALIBRATION="${ARX_CALIBRATION:-$ROOT/runs/arx_gateway_trace_episode000000_attempt1/robot_calibration.json}"
CAMPAIGN="${ARX_CAMPAIGN_ROOT:-$ROOT/runs/arx_campaign_$(date +%Y%m%d_%H%M%S)}"
WORKERS="${ARX_WORKERS:-$(hostname)}"
MAX_GENERATIONS="${ARX_MAX_GENERATIONS:-10}"
VLA_HOST="${ARX_VLA_HOST:-127.0.0.1}"
VLA_PORT="${ARX_VLA_PORT:-5581}"
VLA_GPU="${ARX_VLA_GPU:-}"
SIM_GPU="${ARX_SIM_GPU:-}"
SIM_GL="${ARX_SIM_GL:-egl}"
SIM_EGL_DEVICE_ID="${ARX_SIM_EGL_DEVICE_ID:-0}"

validate_gpu_list() {
  local value="$1"
  [[ -z "$value" || "$value" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "invalid GPU list '$value' (expected e.g. 0 or 0,1)" >&2
    exit 2
  }
}

validate_gpu_list "$VLA_GPU"
validate_gpu_list "$SIM_GPU"
[[ "$SIM_EGL_DEVICE_ID" =~ ^[0-9]+$ ]] || {
  echo "invalid ARX_SIM_EGL_DEVICE_ID '$SIM_EGL_DEVICE_ID'" >&2
  exit 2
}

[[ -n "${CODEX_API_KEY:-}" ]] || { echo "CODEX_API_KEY is required" >&2; exit 2; }
[[ -d "$SCENES" ]] || { echo "scene root does not exist: $SCENES" >&2; exit 2; }
[[ -f "$SCENES/scenes.json" ]] || { echo "missing scenes.json: $SCENES" >&2; exit 2; }
[[ -f "$CALIBRATION" ]] || { echo "calibration does not exist: $CALIBRATION" >&2; exit 2; }
[[ ! -e "$CAMPAIGN" ]] || { echo "campaign output already exists: $CAMPAIGN" >&2; exit 2; }

export CODEX_BASE_URL="${CODEX_BASE_URL:-https://kitcoding.com/v1}"
export OPENAI_API_KEY="$CODEX_API_KEY"
export OPENAI_BASE_URL="$CODEX_BASE_URL"

ZEVA_PID=""
cleanup() {
  if [[ -n "$ZEVA_PID" ]] && kill -0 "$ZEVA_PID" 2>/dev/null; then
    kill "$ZEVA_PID" 2>/dev/null || true
    wait "$ZEVA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${ARX_START_ZEVA:-0}" == 1 ]]; then
  ZEVA_ENV=(env "HOST=$VLA_HOST" "PORT=$VLA_PORT")
  if [[ -n "$VLA_GPU" ]]; then
    ZEVA_ENV+=("CUDA_VISIBLE_DEVICES=$VLA_GPU")
  fi
  "${ZEVA_ENV[@]}" \
    "$ROOT/scripts/deployment/start_zeva_arx_task7_server.sh" \
    >"$ROOT/zeva-arx-${VLA_PORT}.log" 2>&1 &
  ZEVA_PID=$!
  echo "started local Zeva PID=$ZEVA_PID; log=$ROOT/zeva-arx-${VLA_PORT}.log"
fi

"$PYTHON" "$ROOT/scripts/evolution/prepare_arx_campaign.py" \
  --scenes-root "$SCENES" \
  --output "$CAMPAIGN" \
  --privileged \
  --calibration "$CALIBRATION"

echo "campaign=$CAMPAIGN"
echo "vla=$VLA_HOST:$VLA_PORT"
[[ -n "$VLA_GPU" ]] && echo "vla_gpu=$VLA_GPU"
[[ -n "$SIM_GPU" ]] && echo "sim_gpu=$SIM_GPU sim_gl=$SIM_GL sim_egl_device_id=$SIM_EGL_DEVICE_ID"

WORKER_ENV=(env)
if [[ -n "$SIM_GPU" ]]; then
  WORKER_ENV+=("CUDA_VISIBLE_DEVICES=$SIM_GPU")
  WORKER_ENV+=("MUJOCO_GL=$SIM_GL")
  if [[ "$SIM_GL" == egl ]]; then
    WORKER_ENV+=("MUJOCO_EGL_DEVICE_ID=$SIM_EGL_DEVICE_ID")
  fi
fi

exec "$PYTHON" "$ROOT/scripts/evolution/run_campaign.py" \
  --manifest "$CAMPAIGN/manifest.json" \
  --root "$CAMPAIGN/state" \
  --queue-root "$CAMPAIGN/queue" \
  --tool-catalog "$CAMPAIGN/tool-catalog.json" \
  --workers "$WORKERS" \
  --model gpt-5.6-terra \
  --poll-s 5 \
  --max-generations "$MAX_GENERATIONS" \
  --worker-command \
  "${WORKER_ENV[@]}" "$PYTHON" -m zetta.evolution.cli worker \
    --queue-root "{queue_root}" \
    --host "{host}" \
    --poll-s 2 \
    --concurrency 1
