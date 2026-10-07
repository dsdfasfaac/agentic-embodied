#!/usr/bin/env bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
GRASPGEN_ROOT=${GRASPGEN_ROOT:-/mnt/hdd16t/chenfu/grasp_recovery/GraspGen}
GRASPGEN_PYTHON=${GRASPGEN_PYTHON:-/mnt/hdd16t/chenfu/grasp_recovery/venv313/bin/python}
GRASPGEN_CONFIG=${GRASPGEN_CONFIG:?Set GRASPGEN_CONFIG to the downloaded model YAML}
test -x "$GRASPGEN_PYTHON"
test -f "$GRASPGEN_CONFIG"
test -f "$GRASPGEN_ROOT/grasp_gen/grasp_server.py"
export CUDA_VISIBLE_DEVICES=${ARX_GRASP_GPU:-1}
export PYTHONPATH="$GRASPGEN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$GRASPGEN_PYTHON" "$ROOT/scripts/deployment/serve_arx_graspgen_local.py" \
 --config "$GRASPGEN_CONFIG" --host 127.0.0.1 --port "${ARX_GRASP_PORT:-18093}"
