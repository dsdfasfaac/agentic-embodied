#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 REPOSITORY VENV_DIR [RUN_DIR]" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

repository=$1
venv_dir=$2
run_dir=${3:-}
python_bin=${PYTHON_BIN:-python3}
wheelhouse=${WHEELHOUSE:-}

if [[ ! -f "${repository}/pyproject.toml" ]]; then
  echo "not a project root: ${repository}" >&2
  exit 2
fi
if [[ -e "${venv_dir}" ]]; then
  echo "refusing to reuse an existing environment: ${venv_dir}" >&2
  exit 2
fi

"${python_bin}" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
mkdir -p "$(dirname "${venv_dir}")"
"${python_bin}" -m venv "${venv_dir}"
venv_python="${venv_dir}/bin/python"

"${venv_python}" -m pip install --upgrade pip setuptools wheel
if [[ -n "${wheelhouse}" ]]; then
  if [[ ! -d "${wheelhouse}" ]]; then
    echo "wheelhouse does not exist: ${wheelhouse}" >&2
    exit 2
  fi
  "${venv_python}" -m pip install \
    --no-index \
    --find-links "${wheelhouse}" \
    --no-deps \
    openai-codex-cli-bin \
    openai-codex
fi
"${venv_python}" -m pip install -e "${repository}[test,ray,mujoco]"
"${venv_python}" -m pip check

if [[ -n "${run_dir}" ]]; then
  mkdir -p "${run_dir}"
  "${venv_python}" -m pip freeze > "${run_dir}/stage-1-pip-freeze.txt"
  "${venv_python}" -m pip check > "${run_dir}/stage-1-pip-check.txt"
  git -C "${repository}" rev-parse HEAD > "${run_dir}/git-revision.txt"
  if [[ -n "${wheelhouse}" ]]; then
    sha256sum "${wheelhouse}"/*.whl > "${run_dir}/stage-1-wheelhouse-sha256.txt"
  fi
fi

"${venv_python}" - <<'PY'
import ctypes
import ctypes.util
import json
import platform

import gymnasium
import mujoco
import numpy

try:
    ctypes.CDLL("libEGL.so.1")
except OSError as exc:
    raise SystemExit(
        "EGL frontend libEGL.so.1 is not loadable; install the distro GLVND "
        "libEGL package (libegl1 on Ubuntu/Debian) or expose a verified "
        "extracted runtime through LD_LIBRARY_PATH before rerunning this script"
    ) from exc

print(json.dumps({
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "mujoco": mujoco.__version__,
    "gymnasium": gymnasium.__version__,
    "numpy": numpy.__version__,
    "egl_frontend": "libEGL.so.1",
    "libraries": {
        name: ctypes.util.find_library(name)
        for name in ("EGL", "GL", "GLX", "OpenGL")
    },
}, indent=2, sort_keys=True))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader
fi

probe_args=(native)
if [[ -n "${run_dir}" ]]; then
  probe_args+=(--output "${run_dir}/stage-2-native-physics.json")
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
  "${venv_python}" "${repository}/scripts/deployment/smoke_mujoco_runtime.py" \
  "${probe_args[@]}"

echo "MuJoCo environment ready: ${venv_dir}"
