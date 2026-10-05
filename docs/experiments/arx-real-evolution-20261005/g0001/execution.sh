#!/usr/bin/env bash
set -eo pipefail
cd /home/dodo/chenfu/Agentic-Embodied
source /opt/ros/jazzy/setup.bash
source /home/dodo/chenfu/ARX_X5/ROS2/X5_ws/install/setup.bash
export PYTHONPATH=/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages:${PYTHONPATH:-}
PY=/home/dodo/chenfu/.venv_arx_real/bin/python
HW_SHA=f47a5847219cb3d83b0c1de12d902adb0461f475d65508efc165f1764ccea836
INPUT_SHA=982a0ed0ca1f5b0ea5331b0d5a699d79efeaecc4d52b902fc520f569a5024d34
PROVIDER_SHA=274dbb5f628f6b5e7fd85104dae68b2ad2749201c9d5a622c6c41c06fb327f2b
ARX_EVOLUTION_ARM="${ARX_EVOLUTION_ARM:-parent}"
case "$ARX_EVOLUTION_ARM" in
 parent)
  BUNDLE=robots/arx/manifests/real/proposed_retention_candidate_bundle.json
  INPUT_CONTRACT=robots/arx/manifests/real/dodo_retention_real_input_contract.json
  TRIAL_OUTPUT=runs/arx_real_evolution_20261005/g0001/parent
  ;;
 candidate)
  BUNDLE=docs/experiments/arx-real-evolution-20261005/g0001/candidate.json
  INPUT_CONTRACT=docs/experiments/arx-real-evolution-20261005/g0001/real-input-contract.json
  INPUT_SHA=840a439dcba616f2de1016d7ac0e4ffad2c964295e98d662e49a3b68a760dac7
  TRIAL_OUTPUT=runs/arx_real_evolution_20261005/g0001/candidate
  ;;
 *) exit 2 ;;
esac
case "${1:-check}" in
 check)
  "$PY" scripts/deployment/serve_arx_real_gateway.py --check-config \
   --hardware-config robots/arx/manifests/real/dodo_picktube_hardware.json \
   --expected-hardware-sha256 "$HW_SHA" \
   --task robots/arx/manifests/pickup_test_tube.yaml \
   --model-contract robots/arx/manifests/task7_model_a.yaml \
   --runtime-config robots/arx/manifests/real/dodo_picktube_runtime_limits.json \
   --bundle "$BUNDLE" \
   --tool-catalog robots/arx/manifests/real/dodo_picktube_tool_catalog.json \
   --real-input-contract "$INPUT_CONTRACT" \
   --expected-real-input-sha256 "$INPUT_SHA" \
   --feature-provider robots/arx/deployment/picktube_rgbd_provider.py \
   --expected-feature-provider-sha256 "$PROVIDER_SHA" \
   --kinematics-calibration robots/arx/manifests/real/dodo_right_controller_ee_fk.json
  ;;
 audit)
  "$PY" scripts/deployment/audit_arx_live_observation.py --hardware-sha256 "$HW_SHA" --output "runs/arx_live_readonly_20261005/${2}.json"
  ;;
 stage)
  "$PY" scripts/deployment/stage_arx_picktube_start.py --hardware-sha256 "$HW_SHA" --execute-steps "${2}" --output "runs/arx_live_readonly_20261005/${3}.json"
  ;;
 run)
  "$PY" scripts/deployment/run_arx_real_bundle.py \
   --output "$TRIAL_OUTPUT" \
   --python "$PY" \
   --hardware-config robots/arx/manifests/real/dodo_picktube_hardware.json \
   --hardware-sha256 "$HW_SHA" \
   --task robots/arx/manifests/pickup_test_tube.yaml \
   --model-contract robots/arx/manifests/task7_model_a.yaml \
   --runtime-config robots/arx/manifests/real/dodo_picktube_runtime_limits.json \
   --runner-limits robots/arx/manifests/real/dodo_picktube_runner_limits.json \
   --bundle "$BUNDLE" \
   --tool-catalog robots/arx/manifests/real/dodo_picktube_tool_catalog.json \
   --real-input-contract "$INPUT_CONTRACT" \
   --real-input-sha256 "$INPUT_SHA" \
   --feature-provider robots/arx/deployment/picktube_rgbd_provider.py \
   --feature-provider-sha256 "$PROVIDER_SHA" \
   --kinematics-calibration robots/arx/manifests/real/dodo_right_controller_ee_fk.json \
   --zeva-host 127.0.0.1 --zeva-port 5583 --listen-port 8091
  ;;
 *) exit 2 ;;
esac
