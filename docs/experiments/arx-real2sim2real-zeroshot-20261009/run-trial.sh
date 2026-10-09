#!/usr/bin/env bash
set -euo pipefail
# Run on dodo after onsite confirmation, measured staging and camera audit.
cd /home/dodo/chenfu/Agentic-Embodied
EXP=docs/experiments/arx-real2sim2real-zeroshot-20261009
OUT=${1:-/mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/trial03}
exec bash /mnt/hdd16t/chenfu/grasp_recovery/adaptation-20261008/run-env31213.sh \
 scripts/deployment/run_arx_real_bundle.py \
 --output "$OUT" \
 --python /mnt/hdd16t/chenfu/grasp_recovery/adaptation-20261008/real-runtime31213/bin/python \
 --hardware-config docs/experiments/arx-graspgen-live-20261008/hardware-sdk-bounded.json \
 --hardware-sha256 18b47ac84ce7dbec7631d2b10c875e3cd3b938ae72e0374b9f97b2d70f94a431 \
 --task robots/arx/manifests/pickup_test_tube.yaml \
 --model-contract robots/arx/manifests/task7_model_a.yaml \
 --runtime-config "$EXP/runtime-limits.json" --runner-limits "$EXP/runner-limits.json" \
 --bundle "$EXP/bundle.json" --tool-catalog "$EXP/frozen/tool-catalog.json" \
 --real-input-contract "$EXP/frozen/real-input-contract.json" \
 --real-input-sha256 bda7b40ebc0535fa7d11094fce27dc44246d99a9c120a6ae3f0f5210a9daf42e \
 --feature-provider robots/arx/deployment/picktube_rgbd_provider.py \
 --feature-provider-sha256 ff6eb8faad5c77d58ffee82cf475320be238ca8481f183245d308474deab7a2d \
 --zeva-port 5583 --listen-port 8091
