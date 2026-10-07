# Dodo local GraspGen deployment — 2026-10-07

## Verified

- Repository updated to the grasp recovery implementation. No ARX controller
  was started and no robot command was sent in this audit.
- Robotiq 2F-140 generator/discriminator loaded on dodo GPU 1. Loopback service:
  `http://127.0.0.1:18093`; VLA remains on GPU 0/port 5583.
- Five fresh D405 trios passed the camera audit, with the front pink-label
  depth around 409 mm, 56–60 valid label pixels and about 50 ms camera skew.
- Seed 42 inference on the retained 60-point label cloud returned eight
  proposals in 0.257 s. Model scores were 0.730–0.783. Those scores are not
  real grasp success probabilities. Converted model finger centres were
  3.8–16.1 mm from the observed label proxy.
- The selected PickTube `000015/data.npz` contains 54D state, 14D actions and
  timestamps, with RGB in separate folders. It has no depth array; its state
  was not combined with this new camera sample.

Evidence: `camera-audit.json`, `camera-grasp-proposals.json`,
`runtime-audit.json`. PNG originals remain on dodo under
`runs/arx_grasp_recovery_20261007/hashed-images`; each image SHA is recorded.
These are **camera proposal evidence**, not synchronized joint/path evidence.

Model SHA:
`6a378f83e3b691db76992d62fceb088b04d31d3827923d668f911e045e683acd`.
Full config/checkpoint SHA and source/package identities are in the evidence.

## Frame transfer draft

Official Robotiq geometry defines grasp +Z approach, X closing, and
`T_grasp_from_model_tcp` translation `[0, 0, 0.195]` metres. Raw model pose
translations refer to the gripper base, not the finger centre.

The ARX nominal tool uses +X approach and ±Y finger slides. The draft
`T_grasp_from_arx_tcp` rotation columns are `[model +Z, model +X, model +Y]`;
its translation is `[0, 0, 0.195]`. See `grasp-config-transfer-draft.json`.
This preserves the proposed finger centre and maps approach/closing axes.
The ARX nominal chain is the pinned `ac_one_nominal_chain.json`; its TCP is
`[0.12957, 0, 0.0137564]` in link-six coordinates.

This mapping is a **nominal frame draft**. Robotiq width is 0.1360 m; ARX
CAD finger travel is 0–0.044 m per finger, and policy gripper feedback is a
different native coordinate. Matching frame axes does not match finger meshes,
opening/current, swept volume or the visible label to the tube centre.
The draft therefore retains `motion_enabled=false` and
`learned_gripper_transfer_verified=false`; no physical transfer test occurred.

## Reproduce on dodo

Runtime and weights reside in `/mnt/hdd16t/chenfu/grasp_recovery`.
The separate `venv313` reads common packages, including Torch 2.10/cu130,
from the existing VLA environment via `.pth`. Additional versions are installed
only in `venv313`; the VLA package installation was not changed.

For a new environment, use the existing VLA interpreter to ensure matching
Python ABI, then install the pinned small dependencies without dependency
resolution:

```bash
cd /home/dodo/chenfu/Agentic-Embodied
export UV_CACHE_DIR=/mnt/hdd16t/chenfu/grasp_recovery/uv-cache
uv venv --python /home/dodo/chenfu/cosmos-framework-edge-arx5/.venv/bin/python \
 /mnt/hdd16t/chenfu/grasp_recovery/venv313
echo /home/dodo/chenfu/cosmos-framework-edge-arx5/.venv/lib/python3.13/site-packages \
 > /mnt/hdd16t/chenfu/grasp_recovery/venv313/lib/python3.13/site-packages/arx_torch.pth
uv pip install --no-deps --python /mnt/hdd16t/chenfu/grasp_recovery/venv313/bin/python \
 -r scripts/deployment/graspgen-dodo-requirements.txt
```

Use GraspGen source commit `2dd8852e1be60f5f9d277fafcc621835cdf59110`.
Apply the two checked-in patches **once to pristine source**:

```bash
cd /mnt/hdd16t/chenfu/grasp_recovery/GraspGen
patch -p1 < /home/dodo/chenfu/Agentic-Embodied/scripts/deployment/patches/graspgen-pointnet-lazy-ptv3.patch
patch -p1 < /home/dodo/chenfu/Agentic-Embodied/scripts/deployment/patches/graspgen-pointnet-cxx20.patch
cd pointnet2_ops
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDA_HOME=/usr/local/cuda \
 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 uv pip install --no-deps --no-build-isolation \
 --python /mnt/hdd16t/chenfu/grasp_recovery/venv313/bin/python .
```

The first patch delays the unused PTv3 import for the PointNet model; the second
uses C++20 for CUDA 13.3 / Torch headers. Neither changes model layers,
checkpoint contents or inference equations. Both modified source SHAs are
recorded in `runtime-audit.json`.

Download the Robotiq YAML, `_gen.pth` and `_dis.pth` from model repository
`adithyamurali/GraspGenModels` revision
`ec1ccbb5eec0680db669246ac312a3636f16ee43`, directory `checkpoints`, into
`/mnt/hdd16t/chenfu/grasp_recovery/models/checkpoints`. Verify against the
`checkpoint_sha256` dictionary in `camera-grasp-proposals.json`.

```bash
cd /home/dodo/chenfu/Agentic-Embodied
GRASPGEN_CONFIG=/mnt/hdd16t/chenfu/grasp_recovery/models/checkpoints/graspgen_robotiq_2f_140.yml \
 bash scripts/deployment/start_arx_graspgen_dodo.sh
```

Camera capture and replay use the existing real-robot Python 3.12 environment:

```bash
export PYTHONPATH=.:/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages
PY=/home/dodo/chenfu/.venv_arx_real/bin/python
"$PY" scripts/deployment/audit_arx_live_cameras.py --samples 5 \
 --output runs/grasp-audit/cameras.json --image-dir runs/grasp-audit/images
"$PY" scripts/deployment/probe_arx_grasp_camera.py \
 --camera-audit runs/grasp-audit/cameras.json --image-dir runs/grasp-audit/images \
 --expected-model-sha256 6a378f83e3b691db76992d62fceb088b04d31d3827923d668f911e045e683acd \
 --output runs/grasp-audit/proposals.json
```

Camera replay validates image SHA and retains original capture clocks. It
explicitly reports no joint feedback, no IK review and no motion eligibility.

## Next physical acceptance

1. Obtain current synchronized joint/current/TCP and RGB-D feedback with an
   on-site observer; review each candidate against the ARX joint/path envelope.
2. Verify the nominal frame transfer, actual opening and finger clearance with
   bounded pregrasp motion before enabling the learned transfer configuration.
3. Execute target-specific engage/close/lift; require pink-tube contact and
   at least 1 cm lift for five fresh frames.
4. Run matched parent/candidate trials and colour position swaps. Promote only
   the bundle that passes the measured task gate. This audit does not promote
   generation 2, train VLA weights or prove physical success.
