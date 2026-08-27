# MuJoCo Runtime 接入指南

`env_family: mujoco` 将 Gymnasium MuJoCo 环境及受约束的外部 reBot G1-D 场景接入
现有 Rollout Runtime。通用后端支持 rank-1 连续 `Box` action space、state/RGB
观测、动作 chunk、LocalRuntime、Ray EnvWorker 和显式三态 success。reBot 扩展另外
提供 `grasp`、`fallen` 两个哈希固定的 Agentic expert skill，以及由 Runtime 独立
计算的任务成功判据；它不是任意 MJCF 或任意 Python entrypoint 执行器。

## 安装

MuJoCo 是可选依赖，不进入基础安装：

```bash
python -m venv /path/to/mujoco-venv
/path/to/mujoco-venv/bin/python -m pip install -e ".[test,ray,mujoco]"
/path/to/mujoco-venv/bin/python -m pip check
```

在新的机器上可以使用部署脚本。脚本拒绝复用已有 venv，安装完成后会执行原生
physics + EGL probe：

```bash
PYTHON_BIN=/usr/bin/python3 \
WHEELHOUSE=/optional/path/to/preverified-wheels \
  scripts/deployment/install_mujoco_env.sh \
  /path/to/Agentic-Embodied-mujoco \
  /path/to/validation/envs/mujoco-python \
  /path/to/validation/runs/current
```

`WHEELHOUSE` 是可选项，只用于机器的默认 Python 镜像缺少项目既有依赖时；脚本会
先从该目录安装 wheel，并把 SHA-256 写入验证产物，之后仍执行完整依赖解析和
`pip check`。

EGL 路径还要求 NVIDIA 驱动及 GLVND 的 `libEGL.so.1` 前端可由动态加载器发现。
安装脚本会在原生 probe 前直接加载该库并给出明确错误。Ubuntu/Debian 主机通常由
`libegl1` 提供；如果验证账户不能修改系统包，应把经过校验的发行版软件包解压到
仓库外的验证目录，并在启动安装脚本、Python 或 Ray 前通过 `LD_LIBRARY_PATH`
暴露其 `usr/lib/<architecture>` 目录。不要把系统库或机器专用绝对路径提交进仓库。

项目 extra 固定 `mujoco==3.3.1` 与 `gymnasium==1.1.1`。不要与 LIBERO-Pro、
RoboCasa 或其他 simulator track 共用 Python 环境。

reBot 场景必须使用独立环境及其专用 extra：

```bash
python -m venv /path/to/rebot-mujoco-venv
/path/to/rebot-mujoco-venv/bin/python -m pip install -e ".[test,ray,mujoco-rebot]"
/path/to/rebot-mujoco-venv/bin/python -m pip check
```

`mujoco-rebot` 遵循场景归档内 `environment-sim.yml`，额外固定
`numpy==1.26.4`、`opencv-python-headless==4.10.0.84`，并安装 PyYAML、Pillow、
imageio-ffmpeg 和 OmegaConf。不要把它装进已使用 NumPy 2.x 验收的通用 MuJoCo
venv；该隔离可以避免外部场景中的二维 `numpy.cross` 与 NumPy 2.4+ 不兼容。

## 最小配置

State-only 环境不要求 GPU：

```yaml
env_family: mujoco
env_config:
  provider: gymnasium
  env_id: InvertedPendulum-v5
  observation_mode: state
  render_mode: null
  action_dim: 1
  chunk_size: 4
  process_isolation: false
  success_mode: none
```

EGL RGB 环境应启用子进程隔离，并通过每个 EnvSpec 的 resource hint 要求 GPU：

```yaml
env_family: mujoco
env_config:
  provider: gymnasium
  env_id: InvertedPendulum-v5
  observation_mode: rgb_state
  render_mode: rgb_array
  render_backend: egl
  image_width: 256
  image_height: 256
  action_dim: 1
  chunk_size: 4
  process_isolation: true
  success_mode: none
env_resource_hints:
  accelerator: true
```

完整的单卡 Ray 配置见
`rollout_runtime/config/presets/rtx4090_mujoco.yaml`。

## reBot G1-D + Dex1-1 抓取场景

### 准备外置资产

`reBot-DevArm-Grasp.zip` 不是静态 MJCF，而是用
`simulation/g1d_mujoco_env.py` 在运行时把 G1-D + Dex1-1 URDF 转成 MJCF，并加入
三张桌子、瓶子、相机、灯光和接触参数。不要把这个 98 MB zip、STL、权重或历史
输出提交到 Git。

先用受约束的准备脚本解包。下面的 SHA-256 对应本次提供的归档：

```bash
python scripts/deployment/prepare_rebot_g1d_assets.py \
  /path/to/reBot-DevArm-Grasp.zip \
  /absolute/path/to/rebot-g1d-assets \
  --expected-archive-sha256 \
  a4bb41da2a961567bd7802ab6d518da6b9fde5b2ac03d55abda03674ce701b02
```

脚本拒绝绝对路径、`..`、符号链接、超大归档和 SHA 不匹配，只提取两个固定
adapter、`geometric_grasp.py`、两个任务状态机、两份 G1-D 配置、URDF 及 URDF
实际引用的 30 个 mesh。本次归档由 265 个条目缩减为 38 个必要文件，约 17 MB；
YOLO/PT 权重、相机/实机驱动、`.pyc` 和 `outputs/` 均被排除。结果写入外置目录的
`asset-manifest.json`；每个场景的执行清单包含 37 个文件。

本次 bundle 的两个可复现 digest 为：

| 场景配置 | `asset_manifest_sha256` |
|---|---|
| `config/g1d_mujoco.yaml` | `a87424e45b2fd3491b713b7e59cb3a336c52b9b38b3e17813789e2f186af0b10` |
| `config/g1d_fallen_bottle_to_bin.yaml` | `ed6d9700423abbc8866a610081a0ea7e5c89e3f52a3fcdc4168b7c6563bc549b` |

任何 adapter、状态机、几何抓取代码、配置、URDF 或引用 mesh 的内容发生变化，
digest 都会变化；Runtime 和 task policy 都会独立重新计算并拒绝不匹配的 bundle。
`grasp` 固定加载 `simulation/g1d_mujoco_env.py`，`fallen` 固定加载
`simulation/g1d_mujoco_env_bin.py`，不能通过配置指定任意 Python entrypoint。

### Runtime 配置

下面配置使用立瓶抓取场景和 D435i 风格的 `overview` 相机：

```yaml
env_family: mujoco
env_config:
  provider: rebot_g1d
  env_id: reBot-DevArm-Grasp-v0
  asset_root: /absolute/path/to/rebot-g1d-assets
  asset_manifest_sha256: a87424e45b2fd3491b713b7e59cb3a336c52b9b38b3e17813789e2f186af0b10
  rebot_scene_config: config/g1d_mujoco.yaml
  rebot_randomize_bottle: true
  rebot_task: grasp
  observation_mode: rgb_state
  render_mode: rgb_array
  render_backend: egl
  camera_name: overview
  image_width: 640
  image_height: 480
  action_dim: 22
  chunk_size: 32
  clip_actions: true
  max_episode_steps: 2500
  core_form: per_slot
  process_isolation: true
  rpc_timeout_s: 900.0
  instruction: Grasp the upright water bottle and return it safely to ready.
  success_mode: info_key
  success_info_key: is_success
  return_all_frames: false

env_resource_hints:
  accelerator: true
```

`asset_root` 必须是绝对路径。reBot provider 总是启用 slot 子进程隔离，避免外部
URDF 转换代码、MuJoCo native 状态和 EGL context 进入 Gateway 主进程。任务、场景
配置和 digest 必须严格配对。`fallen` 可用 `overview` 观察抓取近景，也可用
`layout_overview` 观察完整三桌和 bin 布局。

动作是 22 维绝对关节 position target，顺序由每个 Observation 的
`extras.action_names` 返回，固定为：

1. `LZ_mt_Joint`、`LZ_it_Joint`、`Yaw_Joint`、`torso_Joint`；
2. 左臂 7 轴：shoulder pitch/roll/yaw、elbow、wrist roll/pitch/yaw；
3. 左 Dex1-1 两个 finger joint；
4. 右臂 7 轴，顺序同左臂；
5. 右 Dex1-1 两个 finger joint。

Runtime 使用从真实 MuJoCo joint range 读取的上下界做整块 action 预校验/裁剪。
`reset(seed=...)` 会按场景的 `bottle_xy_jitter_m` 确定性随机化瓶体；支持的 reset
options 只有 `randomize_bottle`、`bottle_position` 和
`bottle_quaternion_wxyz`。

### reBot 验证

在支持 EGL 的机器上执行真实 Runtime core 验证：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
python scripts/deployment/smoke_rebot_g1d.py \
  /absolute/path/to/rebot-g1d-assets \
  --manifest-sha256 \
  a87424e45b2fd3491b713b7e59cb3a336c52b9b38b3e17813789e2f186af0b10 \
  --steps 100 \
  --output /path/to/artifacts/rebot-g1d-smoke.json \
  --frame-output /path/to/artifacts/rebot-g1d-final.png \
  --video-output /path/to/artifacts/rebot-g1d-smoke.mp4 \
  --video-fps 25
```

验收至少检查：URDF→MJCF 成功、22 维 action descriptor、确定性 reset、有限状态、
100 个 Runtime steps、`640×480 uint8` RGB、子进程回收及 manifest 拒绝篡改。

真实 remote test 可用同一份资产 pin 重跑：

```bash
REBOT_G1D_ASSET_ROOT=/absolute/path/to/rebot-g1d-assets \
REBOT_G1D_ASSET_SHA256=a87424e45b2fd3491b713b7e59cb3a336c52b9b38b3e17813789e2f186af0b10 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
python -m pytest -q -m remote tests/runtime/test_rebot_g1d_remote.py
```

本次归档在 4090 上的验收结果如下：

| 检查 | 结果 |
|---|---|
| 专用环境 | Python 3.12、MuJoCo 3.3.1、Gymnasium 1.1.1、NumPy 1.26.4；`pip check` 无冲突 |
| 立瓶抓取配置 | `overview`，100/100 steps，state dim 131，`640×480 uint8 RGB`，进程已回收 |
| 立瓶最终 RGB 像素 | SHA-256 `127761e3f46ab3062cfce1d2e6199e4008322a8e6f962dc8886b7ba143a662b8` |
| 倒瓶/垃圾桶配置 | `layout_overview`，20/20 steps，state dim 131，`640×480 uint8 RGB`，进程已回收 |
| 倒瓶最终 RGB 像素 | SHA-256 `9dbf1751c2f9a06bc02724971fcff7fed6ab02caa1ee09bbc18196d82e9bc3b2` |
| 自动化 remote test | `tests/runtime/test_rebot_g1d_remote.py`：1 passed |

`camera_name` 只选择输出视角，不会改写外部场景定义的 MJCF camera 名；因此
`layout_overview` 可与场景内保留的 `overview` 同时使用。

### Agentic skill、Runtime 判据与 CLI

两个原始状态机不会作为 standalone baseline 直接冒充 Agent 实验。它们由
`RebotG1DSkillPolicyCore` 在同 seed 的私有 planning simulation 中编译成 22 维绝对关节目标，记录轨迹和 planner summary 的 SHA-256；实际 episode 只能沿下面链路推进：

```text
Zetta/Codex Planner
  -> MujocoToolkit.run_rebot_skill
  -> RuntimeGateway.policy_step
  -> RuntimeRolloutWorker
  -> RebotG1DSkillPolicyCore
  -> MujocoEnvCore
  -> RebotG1DSession
```

Planner 只能看到 `observe_rebot_scene`、`run_rebot_skill`、`finish` 和工具说明。每个
episode 的 skill 只能调用一次。skill 自身的 shadow-planning 成功不构成最终成功；
Runtime 从 MuJoCo body pose、TCP pose 和 contact 独立判定，并要求成功候选连续保持
25 个物理步：

- `grasp`：最大抬升大于 8 cm、瓶子仍与所选 Dex1-1 接触并越过桌面近边、TCP 回到
  fixed ready 位置 3 cm 内；
- `fallen`：最大抬升大于 8 cm、瓶子中心位于 bin 内部且最终不再与 gripper 接触。

本地 Runtime 可以直接由 `zetta` 启动；若仿真在 GPU 服务器，可先把 Runtime HTTP
端口通过 SSH 转发到本机，再传 `--runtime-endpoint`。远端 `asset_root` 由服务端解释，
不要求挂载到 Planner 主机：

```bash
zetta --env mujoco --planner codex \
  --task grasp \
  --asset-root /absolute/server/path/to/rebot-g1d-assets \
  --asset-manifest-sha256 \
  a87424e45b2fd3491b713b7e59cb3a336c52b9b38b3e17813789e2f186af0b10 \
  --seed 17 --camera overview \
  --image-width 320 --image-height 240 \
  --chunk-size 32 --max-episode-steps 2500 \
  --video-fps 4 --runtime-timeout-s 900 \
  --planner-timeout-s 1800 \
  --runtime-endpoint http://127.0.0.1:18710 \
  --output-dir /path/to/run
```

`--planner-timeout-s` 同时设置 Codex MCP 的 `tool_timeout_sec`，避免长时间物理工具被
Codex 默认工具超时提前切断。`return_all_frames: false` 仍保留每个物理步的 reward、
info、success 和 termination 审计，只在每个 action chunk 末渲染/传输一帧，避免
逐步 EGL+PNG 成为主要瓶颈。

### 完整 Agentic 实验结果

在 4090 上使用 seed 17、同一哈希资产和远程 Runtime 完成了两项
真实任务。下表数据来自 `agentic-summary.json`，而不是原始脚本的 stdout：

| 任务 | Runtime 结果 | 物理步 / policy chunk | 关键独立判据 | MP4 |
|---|---:|---:|---|---|
| `grasp` | success，正常 terminated | 1668 / 53 | 最大抬升 0.1508 m；payload retained；ready 误差 0.00682 m；25/25 hold | H.264，320×240，4 fps，54 帧，13.5 s；SHA-256 `fccf48ba1999a9ff0967ddb9f09fc0ead88b772a57be1373d85860a88b312f36` |
| `fallen` | success，正常 terminated | 1737 / 55 | 最大抬升 0.1976 m；deposited；最终无 gripper contact；25/25 hold | 近景 H.264，320×240，4 fps，56 帧，14.0 s；SHA-256 `81f46c9fa991bbbe57735b89ff181ebca9a00c9e27445c9d40b9e12380b13a0b` |

两次 Planner transcript 均记录 3 个工具调用（observe、skill、finish）、1 个 physical
action、完整 terminal event 和 `finish.status=success`。另保留一份 fallen
`layout_overview` 全景 MP4，SHA-256 为
`9b0162f22236d87179d6354472905dc7052ff52671c3c882e2d06b931e255499`。
## Python 使用

所有调用仍通过 Gateway；`RuntimeGymEnv` 不直接持有 Gymnasium 环境：

```python
import numpy as np

from rollout_runtime.adapters.gym_adapter import RuntimeGymEnv
from rollout_runtime.api.messages import EnvSpecMsg
from rollout_runtime.launch.local import build_local_components

config = {
    "env_family": "mujoco",
    "env_config": {
        "env_id": "InvertedPendulum-v5",
        "observation_mode": "state",
        "render_mode": None,
        "action_dim": 1,
        "chunk_size": 4,
        "process_isolation": False,
        "success_mode": "none",
    },
}
runtime = build_local_components(config)
await runtime.start()
env = RuntimeGymEnv(
    runtime.gateway,
    EnvSpecMsg(env_family="mujoco", env_config=config["env_config"]),
)
observation, info = await env.reset(seed=7)
observation, reward, terminated, truncated, info = await env.step(
    np.zeros((1,), dtype=np.float32)
)
await env.close()
await runtime.gateway.stop()
await runtime.aclose()
```

输入动作可以是 `[action_dim]` 或 `[chunk, action_dim]`。backend 在执行任何
environment side effect 前拒绝错误 rank、错误维度、NaN/Inf；`clip_actions: true`
时按真实 action-space bounds 裁剪，否则越界直接报 `INVALID_ARGUMENT`。chunk 在
首次 termination 或 truncation 后立即停止，`executed_horizon` 只统计实际调用
`env.step()` 的次数，不统计 MuJoCo 内部 frame skip。

## 观测与 success

- `state`：ndarray 展平；数值 Dict 按 key 排序展平，并在
  `extras.observation_layout` 记录 path、shape、start、stop 和 dtype；
- `rgb`：只返回 `main_image`；
- `rgb_state`：同时返回两者；
- RGB 固定为配置尺寸的 `uint8 HWC`，不满足契约会在 build/reset 阶段失败；
- `return_all_frames: false` 默认只携带最终 observation，避免 chunk payload 膨胀。

`terminated` 表示 Gymnasium episode lifecycle，不再自动等同于任务成功：

- `success_mode: none`：`success=None`，只报告 episode return；
- `success_mode: info_key`：从 `info[success_info_key]` 读取标量布尔值；
- `success_mode: return_threshold`：累计 return 达到阈值后 success 锁存为 true。

`EvaluationAdapter.success_rate` 只以 `success is not None` 的有效 episode 为
分母。纯 return 任务的 `success_rate` 为 `null`，不会把跌倒 termination 或正常
time-limit truncation 伪造成成功/失败。

## EGL 与进程模型

RGB 默认 `process_isolation: true`。每个 slot 的子进程在 import Gymnasium/MuJoCo
前设置：

```bash
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

若 Python 报告无法加载 `libEGL.so.1`，先修复主机 GLVND frontend；仅存在
`libEGL_nvidia.so.*` vendor library 并不足以满足 PyOpenGL/MuJoCo 的加载链。
可用下面的无渲染命令先做诊断：

```bash
python -c 'import ctypes; ctypes.CDLL("libEGL.so.1")'
```

子进程独占 Gym env、renderer 和 EGL context；reset、step、render、close 通过有
超时的阻塞 IPC 执行。RPC 超时会终止该子进程，避免 worker 无限等待。state-only
模式默认同进程运行，减少进程和 IPC 开销。

`camera_name`、输出尺寸和 action shape 在 pool build 的冷 reset/render 阶段验证。
一个 pool 生命周期内 observation schema 发生变化会返回 `ENV_FAILURE`。

## 分阶段验证

验收脚本的每个 stage 输出可审计 JSON：

```bash
python scripts/deployment/smoke_mujoco_runtime.py native
python scripts/deployment/smoke_mujoco_runtime.py gymnasium
python scripts/deployment/smoke_mujoco_runtime.py core
python scripts/deployment/smoke_mujoco_runtime.py local --steps 100
python scripts/deployment/smoke_mujoco_runtime.py ray --concurrency 1
python scripts/deployment/smoke_mujoco_runtime.py ray --concurrency 4
python scripts/deployment/smoke_mujoco_runtime.py \
  soak --episodes 100 --steps-per-episode 100 --concurrency 4
```

真实测试使用独立 marker，默认基础测试不会 import MuJoCo：

```bash
python -m pytest -q tests/runtime/test_mujoco_backend.py
python -m pytest -q -m "remote and mujoco" tests/runtime/test_mujoco_remote.py
```

完整的环境目录、产物命名、阶段门禁和最终通过条件见
`docs/integration/mujoco-integration-plan.zh-cn.md`。

## 首版边界

- 允许 `provider: gymnasium`，以及固定路径、固定类名、manifest 校验的
  `provider: rebot_g1d`；不能通过配置导入任意 Python entrypoint；
- 只支持 `core_form: per_slot` 和一维连续 action space；
- 不提供通用 MJCF reward/termination/task-language 推断；
- 不暴露 model、contact 或任意 camera 等 privileged state；
- `InvertedPendulum-v5` 用于链路验收，不代表已有机器人策略与该动作空间兼容。

## 接入其他 MuJoCo 场景

接入前先判断场景属于哪一类。能封装为标准 Gymnasium 环境的场景优先走
`provider: gymnasium`；只有需要私有资产校验、非标准生命周期或基于 MuJoCo
内部状态计算任务成功时，才新增固定 provider。若还需要 Planner 自主调用技能，
则在环境接入之上再增加 policy 和 Agent 工具层。

| 场景条件 | 推荐路径 | 是否需要修改 Runtime |
|---|---|---:|
| 已注册为 Gymnasium env，action 为一维连续 `Box` | 通用 Gymnasium provider | 否 |
| 自定义 XML/URDF、控制循环或 contact 成功判据 | 固定 provider + Session | 是 |
| 需要模型或状态机产生 action chunk | 再增加 `PolicyInferenceCore` | 是 |
| 需要 Codex Planner 观察并执行任务 | 再增加 TaskAdapter/Toolkit/CLI | 是 |

### 路径一：复用 Gymnasium provider

场景包应安装在每个 Runtime worker 使用的 Python 环境中，并满足以下接口：

- `reset(seed=..., options=...) -> (observation, info)`；
- `step(action) -> (observation, reward, terminated, truncated, info)`；
- `action_space` 是 shape 为 `(N,)`、具有数值 `low/high` 的连续空间；
- state observation 是数值 ndarray，或只包含数值叶子的 mapping；
- RGB 模式下 `render()` 返回 `uint8 HWC` 的 RGB/RGBA 图像；
- `close()` 可以重复调用并释放 renderer、MuJoCo model/data 等资源。

第三方包若依靠 import 完成 Gymnasium 注册，可使用
`package.module:Environment-v0` 形式的 `env_id`，确保 spawned renderer
子进程也会导入该模块。示例：

```yaml
env_family: mujoco
env_config:
  provider: gymnasium
  env_id: my_mujoco_tasks.envs:PickCube-v0
  env_kwargs:
    model_path: /absolute/path/to/assets/pick_cube.xml
  observation_mode: rgb_state
  render_mode: rgb_array
  render_backend: egl
  camera_name: overview
  image_width: 320
  image_height: 240
  action_dim: 7
  chunk_size: 16
  max_episode_steps: 1000
  process_isolation: true
  success_mode: info_key
  success_info_key: is_success
  return_all_frames: false
```

`action_dim` 必须与运行时发现的 action-space shape 完全一致。RGB/EGL 推荐保持
`process_isolation: true`；每个 worker 节点必须能访问相同的绝对资产路径。
Gymnasium 路径不接受 `asset_root` 或 `asset_manifest_sha256`，因此需要强资产 pin
时应改用固定 provider。

成功语义有三种选择：环境能给出物理成功判据时，在 `info["is_success"]` 中返回
布尔值并使用 `info_key`；只有明确的累计回报阈值时才使用 `return_threshold`；纯
控制或链路测试使用 `none`，此时 success 保持 `null`。不要把 `terminated` 自动
解释为成功，因为跌倒、越界等失败也可能正常终止 episode。

### 路径二：新增固定 provider

固定 provider 应采用 reBot 接入相同的边界：配置只能选择仓库中显式注册的实现，
不能传入任意 Python 类名或脚本路径。建议按以下顺序实现。

1. **收敛并固定资产。** 编写 `scripts/deployment/prepare_<scene>_assets.py`，只复制
   运行所需的 XML/URDF、mesh、texture、配置和经审计代码。输出逐文件 SHA-256
   manifest 和总 digest；Runtime 启动时重新计算并拒绝缺失、额外或被篡改文件。
   大型资产、原始压缩包、录屏和运行产物必须留在仓库外。
2. **实现 Session。** 在 `rollout_runtime/backends/<scene>_session.py` 中封装一个
   同步、可关闭的场景对象。MuJoCo 和场景包必须延迟到 Session 构造时导入，确保
   EGL 环境变量先在 renderer 子进程中设置。
3. **扩展配置白名单。** 在 `MujocoEnvConfig` 中加入新的 provider 名和必要字段，
   对字段类型、绝对资产根、manifest、scene/task 配对、action dim、camera 及
   process isolation 做 fail-fast 校验；同时把字段显式投影到 `session_config()`。
4. **注册 Session factory。** 在 `create_mujoco_session()` 中使用固定字符串分支和
   lazy import 构造新 Session。不要使用配置驱动的 `importlib` entrypoint。
5. **实现独立成功判据。** 从实际 MuJoCo `data`、body/site pose、传感器和 contact
   计算成功，把标量布尔值写入 `info["is_success"]`。对瞬时接触类任务增加连续
   hold steps，并区分成功终止、失败终止和 time-limit truncation。

Session 不要求继承基类，但必须提供下面的最小协议：

```python
class MySceneSession:
    descriptor = {
        "action_shape": (ACTION_DIM,),
        "action_low": action_low_float32,
        "action_high": action_high_float32,
    }

    def reset(self, *, seed, options):
        return observation, info

    def step(self, action):
        return observation, reward, terminated, truncated, info

    def render(self):
        return rgb_uint8_hwc_or_none

    def close(self):
        ...
```

`MujocoEnvCore` 会负责 action chunk 校验、裁剪、逐步审计、success 锁存、首次
termination/truncation 后停止、observation schema 检查以及进程隔离。Session
不应绕过这些语义，也不应把内部 `MjModel`、完整 contact 或任意相机访问暴露给
Planner。

新增 provider 后，至少会修改以下位置：

| 文件 | 改动 |
|---|---|
| `rollout_runtime/backends/mujoco_env.py` | provider 字段、严格校验和 Session 配置投影 |
| `rollout_runtime/backends/mujoco_session.py` | 固定 provider factory 分支 |
| `rollout_runtime/backends/<scene>_session.py` | 资产加载、reset/step/render/close、成功判据 |
| `scripts/deployment/prepare_<scene>_assets.py` | 最小资产包与 manifest |
| `tests/runtime/test_<scene>_backend.py` | 无仿真依赖的契约、拒绝路径和清理测试 |
| `tests/runtime/test_<scene>_remote.py` | 真实 MuJoCo/EGL 验收，标记为 `remote` |

### 接入 policy 与 Agent

如果上层直接提供 action，可在 `RuntimeGateway.step` 路径结束，不需要新增 policy。
若动作来自模型、轨迹或受控状态机，则实现一个 `PolicyInferenceCore`，并完成：

- 用严格 dataclass 校验 `policy_config`，固定 `policy_family`、`policy_id`、action dim
  和 `actions_per_chunk`；
- `infer_batch()` 只返回 `[chunk, action_dim]` 的有限数值，不直接推进真实环境；
- 在 `rollout_runtime/backends/__init__.py` 的 `POLICY_BACKENDS` 和
  `build_policy_core()` 中注册固定 backend；
- 在 `rollout_runtime/core/policy_inference.py` 声明可混批参数。任何影响图、shape
  或执行语义的参数都必须进入 compatibility key；
- policy 规划结果和环境成功判据相互独立。policy 自报成功不能替代 Runtime
  从真实 episode 得出的 success。

需要 Codex Planner 时，再在 `robots/mujoco/` 增加或扩展任务适配器和工具：

```text
Planner -> Toolkit -> RuntimeGateway.policy_step
        -> RuntimeRolloutWorker -> PolicyInferenceCore
        -> MujocoEnvCore -> SceneSession
```

工具层只能调用 Gateway，不得持有或直接 step MuJoCo 环境。建议只暴露
`observe_<scene>`、一个受约束的物理技能工具和 `finish`；把物理技能工具加入
Planner 的 physical-tool 分类，并在 CLI 中把任务、相机、asset pin、Runtime
endpoint 和超时参数显式映射到 TaskAdapter。对于没有稳定低层 policy 的场景，
先完成显式 action 的 Runtime 接入，不要用伪动作或 standalone 脚本冒充 Agentic
实验。

### 分阶段验收清单

新场景至少按下面顺序验证，失败时保留 seed、effective config、Git SHA、资产 digest
和错误分类：

1. **资产与原生 MuJoCo：** manifest 可复算；模型 load/reset；固定 action 重放后
   qpos/qvel 全部 finite 且同 seed 可复现。
2. **渲染：** 在目标服务器用 EGL 输出目标尺寸 `uint8 HWC`；重复创建、reset、
   render、close 后没有 context 或子进程残留。
3. **Session/Core：** action shape、NaN/Inf、越界策略、observation schema、chunk
   early-stop、success/termination/truncation 和动态 slot 全部有测试。
4. **Local/Remote Runtime：** 显式 action 和 policy loop 都只经过 Gateway；验证
   单 session、并发 session、超时、非法 camera 和 worker 退出。
5. **任务验收：** 至少一个成功 episode 和一个失败/超时 episode；成功必须来自
   Runtime 独立判据，并保留关键物理量与连续 hold 计数。
6. **Agentic 验收：** transcript 能证明 observe→physical tool→finish，记录真实
   physical action 数、policy chunk 数、物理步数和 terminal event。
7. **可视化与回归：** 从同一次正式 episode 生成 MP4，记录编码、分辨率、fps、
   帧数和 SHA-256；随后运行 focused tests、simulator-free 契约、Ruff 和
   `git diff --check`。

完成标准不是“场景可以打开”或“脚本能跑”，而是资产可复现、动作确实通过
Runtime 推进、成功判据独立、资源可回收、Agent transcript 与 MP4/summary 指向
同一个正式 episode，并且默认 simulator-free 测试仍不导入 MuJoCo。
