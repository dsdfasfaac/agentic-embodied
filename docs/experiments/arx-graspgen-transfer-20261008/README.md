# GraspGen → ARX 真机适配第一阶段（2026-10-08）

**状态：候选转换、审核、受限预抓取入口已实现；尚无通过审核的 learned 抓取轨迹，也未执行真机运动。** Oct7 成功记录仍来自几何恢复，不能作为 GraspGen 迁移验收。

## 已实现

- dodo 本地 GPU1 / loopback `18093` GraspGen 服务已启动，模型身份及 SHA 已核对。模型、缓存和新增 SciPy overlay 均在 `/mnt/hdd16t/chenfu/grasp_recovery/`。
- 通用候选转换：`T_base_tcp = T_base_camera @ T_camera_model @ T_model_tcp`。不能将模型夹爪底座当作手指中心。
- ARX +X 前向、±Y 闭合；Robotiq 模型 +Z 前向、X 闭合，底座到 TCP 深度 0.195 m。使用下面的**名义转换**，未声称完成实物夹爪标定：

```text
T_model_tcp = [[0,1,0,0], [0,0,1,0], [1,0,0,0.195], [0,0,0,1]]
```

模型坐标约定来自 [GraspGen 官方夹爪说明](https://github.com/NVlabs/GraspGen/blob/main/docs/GRIPPER_DESCRIPTION.md)。轴向和中心匹配不等于手指形状、开度、接触点匹配。

- 显式可选的平行夹爪半周旋转变体：保持中心和接近方向，交换闭合轴符号；继承分数仅用于排序，未重新评估为新的成功概率。32 个审核名额包含前 16 个模型姿态及对应 16 个变体。每个变体单独审核。
- 审核目标距离、同步传感器、目标身份/漂移、IK、真机配置关节命令包络、每步关节增量、TCP 场景间隙、步数预算。执行前仍重新审核当前场景；已发送命令和实测到位分别记录。
- 角速度可冻结配置：默认 0.15 rad/s（15 Hz 下与原来每步 0.01 rad 相同），上限 0.30 rad/s。每步关节上限仍为 0.035 rad。
- `learned_pregrasp_commissioning=true` 可在 dedicated harness 中审核/执行张爪预抓取，同时 `learned_gripper_transfer_verified=false`。未验证时禁止 engage/lift，禁止 `--resume-vla-once`；普通 runner/factory 拒绝该验收配置。闭合、接触、抬升的现有观测条件继续适用于完成迁移验证后的正常执行。
- 离线工具可按 SHA 校验只读 journal 中的过去帧建立跨相机身份；拒绝重复/未来状态时间戳。不会发放真机 token。观测失败也写出失败审计。

## 验证与结果

dodo 已通过同一冻结包的静态预检：专用预抓取验收可加载，普通 factory 明确拒绝未验证迁移配置；硬件未打开。记录见 [dodo-preflight.json](evidence/dodo-preflight.json)。

本地 60 项测试通过：转换轴/位移、对称姿态、目标距离、真实关节包络、未验证阶段隔离、过去帧/SHA、失败审计、冻结 bundle/catalog/contract、既有 recovery/commissioning/输入契约。

真机留存来源：
`dodo:/mnt/hdd16t/chenfu/grasp_recovery/contact-lift-full-rollout-14/private/gateway/`。
本次工作目录：`/mnt/hdd16t/chenfu/grasp_recovery/adaptation-20261008/`。
NPZ/原始 RGB-D 和模型不进入 Git，审核 JSON 见 [evidence/summary.json](evidence/summary.json)。

| 留存观测 | 审核数 | 可行数 | 拒绝原因 |
| --- | ---: | ---: | --- |
| obs-1，0.15 rad/s | 32 | 0 | 目标偏差 8、行程 12、IK 3、步数 6、关节增量 3 |
| obs-100，0.15 rad/s | 32 | 0 | 步数 18、TCP 障碍 7、IK 7 |
| obs-100，0.30 rad/s | 32 | 0 | 目标偏差 4、TCP 障碍 9、关节增量 8、IK 11 |
| obs-200，过去帧身份重放 | 未进入推理 | 0 | 目标位置连续性不成立 |

不同请求 seed/index 已记录；上述采样不是同种子的严格配对消融，不能把拒绝数差异解释成效果提升。最初旧配置和无变体检查同样未找到可行候选，原始报告仍保留在 dodo 工作目录。

输入分别只有 **51/61 个粉色标签深度点**，是局部表面，不能代表完整透明试管形状。TCP 间隙也不认证完整机械臂/手指碰撞自由。

SciPy 1.18.0 导入/优化初始化曾间歇性报 `SystemError: error return without exception set`；根因尚未确定。隔离固定 SciPy 1.16.3、复用 NumPy 2.2.6 后，连续三次独立导入/求解及本次离线审核通过依赖初始化；VLA 和原有环境未改动。部署验收使用此 overlay，不宣称已修复原 SciPy 环境。

## 冻结文件

- `grasp-config-offline.json`：名义转换、不开运动，无半周变体。
- `grasp-config-half-turn-offline.json`：半周变体，不开运动，0.15 rad/s。
- `grasp-config-half-turn-rate-offline.json`：离线角速度探查，不开运动，0.30 rad/s。
- `grasp-config-commission.json`：受限张爪预抓取验收配置，0.15 rad/s；完整迁移验证标志仍为 false。
- `bundle-pregrasp.json` 与 `frozen/`、`freeze-report.json`：已冻结。为兼容 bundle 编译器保留 review_reentry/zeva 后缀；专用 harness 在预抓取后结束，未验证时不能执行该后缀。这是验收包，不是进化晋升候选。

当前几何部署文件没有被替换。当前模型 SHA：
`6a378f83e3b691db76992d62fceb088b04d31d3827923d668f911e045e683acd`。

## 离线重放

在 dodo 仓库根目录执行，模型服务须运行。没有相机/机械臂设备连接：

```bash
ARX_PYTHON=/home/dodo/chenfu/.venv_arx_real/bin/python
EXPERIMENT=docs/experiments/arx-graspgen-transfer-20261008
WORK=/mnt/hdd16t/chenfu/grasp_recovery/adaptation-20261008
SOURCE=/mnt/hdd16t/chenfu/grasp_recovery/contact-lift-full-rollout-14/private/gateway
export PYTHONPATH=.:$WORK/scipy-overlay:/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages:${PYTHONPATH:-}

"$ARX_PYTHON" scripts/deployment/probe_arx_grasp_snapshot.py \
  --snapshot "$SOURCE/grasp-sensors/obs-100.npz" \
  --observation "$WORK/obs-100-observation.json" \
  --grasp-config "$EXPERIMENT/grasp-config-half-turn-offline.json" \
  --hardware-config docs/experiments/arx-target-visibility-20261007/hardware-wrist-observer.json \
  --hardware-sha256 d6c40cc4ccec4ee687bc485e5a7f9632db4dd5ed90b0f31ba5276bc63b2e66a0 \
  --engine graspgen --max-candidates 32 --max-steps 240 \
  --output "$WORK/obs-100-new-request.json"
```

可加 `--history-journal "$SOURCE/journal.sqlite3"`，只使用选定观测之前的 SHA 已校验帧。输入没有真实可验证的目标时，保持失败。

## 真机验收准备与剩余工作

**还不能把本包作为 learned 完整 rollout 使用。** 先完成：

1. 取得可代表试管几何的目标点云，明确透明表面无法测得的部分；保持正确颜色/身份来源。不能把试管架混入目标云，也不能用复制标签点伪造更多观测。
2. 测量 ARX 手指中心/实际开度与抓取接触几何，核对名义 `T_model_tcp`。Robotiq checkpoint 的分数不直接等价于 ARX 成功率。
3. 在真实关节包络内获得可达姿态和无障碍路径；有必要时扩展真实碰撞模型/路径规划。当前候选全部拒绝，应先解决点云与可达性问题。
4. 再进行有现场看护的张爪预抓取验收，接着验证 engage/夹紧/抬升；每阶段保存实测证据，另建新配置冻结完整迁移状态。
5. 最后用同起始场景的 baseline/candidate 对照评估 recovery，再决定是否晋升。VLA 权重保持冻结不等于闭环已自进化成功。

下面是**后续可行候选和现场条件满足后的**预抓取入口，本次未运行。它会发送动作，期间控制器须运行；结束保留使能：

```bash
source /opt/ros/jazzy/setup.bash
source /home/dodo/chenfu/ARX_X5/ROS2/X5_ws/install/setup.bash
"$ARX_PYTHON" scripts/deployment/commission_arx_pregrasp.py \
  --bundle "$EXPERIMENT/bundle-pregrasp.json" \
  --grasp-config "$EXPERIMENT/grasp-config-commission.json" \
  --frozen "$EXPERIMENT/frozen" \
  --hardware-config docs/experiments/arx-target-visibility-20261007/hardware-wrist-observer.json \
  --hardware-sha256 d6c40cc4ccec4ee687bc485e5a7f9632db4dd5ed90b0f31ba5276bc63b2e66a0 \
  --retain-step-sensors --max-physical-steps 301 --execute \
  --output "$WORK/pregrasp-new-attempt"
```
