# 当前 sim bundle 真机迁移（2026-10-09）

当前执行入口为 [原 critic、仅预算适配版本](../experiments/arx-real2sim2real-budgetonly-20261009/README.md)。
原 critic 条件、12 步持续窗口、50 步 cooldown 和“张爪 → 保持 → VLA”顺序已恢复；
只有张爪预算从 20 增至 60 步。当前接口编译的恢复上限为 129 步。
55 项测试、dodo 静态预检与历史特征只读回放通过；本次尚未执行新真机 rollout，
控制器保持停止。下文保留各日期的历史部署快照。

---

# 历史 ARX 部署（2026-10-08）

当前正在进行 [GraspGen 真机验收](../experiments/arx-graspgen-live-20261008/README.md)。已实测完成 learned pose 的张爪预抓取，195 个执行步到位；完整夹紧和抬升尚未验收。现场已授权失能再使能；重新启用后，起始姿态和空夹爪张开实测通过，正在执行完整抓取验收。之前右腕反馈停止进展的根因尚未确定。

新配置与冻结契约位于 `docs/experiments/arx-graspgen-live-20261008/`；完整验收使用 `bundle-full-commission.json`、`grasp-config-full-commission.json`、`hardware-sdk-bounded.json` 和 `frozen-full-sdk/`。由 `commission_arx_pregrasp.py --full-grasp` 执行，须有人看护并明确授权运动；该范围不调用 VLA，不将未验证的 transfer 自动晋升为正式配置。

注意：下面保留的是 **10 月 7 日的历史部署记录**。特征 provider 和工具 schema 已变化，旧 SHA 命令不能直接用于当前源码。历史成功来自几何恢复；当前 GraspGen 已执行 learned pregrasp，但尚不能宣称完整抓取成功。

---

# 历史 ARX PickTube 部署（2026-10-07）

此页指向当前已验证配置。当前候选未获晋升；最近的任务成功是有人看护、分段续跑的结果。代码整理后的配置预检不等于新的真机 rollout 验收。

## 主机与环境

| 项目 | 当前配置 |
| --- | --- |
| 执行主机/仓库 | `dodo:/home/dodo/chenfu/Agentic-Embodied` |
| 备份开发仓库 | `aigc31:/home/dingxin/Agentic-Embodied` |
| 真机 Python | `/home/dodo/chenfu/.venv_arx_real/bin/python` |
| 相机/数据依赖 | `/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages`，通过 `PYTHONPATH` 复用 |
| ROS2 | `/opt/ros/jazzy/setup.bash`；ARX workspace `/home/dodo/chenfu/ARX_X5/ROS2/X5_ws/install/setup.bash` |
| 当前 VLA | Task7 Model A，`/mnt/hdd16t/chenfu/cosmos_models/arx_model_a_5task_iter5000_20260817`，本机 loopback `5583` |
| 抓取服务 | GraspGen loopback `18093`，候选推理已验证；当前几何恢复不执行其 learned pose |
| 原始实验数据 | `/mnt/hdd16t/chenfu/grasp_recovery/`，不进 Git |
| 任务终止 | 正确粉色试管抓取并抬升 ≥1 cm，连续五个新有效观测帧；无需归位后才判成功 |

此前提供的 chemistry RealData checkpoint 与当前 Task7 模型契约不兼容，不能仅替换权重路径运行。模型服务和机器人动作执行都在 dodo；本机 loopback 服务调用仍存在。

## 相机、单位与频率

- front=`260422272500`，left=`260422271945`，right=`260422275847`，映射来自现场 README。
- 三路 RGB 采集 640×480@15 Hz，模型/观测输入 320×240 RGB；front 与 right 提供对齐的毫米深度。
- 14D 为左右各六关节 + 一夹爪。关节为 rad；夹爪使用已校准的 native policy 坐标，不能当作米或弧度。
- 右夹爪发出的命令沿用 `+0.9` 预紧；反馈原样记录，不重复添加偏置。
- rollout 控制频率 15 Hz。设备 keepalive 与控制频率是不同参数。
- RGB-D 与反馈使用 dodo 单调时钟；当前门限为 age≤150 ms、skew≤100 ms。每条命令分别保存发送回执、到位反馈及接受模式。
- 部署关节包络来自历史轨迹及已冻结配置，并不等于已经核对完的机械硬限位。完整关节/夹爪碰撞几何仍未认证。

## 当前冻结文件

`EXPERIMENT=docs/experiments/arx-no-lift-timing-20261007`。以下 SHA 是文件 SHA，不是全部文件的语义 SHA；bundle 语义 SHA 单独列出。

| 输入 | 仓库相对路径 | 文件 SHA-256 |
| --- | --- | --- |
| bundle | `docs/experiments/arx-no-lift-timing-20261007/bundle.json` | `60545edef75720945975823d46f59180208b28aaa19d05e39e77932b5a076244` |
| 真机输入契约 | `docs/experiments/arx-no-lift-timing-20261007/frozen/real-input-contract.json` | `b37e6d78badb37b4e7e866154edee7d07a1cc8f82a51f20d66801768b400a044` |
| hardware | `docs/experiments/arx-target-visibility-20261007/hardware-wrist-observer.json` | `d6c40cc4ccec4ee687bc485e5a7f9632db4dd5ed90b0f31ba5276bc63b2e66a0` |
| 特征 provider | `robots/arx/deployment/picktube_rgbd_provider.py` | `7e807a1043f0c007f01dc27b53379c0cb65c8fbb12debb42fa2caa72f01c1630` |
| 抓取配置 | `docs/experiments/arx-no-lift-timing-20261007/grasp-config-held-tube.json` | `115d8b4e57bd01a8a79d6d115f0e81071526905aa98ca8e1a855df49b905d356` |
| FK | `robots/arx/manifests/real/dodo_right_controller_ee_fk.json` | `159964e1ac841d5490e6de9bbc00c2afdbc11c981076d5b5428bb68f5b2bbb86` |
| task | `robots/arx/manifests/pickup_test_tube.yaml` | `60b9db6251795f2eb7a458d4bbc01aa8c077e66e6e95245bd4e76b02d8c94c56` |
| model contract | `robots/arx/manifests/task7_model_a.yaml` | `51bec2a13e4dc0e4b6cdaed4dd17ba683a56a922b1ecff5d176a4d4b1c27e24c` |
| runtime limits | `docs/experiments/arx-no-lift-timing-20261007/runtime-limits.json` | `ed12e3cd8251f6f8aa9244c9ca18e0c5f2cdc3e811e39d028764e7768d35f1e6` |
| runner limits | `docs/experiments/arx-no-lift-timing-20261007/runner-limits.json` | `c946695d48cd307fc950a634294748ead7edfafc7fdd404a270a7b8517adb965` |

工具目录在 `$EXPERIMENT/frozen/tool-catalog.json`，其规范化语义 SHA 为 `b60a5187d7456e688ec9415093a51e8cc931dafa2a6b699f9fe0fa7a41b0f406`。bundle 语义 SHA 为 `a0cf09480fc15f1b5fbd835bf6d519f8e63a20317cc67ce47fa96c0c894914f8`。

预算为 600 物理步、64 决策、最多 4 次 recovery。规划抬升 2 cm，但成功条件在实测 ≥1 cm、五帧保持时终止。当前 no-lift dwell 为 48 帧。续跑保留原预算和源 journal SHA，不重新分配一整轮预算。

## 配置预检（不开硬件）

以下命令在仓库根目录执行：

```bash
ARX_PYTHON=/home/dodo/chenfu/.venv_arx_real/bin/python
EXPERIMENT=docs/experiments/arx-no-lift-timing-20261007
export PYTHONPATH=.:/home/dodo/chenfu/.venv_data_collect_py312/lib/python3.12/site-packages:${PYTHONPATH:-}

"$ARX_PYTHON" scripts/deployment/arx_real.py check \
  --hardware-config docs/experiments/arx-target-visibility-20261007/hardware-wrist-observer.json \
  --expected-hardware-sha256 d6c40cc4ccec4ee687bc485e5a7f9632db4dd5ed90b0f31ba5276bc63b2e66a0 \
  --task robots/arx/manifests/pickup_test_tube.yaml \
  --model-contract robots/arx/manifests/task7_model_a.yaml \
  --runtime-config "$EXPERIMENT/runtime-limits.json" \
  --bundle "$EXPERIMENT/bundle.json" \
  --tool-catalog "$EXPERIMENT/frozen/tool-catalog.json" \
  --real-input-contract "$EXPERIMENT/frozen/real-input-contract.json" \
  --expected-real-input-sha256 b37e6d78badb37b4e7e866154edee7d07a1cc8f82a51f20d66801768b400a044 \
  --feature-provider robots/arx/deployment/picktube_rgbd_provider.py \
  --expected-feature-provider-sha256 7e807a1043f0c007f01dc27b53379c0cb65c8fbb12debb42fa2caa72f01c1630 \
  --grasp-config "$EXPERIMENT/grasp-config-held-tube.json" \
  --expected-grasp-config-sha256 115d8b4e57bd01a8a79d6d115f0e81071526905aa98ca8e1a855df49b905d356 \
  --kinematics-calibration robots/arx/manifests/real/dodo_right_controller_ee_fk.json \
  --zeva-port 5583
```

## 新 rollout 与收尾

从原有 CLI 取得完整参数：

```bash
"$ARX_PYTHON" scripts/deployment/arx_real.py run --help
"$ARX_PYTHON" scripts/deployment/arx_real.py continue --help
"$ARX_PYTHON" scripts/deployment/arx_real.py finish --help
```

`run` 使用上述冻结文件，同时传 `--python "$ARX_PYTHON"`、`--runner-limits "$EXPERIMENT/runner-limits.json"`、新 `--output` 目录和 `--zeva-port 5583`。它的 SHA 参数名是 `--hardware-sha256`、`--real-input-sha256`、`--feature-provider-sha256`、`--grasp-config-sha256`，与 gateway 配置预检的 `--expected-*` 名字不同。新 rollout 前需 source ROS2、状态控制器运行并检查当前场景/起始反馈。

`continue` 是针对已审计的中断 closing checkpoint 的专用路径，仍固定 PickTube/Model A 和 600/64 源预算；它不是任意机器人/任务的通用暂停恢复。原始来源和全部后续 segment 必须逐个校验，审核 token 不延续。

成功后保持夹持等待卸载；收到明确卸载确认后，`finish --unloaded --execute` 从实时反馈核验归位并关闭控制器。卸载后可用 hardware-home 范围核验，不要求移走的试管仍然可见；正常任务观测继续要求真实特征。收尾参数与记录见[已完成记录](../experiments/arx-no-lift-timing-20261007/README.md#controller-cleanup)。

当前实测只支持这一组标定、夹爪和试管布局。引入新机械臂、移动相机或更换夹爪时，需要新的适配器/标定及验收，不能沿用这次成功作为新配置证明。

2026-10-08 homing update: `finish_arx_real_episode.py` now uses the pinned
PickTube 000015 return segment through `replay_arx_picktube_home.py`. It lifts
before retracting, preserves grippers until arm home, and verifies actual
arrival before opening the empty grippers and disabling. See the GraspGen live
experiment README for provenance, entry checks and commissioning evidence.

The latest `full-grasp-live02` was rejected at review before pregrasp motion.
Its new post-episode cleanup verified the task start and disabled the controller.
Physical homing validation in this attempt covered only near-home alignment,
not the entire taught return corridor; see `home-commission-result.json`.
