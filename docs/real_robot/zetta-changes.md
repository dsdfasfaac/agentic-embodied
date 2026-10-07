# 相比 Zetta-Embodiment，我们改了什么

## 比较基线

2026-10-07 读取并固定 [Zetta 上游提交 `21d7a3a`](https://github.com/air-embodied-brain/Zetta-Embodiment/tree/21d7a3afd6ca5a34de9d0350f0158252c5c55467)。本项目整理前的基线为 `91c6370910833e4ceea99e9e5fce0276388f36b1`，共同祖先为 [`d21d1a6`](https://github.com/air-embodied-brain/Zetta-Embodiment/tree/d21d1a68000beff69e564ac1784103f1c0dff8e2)。

[逐文件清单](zetta-source-comparison-20261007.json) 保存三方 Git blob 和变更类型：相对共同祖先，本项目新增 **762** 文件、修改 **37** 文件、删除 **0** 文件。大量新增文件是实验证据、测试和两份历史仿真交付快照；数量不能当作算法改动量。

当前上游还有 **105** 个只存在于上游的文件，涉及后来增加的环境/文档等。它们是尚未同步的上游更新，不表示我们主动删除了那些功能。这次整理没有合并上游。

复查或生成下一份清单：

```bash
git fetch https://github.com/air-embodied-brain/Zetta-Embodiment.git main:refs/remotes/zetta-upstream/main
python scripts/maintenance/compare_zetta_upstream.py \
  --local 91c6370 \
  --upstream 21d7a3afd6ca5a34de9d0350f0158252c5c55467 \
  --output /tmp/zetta-comparison-new.json
```

脚本只读本地 Git 对象，输出路径必须是新文件。这里的清单记录整理前源码；本次模块移动与导航文档由后续整理提交记录。

## 保留的核心设计

保持基础 VLA 权重冻结，改进 critic 与 recovery；保留 CandidateBundle、TemporalCritic、工具目录、恢复预算、shadow replay 和候选验证思路。上游定义的“critic 提议、决策方审核、恢复执行方按授权程序执行”角色边界仍成立。见[上游角色与进化协议](https://github.com/air-embodied-brain/Zetta-Embodiment/blob/21d7a3afd6ca5a34de9d0350f0158252c5c55467/README.md)。

## 主要改动

| 方面 | Zetta 上游对应机制 | 本项目的改变 | 代码与实际状态 |
| --- | --- | --- | --- |
| 执行环境 | 多仿真环境 backend/env actor | 增加 ARX MuJoCo 和真实 ARX 后端；真机复用相同 Backend/StepCommit 边界 | `robots/arx/environment.py`、`gateway/real_backend.py`；ARX sim 与真机均已运行 |
| 模型输入/动作 | 按各环境连接 VLA | 增加 Cosmos/Task7 Model A 的三路 RGB、14D state/action、chunk 与归一化契约；模型部署 dodo 本机 | `robots/arx/contracts.py`、`gateway/zeva.py`、`rollout_runtime/backends/cosmos3_edge_arx_policy.py`、模型部署脚本 |
| 真实观测 | 上游各环境有自己的观测与 evaluator，抓取适配也有 sensor-only 边界 | D405 RGB-D + 相机外参 + 新鲜关节/FK + 夹爪电流，形成夹爪关闭、距离、接触、抬升、抓取、成功六类特征 | `deployment/picktube_rgbd_provider.py`；接触与抓取是传感器代理量，非触觉真值 |
| 输入契约 | CandidateBundle/工具 schema 和冻结 artifact | 增加真机特征来源、设备身份/相机标定、时间戳、schema 与 SHA 核验；沿用示例 feature 名称 | `deployment/real_input.py`、`gateway/real_config.py`；`privileged.*` 名字保留兼容，真机不读取 MuJoCo 私有状态 |
| bundle 执行 | critic、Role1、recovery actor 的审核边界 | 结构化 bundle 编译为有序工具调用，runner 执行；gateway 强制顺序、参数、预算与 token，替换这条路径的固定恢复文案和仿真重入判断 | `deployment/bundle_program.py`、`real_runner.py`、`gateway/session_core.py`、`bundle_runtime.py` |
| 物理动作完成 | simulator action/step 与回传状态 | 分开记录 command_sent、命令回执、arrival_observed、到位接受模式和新鲜反馈；夹爪闭合允许实测稳定到位，保留 +0.9 预紧 | `gateway/real_backend.py`、`arx_ros2_device.py`；当前336条 rollout 命令均有到位记录 |
| 抓取恢复 | 上游已有 Contact-GraspNet/GraspGen 候选适配与恢复工具 | 增加机器人无关 TargetCloud/GraspProposalEngine；ARX propose→review→execute 的 pregrasp/engage/lift 路径、关节 IK 与 TCP 扫掠审核 | `robots/manipulation/grasp_proposals.py`、`gateway/grasp_recovery.py`；当前成功执行的是 tube_geometry |
| 观测缺失 | 原 TemporalCritic resolve_feature 和窗口计算 | 增加 unavailable_features；缺失打断 dwell/stagnation 窗口，保持 cooldown 处理，避免把遮挡直接当任务失败 | `zetta/evolution/critic.py`、`gateway/bundle_runtime.py`；长时间观测不可用仍可暂停 |
| 成功与收尾 | 各仿真环境 task evaluator、reset/shutdown | 真机观测满足正确目标、抬升≥1 cm、五帧保持即终止；之后持物等待卸载，实测归位，再停控制器 | `gateway/session_core.py`、`finish_arx_real_episode.py`；当前成功与收尾都已记录 |
| 中断续跑 | campaign/job 重试不等于物理姿态恢复 | 增加已耗预算、源 journal SHA、实测 engage、目标历史及 preload 的 checkpoint 核验；每次重新审核，不恢复旧 token | `deployment/grasp_continuation.py`；仅验收过 PickTube 专用分段路径 |
| 真机验证 | 同 seed 的 simulator paired gate、regression/held-out、promotion | 增加实际起始状态、目标距离、硬件/模型/预算身份和到位记录匹配；单次物理配对最多进入继续验证 | `scripts/evolution/audit_arx_real_evolution.py`；旧候选配对未成功、未晋升，当前成功候选尚未做配对 |
| 主机与数据 | 模型/环境服务及 Git 外的大资源 | dodo 本机模型+ROS2+相机执行；大权重/原始数据在 HDD，Git 保存代码、冻结配置和精简证据；保留驱动修复与启动曝光预热 | 部署脚本、`real_camera.py`、实验与硬件来源文件 |

上游抓取适配来源：[Contact-GraspNet](https://github.com/air-embodied-brain/Zetta-Embodiment/blob/21d7a3afd6ca5a34de9d0350f0158252c5c55467/robots/libero/contact_graspnet.py)、[GraspGen](https://github.com/air-embodied-brain/Zetta-Embodiment/blob/21d7a3afd6ca5a34de9d0350f0158252c5c55467/robots/libero/graspgen.py)。这些方法不是本项目首次引入；新增的是 ARX 真机点云、坐标/夹爪转接、审核与执行边界。

## 通用进化框架也改过

这些改动相对共同祖先存在，不能把整个 `zetta/` 目录称为“原封不动”：

- `candidate_artifacts.py`：统一解析结构化 bundle 与 ARX RGB package 的身份和路径。
- `campaign.py`、`gate_runner.py`、`store.py`、`supervisor.py`：加入 ARX artifact 分发、注册、验证、恢复与子代引用，保留原结构化 bundle 分支。
- `evidence_policy.py`、`trajectory.py`、`lifecycle.py`：增加 ARX 公共 RGB 证据白名单、索引与 episode/segment 归属。
- `stages.py`：新增 ARX package 提案路径、限制候选工具参数/特征名、压缩视觉索引，处理 provider 请求大小限制。
- `visual_artifacts.py`、`fault_injection.py`：增加事件/overview 采样和相应验证接口。
- `zetta/evolution/arx/`：ARX campaign 协调及工具适配。这主要是既有仿真/RGB package 扩展，尚不代表当前真机结构化 bundle 已由 campaign 自动多代演进。

此外已有通用 MuJoCo 后端与 RoboCasa mid-episode snapshot 扩展。它们属于仓库的其他改动，不是这次粉色试管成功的必要实现。

## 自进化实现与验收差距

| 阶段 | 当前情况 |
| --- | --- |
| 收集轨迹/失败、诊断、提出候选 | 有证据和候选谱系；真机试验主要由本对话中的 Codex 承担 learner |
| shadow replay 与真机配对 | 已有工具，2026-10-05 完成旧候选配对并拒绝晋升 |
| 当前候选抓取与成功终止 | 2026-10-07 分段续跑成功，期间修过观测与执行代码；不能当作冻结系统从头连续自主成功 |
| 当前候选回归/不同摆放/held-out | 尚未完成，不能报告泛化成功率 |
| 真机 campaign 自动多代晋升 | 尚未完成；已有框架接口不等于真实部署验证 |
| learned grasp 真机执行 | 本地 GraspGen 候选推理已验证；Robotiq→ARX 夹爪迁移仍未验收 |

当前有效结论是：真实传感器、bundle 恢复执行、正确目标抬升判定与收尾链路都已有运行证据。下一步应冻结当前系统，从初始状态连续复现，再做父代/候选配对和不同颜色位置验证。

## 本次整理的具体改动

1. 根 README 改为本项目入口，原文归档；新增 `docs/real_robot/` 当前状态、部署索引与此对照说明。
2. 新增 `scripts/deployment/arx_real.py`，提供 check/observe/run/continue/stage/finish 等导航，转发原参数/退出码。
3. `RealCoreFactory` 从服务脚本移到 `robots/arx/gateway/real_factory.py`；服务、commission 和 continuation 共用。服务脚本仍导出原类名，保持旧 import 可用。
4. 续跑 orchestration 移到 `robots/arx/deployment/grasp_continuation_runner.py`；原脚本保留 `main`/`run` 入口，PickTube 约束与预算逻辑保持原样。
5. 增加三方 Git 差异生成工具。历史 bundle/provider/标定、旧协议和 vendor 快照原样保留；本次整理本身不产生新真机成功证据。

验证与同步结果见 [整理验证记录](reorganization-validation-20261007.md)。
