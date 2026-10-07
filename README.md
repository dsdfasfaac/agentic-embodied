# Agentic-Embodied：ARX 真机 Critic–Recovery

基于 [Zetta-Embodiment](https://github.com/air-embodied-brain/Zetta-Embodiment)，将冻结 VLA + 时序 critic + 有预算的 recovery 部署到 ARX 双臂真机。当前任务是选择粉色试管、抓取并抬升至少 1 cm，保持五个新观测帧。

## 当前状态（2026-10-07）

- 模型、gateway、runner、ROS2 和三台 D405 均在 **dodo 本机**执行。
- bundle 的 critic 只提出中断；runner/gateway 按冻结的工具顺序、参数、预算执行恢复，并用实测状态审核重入。
- 已完成一次**有人看护的分段成功**：336 条动作均有到位记录，粉色试管抬升 13.863–14.877 mm 并保持五帧。卸载后已实测归位并关闭控制器。
- 当前成功采用 RGB-D 几何恢复。GraspGen 本地候选推理已验证，ARX 夹爪迁移尚未验收。
- 连续自主 rollout、当前候选的配对/泛化验证以及真机多代自动晋升仍待完成。上游的仿真成绩不代表本项目的真机成绩。

## 从这里开始

| 文档 | 内容 |
| --- | --- |
| [真机导航](docs/real_robot/README.md) | 模块分工、执行流程、部署命令分类 |
| [当前部署](docs/real_robot/current-deployment.md) | 当前 bundle、相机、单位、频率、冻结文件与复现入口 |
| [相比 Zetta 的改动](docs/real_robot/zetta-changes.md) | 固定提交对照、保留的设计、新增/修改的代码、验收差距 |
| [最近成功记录](docs/experiments/arx-no-lift-timing-20261007/README.md) | 五帧成功证据、分段限制、归位/停机审计 |
| [文档总索引](docs/README.md) | 当前真机、历史实验、仿真与 Sphinx 文档 |

## 代码目录

```text
robots/arx/
  gateway/              后端、相机/机械臂适配、工具审核、唯一动作写入与 journal
  deployment/           bundle 编译、真机特征、runner、受审计的续跑
  manifests/            任务、模型、硬件、标定和输入契约
  critics/              ARX 仿真/RGB critic 包与兼容路径
  environment.py        ARX MuJoCo 环境
robots/manipulation/    与机器人无关的抓取候选接口
zetta/evolution/        Zetta 进化流程及本项目的 ARX 扩展
rollout_runtime/        通用 rollout 基础设施、仿真后端及 ARX 模型适配
scripts/deployment/    部署命令；arx_real.py 是真机导航入口
scripts/evolution/     进化 campaign 与真机 evidence/shadow/compare 工具
scripts/maintenance/   Git 上游差异清单生成工具
docs/real_robot/       当前真机说明
docs/experiments/      按日期冻结的实验、配置和证据
integrations/          历史仿真交付快照，保留 vendor/ 和哈希
tests/                 硬件无关的单元/契约测试及可选运行测试
```

## 真机命令入口

在 dodo 仓库根目录、已配置的 ROS/Python 环境中执行：

```bash
python scripts/deployment/arx_real.py --help
python scripts/deployment/arx_real.py check --help
python scripts/deployment/arx_real.py run --help
python scripts/deployment/arx_real.py finish --help
```

这个入口转发到原有命令，保留参数和退出码。`check` 仅校验配置；`observe` 读传感器；`run` 执行动作；`continue` 用明确的只读/执行模式续跑；`finish` 在卸载确认后归位并失能。完整参数见[当前部署](docs/real_robot/current-deployment.md)。

模型权重、原始 RGB-D、场景和环境放在大盘，排除在 Git 外。Git 管理代码、冻结契约、标定来源和精简证据。

## 仿真与上游

- 原有 ARX 十场景 VLA：`scripts/deployment/run_arx_10_scenes.py`。
- 通用 MuJoCo：[集成说明](docs/integration/mujoco.md)。
- ARX 仿真进化：[集成计划](docs/integration/zeva-arx-evolution-plan.md)。
- 本轮整理前的根 README：[历史副本](docs/upstream/zetta-readme-before-arx.md)。其中的命令、成绩与 TODO 属于原文上下文。
- 最小开发测试环境：`python -m pip install -e '.[test]'`。真机 ROS、相机与模型使用 dodo 上各自已有环境，详见当前部署文档。
