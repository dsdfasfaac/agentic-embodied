# ARX 真机导航

当前范围是 dodo 的双臂 ARX PickTube。跨机器人复用主要位于 Backend、设备协议和抓取候选接口；观测器、任务阈值、机械臂适配与标定仍需按机器人实现和验收。

## 运行流程

```text
冻结模型/任务/硬件/标定/bundle/工具目录/输入契约
  → 配置校验 → 同步传感器和起始状态核对
  → VLA 动作 → gateway 发送命令 → 实测到位 → 特征与 critic
  → critic 提出中断 → runner 按 bundle 恢复程序执行
  → 每步审核、预算扣除、观测与结果记录
  → 真机重入审核 → 新 VLA chunk，或成功终止
  → 卸载确认 → 实测归位 → 停控制器
```

观测不足会产生“未知/不可用”，不会直接变成接触、抓取失败或成功。长时间无法恢复观测仍可暂停执行。可恢复中断保留控制器使能；归位/失能是独立的收尾阶段。

## 修改时到哪里找

| 职责 | 主模块 | 修改原则 |
| --- | --- | --- |
| backend 接口与实测执行 | [backend.py](../../robots/arx/gateway/backend.py)、[real_backend.py](../../robots/arx/gateway/real_backend.py) | 保持已发送命令与实测到位分开；反馈时间戳与健康状态随动作记录 |
| ROS2/SDK 与相机适配 | [arx_ros2_device.py](../../robots/arx/gateway/arx_ros2_device.py)、[arx_x5_device.py](../../robots/arx/gateway/arx_x5_device.py)、[real_camera.py](../../robots/arx/gateway/real_camera.py) | 设备特有逻辑放适配器，频率/单位由硬件和任务契约固定 |
| gateway 组装 | [real_factory.py](../../robots/arx/gateway/real_factory.py) | 服务和续跑复用同一个 factory；配置校验模式不开硬件 |
| 唯一写入、预算与审核 | [session_core.py](../../robots/arx/gateway/session_core.py)、[tools.py](../../robots/arx/gateway/tools.py) | 所有 rollout 动作经过 gateway；审查 token 不跨中断复用 |
| bundle 输入及程序编译 | [real_input.py](../../robots/arx/deployment/real_input.py)、[bundle_program.py](../../robots/arx/deployment/bundle_program.py) | 校验 schema/SHA/feature 来源；恢复文案编译成有序工具调用 |
| 真机特征 | [picktube_rgbd_provider.py](../../robots/arx/deployment/picktube_rgbd_provider.py) | 距离/抬升来自标定深度与 FK；接触/抓取是有明确来源的代理量 |
| critic 与重入 | [bundle_runtime.py](../../robots/arx/gateway/bundle_runtime.py)、[critic.py](../../zetta/evolution/critic.py) | critic 仅提议中断；重入要重新核对实时观测 |
| 正常 rollout | [real_runner.py](../../robots/arx/deployment/real_runner.py)、[runner.py](../../robots/arx/deployment/runner.py) | 正常启动用 RealBundleRunner，续跑仅用于符合审计条件的 checkpoint |
| checkpoint 续跑 | [grasp_continuation.py](../../robots/arx/deployment/grasp_continuation.py)、[grasp_continuation_runner.py](../../robots/arx/deployment/grasp_continuation_runner.py) | 保留原始证据和已耗预算；恢复身份历史后仍需新的实时观测 |
| 抓取候选与执行 | [grasp_proposals.py](../../robots/manipulation/grasp_proposals.py)、[grasp_recovery.py](../../robots/arx/gateway/grasp_recovery.py) | 候选生成不写机械臂；审核与执行分别进行 |
| 收尾 | [finish_arx_real_episode.py](../../scripts/deployment/finish_arx_real_episode.py) | 卸载后核验实时归位，再失能；保留单独审计 |
| 进化证据与对照 | [audit_arx_real_evolution.py](../../scripts/evolution/audit_arx_real_evolution.py) | evidence/shadow/compare；一次配对不会自动晋升 |

修改已冻结的 provider、bundle、工具目录、硬件或抓取配置后，要在**新目录**重新冻结输入契约；旧实验文件的 SHA 仍代表旧实现。

## 命令分类

统一入口是 `scripts/deployment/arx_real.py`；子命令转发到原脚本。

| 子命令 | 原脚本 | 影响 |
| --- | --- | --- |
| `check` | `serve_arx_real_gateway.py --check-config` | 仅文件/契约校验，不开硬件 |
| `observe` | `audit_arx_live_observation.py` | 只读 RGB-D、14D、特征；需要已运行的状态控制器 |
| `camera-audit` | `audit_arx_live_cameras.py` | 只读相机 |
| `freeze` | `freeze_arx_picktube_inputs.py` | 生成新冻结契约/工具目录 |
| `run` | `run_arx_real_bundle.py` | 真机新 rollout |
| `continue` | `continue_arx_grasp_bundle.py` | `--check-only` 只读审核；`--execute` 写动作 |
| `stage` | `stage_arx_picktube_start.py` | 不加执行步数时只计划；执行时写动作 |
| `finish` | `finish_arx_real_episode.py` | `--execute` 归位并失能；持物需要卸载确认 |
| `evolution-audit` | `audit_arx_real_evolution.py` | 只读 evidence/shadow/compare，输出新审计文件 |

模型服务/控制器的生命周期仍由 `start_arx_model_a_dodo.sh`、`manage_arx_dodo_controller.sh` 管理。模型服务只做推理，控制器提供设备通信。启动控制器的行为与单纯启动模型服务不同。

## 当前文件与历史文件

- [当前部署索引](current-deployment.md) 给出这一轮验证过的 bundle 和 SHA，候选状态仍为未晋升。
- `docs/experiments/` 保留各轮协议、冻结输入和精简证据，原始 journal/RGB-D 在 dodo 大盘。
- `robots/arx/manifests/real/sample_*` 是示例，不自动等于当前实验候选。
- `robots/arx/critics/`、`zetta/evolution/arx/` 包含此前仿真/RGB package 路径；结构化真机 bundle 通过 `deployment/real_input.py` 和 `bundle_program.py` 执行。两种 artifact 不能互换。
- `integrations/cosmos-arx-rgb-cr*` 是带 vendor/ 和固定哈希的仿真交付快照。保留原文件以复现实验，当前开发修改 `robots/arx/` 主模块。
