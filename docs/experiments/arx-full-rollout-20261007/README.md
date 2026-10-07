# Zetta 真机完整 runner 试验（2026-10-07）

本次使用 `run_arx_real_bundle.py` / `RealBundleRunner`，允许 recovery 重入后的 VLA 持续运行。模型为 dodo 本机 Model A iter5000、seed42，600 个物理步骤、64 次工具决策、最多 4 次 recovery。没有 commissioning 的单动作块结束条件。成功仍为粉色目标的可验证抓取、抬升至少 1 cm 并保持五个新帧。

输入沿用上一轮已验证的 `arx-target-visibility-20261007/bundle-wrist-refine.json`、`hardware-wrist-observer.json`、`grasp-config-wrist.json`、`frozen-refine/`，SHA 与运行预算见 `protocol.json`、`protocol-10.json`。候选包含初始远距空夹爪触发和两次预抓取；它仍不是完整的错误颜色或抓取丢失 detector，也未晋级为已学好候选。

## 执行结果

| 试验 | 命令 / 实测到位 | recovery / 重入 | 终止 |
| --- | --- | --- | --- |
| `full-zetta-rollout-09` | 220 / 220 | 完成两阶段预抓取；重入未通过 | 日志汇总使旧终点观测过期，审核拒绝；runner 报 `recovery_step_failed`。这是运行框架问题，不能当作任务失败比较。 |
| `full-zetta-rollout-10` | 265 / 265 | 1 次 recovery，重入已通过 | VLA 连续执行后目标观测不可用；等待两秒未恢复，返回 `observation_unavailable`，`error=null`。任务成功结果为 unknown。 |

第 10 轮步骤为：初始 VLA 1 → 首次预抓取 199 → 腕部新点云校正 18 → 新帧重入审核 → VLA 16 + 16 + 15，共 265 步。运行由真实观测中断，未人为提前停止正常运动，也未跑到 600 步预算。

第 236 步进入闭爪状态，第 237 步出现接触证据，最近目标距离 24.46 mm。第 249 步附近目标与工具相对关系失去保持，后续 contact 为 false，grasped 全程未通过。最大可验证抬升为 5.30 mm（第 262 步）；没有达到 1 cm / 五帧条件。

第 248→249 步右夹爪反馈由 −0.352 变到 −0.080，电流由 1.321 降至 0.564；之后反馈接近 −0.05。腕部中粉色标签逐渐被夹爪遮挡，前置镜头也受到遮挡，并检测到与既有三维目标相距较远的粉色区域，连续性检查拒绝这些区域。终止时缺失的 contact、lift、grasped、success、distance 保持空值，没有用旧位置补足或判成功。

现场人员随后反馈：“已经接触到边缘了，没有完全加持，但是末端执行器没起来”。结合 grasped 全程未通过，本轮记为未形成稳定抓取、未完成抬升。runner 的自动任务结果仍为 unknown，现场结论单独留存，不回填成自动判定。电流本身也可能受到 `+0.9` 预载和机械闭合止点影响，不能单独作为稳定抓取证明。

## 本次框架修复

- 真机 `arx.review_reentry` 在请求校验后读取新的同步 RGB-D / state，不发送运动命令。有效审核窗口的最后一项替换为新观测，并记录请求窗口与实际窗口；重入 token 绑定新观测 ID。
- 原有身份、时间戳、目标位置、到位和成功条件继续由审核执行。没有放宽新帧的时效要求。
- `RuntimeLimits.retain_step_sensors=true` 记录每个实际步骤的原始 RGB-D，关节、电流和时刻保留在对应 journal 中；该选项默认关闭。
- runner reconciliation 为 180 秒，足以等待最多 120 秒的单工具操作，避免把仍在执行的长预抓取误判为未决。

相关测试 122 项通过、1 项因本地没有 RealSense SDK 跳过。新增回归覆盖过期旧帧的只读刷新、token 与新帧绑定，以及留存传感器不增加物理命令。`git diff --check` 通过。

## 原始记录与现场状态

原始文件在 dodo 机械硬盘：

- `/mnt/hdd16t/chenfu/grasp_recovery/full-zetta-rollout-09/`
- `/mnt/hdd16t/chenfu/grasp_recovery/full-zetta-rollout-10/`

仓库 `runs/arx_grasp_recovery_20261007/` 中同名目录是软链接。原始 RGB-D 不进入 Git；`evidence/` 保存结果、journal SHA、关键观测与图片。

第 09 轮结束后已实测归位再重跑。第 10 轮结束后控制器保持使能、保持当前姿态。现场确认末端接触边缘、没有完全夹持；已请求移开接触物、确认空夹爪并退出作业区，等待确认后执行归位与失能。

## 下一轮学习证据

本轮没有成功证据，不允许候选晋级。后续候选应针对“闭爪后未形成目标与工具共运动”建立明确的任务级 critic，早于彻底遮挡提出中断；恢复还需验证实际 engage / 夹持保持，并区分 `+0.9` 预载下空夹爪止点电流与真实持物。上述动作尚未在本轮自动执行。
