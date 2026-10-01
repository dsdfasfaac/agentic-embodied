# 无开发历史的在线 Agent：一次独立执行结果

## 结论

独立 Agent 根据冻结的 recovery skill 和本次 RGB/工具反馈，完成了开口 EEF restaging、pregrasp 门控和新 Cosmos 接回；**完整 pickup 失败**。因此本次支持“skill 足以让新上下文 Agent 独立操作这套恢复接口”，不支持“skill 已能稳定修复任务”或“失败完全与 C&R 无关”。

## 实验隔离与范围

- 新 subagent `/root/independent_online_executor`，创建参数 `fork_turns="none"`，未指定不同模型。
- 开发端只提供独立 skill、当前状态及工具入口。在线 skill 的 3 个文件全程哈希不变，无历史 seed/轨迹/恢复动作/结果。
- 执行 Agent 可以保留自己的本轮观察和动作记忆；不是每个 action 都重置上下文。
- 父 Agent 仅进行一次工具连接状态询问，未给动作方向、参数或历史解决方案。关闭之后才读取接触、位姿等私有审计。
- 接口审计：91 次决策 ID 均唯一；暴露状态未出现所检查的私有字段；几何引用仅来自已交付的本轮 snapshots；明确 `finish`，worker exit=0。
- 这是上下文/接口输入隔离，不是通用工具的 OS 级沙箱；审计不能证明所有文件访问都受到权限强制限制。
- 后端 seed=183173，是已知开发失败 seed，不是 held-out seed。后端先重放原 Cosmos 动作前缀到 critic 触发状态；没有重放旧 recovery 动作。
- 仿真/critic 运行版本为 v1.0.1 两视角 critic；新独立执行工具和文档在 v1.1.0 交付。模型仍为原 checkpoint。

## 控制来源与结果

| 帧范围 | 视频时间（约） | 控制来源 |
|---|---|---|
| 1–183 | 0–12.2 s | 原 Cosmos 动作前缀回放，构造失败初始状态 |
| 184–983 | 12.2–65.5 s | 全新在线 Agent 自主选择的 recovery 工具 |
| 984–1047 | 65.5–69.9 s | 接回后的新 Cosmos 推理 |

第 183 帧 critic 提交尝试抓取但目标留在架上的提案。Agent 打开空夹爪，进行了 47 次小步 EEF 移动、38 次只读 RGB 几何查询，并在同步视角受遮挡时自主采集了两轮侧向观察基线。另有 1 次开夹爪、1 次 hold、1 次 pregrasp review、2 次 resume_vla、1 次 finish，共 91 次在线决策。

第 983 帧，连续 hold 后 pregrasp gate eligible=true。时序 RGB 基线 29.24 mm、最大射线残差 0.686 mm、子集拟合差 2.249 mm；band-to-commanded-TCP 的 tool 坐标偏移为 [31.53, 2.33, -7.57] mm。这里只是 RGB+命令预测的候选证据，不是真实目标位姿或深度误差上界。

Agent 自己填写五项视觉核查并批准 handoff。Cosmos 共进行了 4 次新推理，执行 64 步。第 1047 帧，Agent 从 RGB 判断空夹爪离开、目标仍在架中，按预先声明的一轮恢复协议主动终止，而不是在 handoff 成功后立即结束。

关闭后的独立仿真审计确认：

- 没有任何满足完整 pickup 条件的连续窗口；最终物理通过=false。
- 接回后的双指同时接触帧数为 0。
- 接回后最大 lift 为 -0.583 mm（相对初始基准），没有有效抬起。
- 环境宽松 success 判定也从未为真。

## 证据位置

H20 run：

`/data4/dingxin/cosmos_arx_cr_delivery_20260921/cosmos-arx-rgb-cr/runs/independent_online_20260922_A/`

其中 `three_view_controller_labels.mp4` 为 1048 帧、15 fps 的控制来源标注视频；`delivery_audit.json` 是关闭后物理评估，`agent_decisions.jsonl` 记录实际执行器决策（只读 geometry 查询另在 bridge 日志中）。

Mac 侧 capsule 和桥接审计：

`/Users/huangmingzhe/Documents/ChatGPT/zetta_v2/outputs/independent_online_20260922_A/`

`evaluator/` 下有冻结输入哈希、执行协议、全部 91 次在线决策、暴露状态记录和输入边界审计结果；`online/` 是本轮实际给执行 Agent 的独立 capsule。不要在未来在线 Agent 输入中包含 evaluator 或本结果文档。

## 本次尚未证明的事

没有证明跨 seed 泛化、稳定成功率或严格权限沙箱。Recovery 使用 800 个仿真步，观察成本较高；预抓取 gate 的候选范围是否真正适合 Cosmos 接手仍需进一步验证。不能仅因通过 gate，就将后续失败全部归因于 Cosmos。
