# 外部 Agent 协议

适用这一个 MuJoCo 场景；不适用于真机。Agent 自行看 RGB、判断证据并选择工具，critic 和几何模块只给 proposal。这里没有自动调用 LLM 的程序。人工输入的决定应称人工操作，不要写成模型自主决策。

## 读什么

运行窗口返回 `frame`、`rgb`（三张 PNG 路径）、`active_tool`、`environment_ended`、`latest_proposals`。最新观察也在 `agent_observations.jsonl`，三视图图像在 `frame_XXXX/three_view.png`。另可读自己的 `actions.jsonl` 和在线 `pregrasp_features.jsonl`。

critic 只看 RGB。恢复允许 RGB、静态标定和自己的命令历史；不允许读当前 measured joints、物体 pose、contact、reward、`posthoc_trace.json`、`audit.json` 或离线 reference_results 来制定在线动作。Nominal Cosmos 保持原 RGB+state 接口，不能称整个系统无 proprioception。

critic 的 `attempted_grasp_target_left_behind` 表示：腕部目标曾接近、图像中手指闭合提示成立，然后腕部目标缩小/移开，而固定视角目标仍接近初始架上位置且稳定，持续条件达到阈值。它不是触觉证明。`grasp_outcome_unknown` 仅表示证据不够，不能直接解释成已掉落，也不能直接解释成成功。颜色、遮挡、手指模板与初始 rack 关系均可能失效。

## 一次一条 JSON

每条请求四个必填字段：唯一 `decision_id`、与当前观察相同的 `evidence_frame`、`tool`、非空 `reason`。`args` 依工具而定。单位 m/rad/s。不要在上一请求仍执行时发送下一条，也不要预填未来 frame。

示例里的 frame、ID、坐标均需按当前证据填写，不是一串可直接执行的恢复策略。

### hold

```json
{"decision_id":"inspect-183","evidence_frame":183,"tool":"hold","args":{"frames":8},"reason":"说明当前 RGB 为什么适合保持，是否需要确认目标稳定。"}
```

`frames` 为 1..15。pregrasp 至少需要 6 帧连续显式 hold，通常发 8 帧。任何实际动作（含 hold）都会让旧 gate proposal 失效。

### set_gripper

```json
{"decision_id":"open-183","evidence_frame":183,"tool":"set_gripper","args":{"opening":1},"reason":"RGB 显示空手且目标仍由架子支撑，先张开夹爪。"}
```

`opening` 0=闭、1=开，范围 0..1；这套 pregrasp 恢复应保持张开，交由 Cosmos 完成闭合与提取。工具保持其他关节目标不变。

### move_eef

```json
{"decision_id":"move-231","evidence_frame":231,"tool":"move_eef","args":{"delta_xyz_m":[0.002,0.001,-0.005],"delta_rotvec_rad":[0,0,0],"frame":"tool","speed_m_s":0.012},"reason":"根据当前 RGB 和可用深度证据说明这一个小步的方向与目的。"}
```

- 平移向量长度 <=0.01 m，不是每轴各 0.01 m；旋转向量长度 <=0.1 rad。
- `frame` 为 `world` 或 `tool`；默认 world，旋转默认不变。
- `speed_m_s` 在 (0,0.03]；每条包含 <=1 mm 规划点和命令滤波稳定等待。
- IK 使用静态机器人链和 commanded TCP，不读取实际机器人状态；它不做碰撞检测，不保证命令真的到位。动作后重新看 RGB。
- 接近目标时先保持开口和安全的可见空间，不能在失去目标/深度不明时盲目继续下降。

### RGB 几何 proposal（另一终端，只读）

```bash
python3 cr.py geometry runs/live_183173 --frame 231
python3 cr.py geometry runs/live_183173 --frame 723 \
  --temporal-frames 573 594 612 632 652 673 694 715 723
```

第一种同时多视角，第二种多时刻腕部 RGB。后者的相机位移来自自己的命令，不是测得位姿，且假设目标在窗口内静止。输出 `unknown` / 射线不一致 / 病态条件时，不要拿最近一次估计当保证；可基于可见空间选择有限主动观察，或停止。小射线残差不等于真实深度误差小。

### review_pregrasp

```json
{"decision_id":"gate-723","evidence_frame":723,"tool":"review_pregrasp","args":{"measurement_mode":"temporal_rgb","observation_frames":[573,594,612,632,652,673,694,715,723]},"reason":"Agent 已看当前和历史 RGB，目标稳定，选择多时刻测量复核，尚不请求接回。"}
```

默认 `args:{}` 使用 simultaneous。temporal 要求 >=5 个有 PNG 的有序、唯一观察帧，均在最近 150 帧内，末帧为当前帧。运行器自己重算几何，不能传入伪造的 eligible 或深度。结果在 stdout 的 `pregrasp_proposal` 以及 `pregrasp_proposals.jsonl`；review 不推进仿真，也不自动 snapshot。

候选范围：色带相对 commanded TCP 的 tool X 为 15..35 mm，|Y|<=6 mm，|Z|<=8 mm；夹爪视觉张开且近似水平、6 帧 hold 稳定、目标支撑/架子稳定提示成立。simultaneous 残差 <=3 mm；temporal 基线 >=25 mm、残差 <=2 mm、两个子集拟合差 <=3 mm，子集也需独立通过检查。这是开发候选范围，不是已验证的 Cosmos initiation region。

### resume_vla：首次接回

Agent 看 RGB 后填五项事实核查，引用当前真实 proposal ID。每项至少 12 字符；不应为了凑长度写无依据结论。

```json
{"decision_id":"handoff-723","evidence_frame":723,"tool":"resume_vla","args":{"max_chunks":2,"pregrasp_proposal_id":"从本次 gate 输出复制，不可沿用示例","reentry_reason":"说明为何已处于可供策略重新抓取的开口 pregrasp。","visual_checks":{"target_identity":"说明是同一粉色标记目标，不是邻管。","open_finger_corridor":"说明两指开口及目标与开口的位置关系。","approach_clear":"说明可见的接近空间与架子边缘关系。","supported_target":"说明目标仍保持直立且由架子支撑。","appropriate_policy_phase":"说明尚未抓住，接下来的闭合和拔出由策略执行。"}},"reason":"Agent 基于当前 RGB 和候选检查批准一次新 Cosmos 接手。"}
```

运行器要求此前确有 EEF 重定位，`needs_pregrasp` 期间禁止直接重试 Cosmos。实际 handoff 会丢弃旧 chunk，请求当前观察下的新推理。默认每次返回 32 个动作，执行前 16 个，再看新的观察；提案/环境结束会中断剩余动作。`max_chunks` 1..32，建议首次 1..2，按 RGB 决定是否增加审阅间隔。

### resume_vla：nominal 继续

```json
{"decision_id":"continue-755","evidence_frame":755,"tool":"resume_vla","args":{"max_chunks":2},"reason":"根据当前 RGB 说明为何继续策略，尚未满足完整 pickup 判据。"}
```

若再次出现明确失败，不能绕过锁存门控；本次可比回归最多一次恢复，记录失败并结束。发生 unknown 时先看图，可在确认场景仍适用后继续 nominal 至预算，但不能把 unknown 当成功。

### finish

```json
{"decision_id":"stop-867","evidence_frame":867,"tool":"finish","args":{},"reason":"说明是再次失败、预算用尽、场景不适用，还是视觉完整提取且保持待离线核验。"}
```

写完等进程退出和 `CLOSED.json`。stdin EOF 也会关闭，但标为 transport close，不假装是 Agent 选择。不要用强杀代替 finish，否则视频/评估可能不完整。显存服务需在另一终端单独停止。

## 结束标准与汇报

固定本批协议：1400 总步数（包括 prefix/recovery），最多一次恢复；架子翻倒/目标不再支撑则停止。视觉看起来完整拔出后 hold 15 帧；全部 live trial 结束再用 `audit` 评估完整离架、双指接触、无其他接触、15 帧相对稳定。不保证环境较宽松的 success 判定足够。

分开报告：检测帧/提案、工具数与恢复耗时、gate 条件、handoff 帧、新 Cosmos 请求/步数、终止原因、完整任务结果。回放模式标注 recorded recovery reproduction，不能纳入新 Agent 成功率。
