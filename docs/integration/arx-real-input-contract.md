# ARX 真机 CandidateBundle 输入契约

入口：`robots/arx/deployment/real_input.py` 的 `preflight_real_bundle`。它在动作开放前校验冻结文件与真机适配器报告的能力；当前仓库尚无真机适配器或真机执行器，因此这个入口只完成输入审计，不会启动机器人。

以 `runs/arx_privileged_test_20260929_061734/campaign/bundle.json` 为基准：沿用 CandidateBundle v1 的字段、`post-recovery` 观察 ID 占位值、`token-from-review` reentry token 占位值，以及 2 cm EEF 移动。EEF 单次调用的网关限制是 1 cm，预检把该步骤规划为两次调用并计入 `max_recovery_tool_calls`。后续真机执行器必须按这一规划拆分并在每次调用后检查反馈；预检不会代它执行。

## 冻结文件与能力

`RealInputContract` 是严格 JSON 对象，`schema_version` 固定为 `arx.real.input.v1`。其中绑定：

- CandidateBundle 的规范化 SHA-256 和原始文件 SHA-256；
- 任务清单的原始文件 SHA-256、task ID 和 task name；
- VLA 模型契约的原始文件 SHA-256；
- 工具目录 SHA-256；
- 按模型契约顺序排列的 front/left/right 三路相机及设备、标定 ID、标定文件 SHA、图像尺寸和颜色格式；
- 14 路关节反馈名称；
- 每个 critic 特征的 `name`、provider ID/SHA、来源种类、相机或关节通道、标量类型、单位和最大数据年龄；
- critic 历史/冷却预算以及恢复工具调用预算。

`LiveCapabilities` 必须由可信的真机适配器在启动时从实际设备与已加载的特征 provider 生成。预检逐一比较相机、标定、关节映射、工具目录和 provider 声明。声明中的 `source_ids` 只能引用已验证的相机或关节通道；`fused` 特征必须同时引用两类输入。critic 主谓词及 activation 条件的每个特征名都必须有匹配的 provider。没有真机来源的 `privileged.*` 名称直接拒绝。仅把 MuJoCo 字段改名或填写一个 JSON 声明，不构成真机观测能力。

示例 bundle 的两个 privileged 特征需要实际实现并标定：

| critic 特征 | 所需真机观测 |
| --- | --- |
| `privileged.interaction.gripper_closed` | 对应夹爪的关节/夹爪反馈，以及标定过的闭合判定 |
| `privileged.selected.target_gripper_distance_m` | 相机中的目标检测/定位与关节反馈得到的夹爪位姿，统一坐标系和距离计算 |

当前仓库没有上述真机 provider。测试里的 provider 与 SHA 是夹具数据，只验证校验器行为，不能用于真机启动。

## 恢复参数

每个恢复工具必须存在于冻结工具目录、允许处于 `RECOVERING`，其参数须符合当前网关模型。示例 bundle 的 2 cm 位移按单次 1 cm 上限展开；展开后的总调用次数必须在契约预算内。`post-recovery` 必须在执行时替换为最新真机 observation ID，`token-from-review` 必须替换为前一步 `arx.review_reentry` 返回的 token。预检接受这两个示例占位值并在报告中给出工具调用数量；执行器尚未实现时，不应把预检通过理解为恢复动作已可执行。

## 离线审计

`scripts/deployment/check_arx_real_bundle.py` 读取冻结契约和能力快照，输出 JSON 报告；失败时退出码为 2。它用于文件和 schema 审计。最终真机启动必须在同一进程里从适配器取得实时 `LiveCapabilities`，调用 `preflight_real_bundle`，通过后才开放动作。CLI 快照不能代替实时设备核验。
调用形式：

```bash
python scripts/deployment/check_arx_real_bundle.py \
  --bundle runs/arx_privileged_test_20260929_061734/campaign/bundle.json \
  --task-manifest robots/arx/manifests/pickup_test_tube.yaml \
  --model-contract robots/arx/manifests/task7_model_a.yaml \
  --tool-catalog runs/arx_privileged_test_20260929_061734/campaign/tool-catalog.json \
  --real-contract /path/to/frozen-real-input.json \
  --capabilities-snapshot /path/to/adapter-capabilities.json \
  --expected-contract-sha256 SHA256_FROM_DEPLOYMENT_CONFIG
```

`--expected-contract-sha256` 必须从独立的部署配置取得，不能临时从待校验文件重新计算后填入。CLI 的成功报告带有 `audit_mode=offline_capability_snapshot`，不能作为真机动作准入凭据。
