# 闭爪无抓取 / 无抬升候选（2026-10-07）

父候选是 `real-pink-wrist-refine-20261007`，父试验 `full-zetta-rollout-10` 现场反馈为接触边缘、没有稳定夹持或抬升。新候选尚未晋级。

## 修改与验证

- `closed_contact_no_lift`：闭爪且有接触，但未验证 grasp / success，8 个连续观测的抬升变化不超过 2 mm。
- `closed_target_separated`：闭爪、无接触且无 grasp / success，目标距离连续 3 个观测大于 35 mm。
- 旧轨迹离线回放分别在第 244、251 步触发，早于第 265 步的观测丢失。回放未执行运动，新 recovery 会改变后续轨迹，因此不能将此回放当作新候选的真实成功率。
- recovery 按 bundle 编译：张爪审核 → 两阶段预抓取 → engage 审核/实测 → 闭爪 → contact 审核 → 2 cm lift 审核/实测 → 五帧 hold → 新帧重入审核 → VLA。最多 15 个工具调用，保守预留 561 个物理步骤，仍受总 600 步预算约束。
- 张爪需新同步观测；闭爪阶段只有在配置的空夹爪闭合止点（最多 3% 开度）、目标已分离且未抬升、无 contact / grasp / success 时允许。高电流不能单独判持物或空爪；接触沿或持物不明时拒绝张爪，保留控制器供现场清理。这个门限仅适用现有 PickTube 夹爪，其他工具不能照搬。
- `+0.9` 闭爪预载保留；成功仍需粉色目标真实抬升至少 1 cm 并保持五个新帧。

## 真机试验

冻结输入见 `protocol-11.json` 与 `frozen/`。模型、相机、腕部外参、特征观察器与总预算沿用父试验。新增候选先经过规则回放、编译与夹爪准入测试，再部署 dodo。实时结果另存 `evidence/`，原始 RGB-D 留在 `/mnt/hdd16t/chenfu/grasp_recovery/retention-full-rollout-11/`。

当前：部署与起始状态检查中，尚无本候选真机结果。
