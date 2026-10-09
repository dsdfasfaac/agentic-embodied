# 原 critic 与恢复顺序，仅适配真机预算（2026-10-09）

本目录是当前 sim bundle 真机迁移的执行入口。父候选为用户原附件
`cand-005-alignment-restage-settle-v2`，原 SHA 为
`7d922aeff6d64b77a7c0db938aad7052d4a80c29baa2f76a21b350ef306d6f7c`。
原附件和此前加入 8 cm 门槛的实验文件均保留，实验数据不改写。

## 执行约束

| 项目 | 本次配置 |
| --- | --- |
| critic | 与原 bundle 的整个 `critic_rules` 逐字段相同 |
| 触发条件 | 闭爪 ≥1、接触 ≤0.5、抓取 ≤0、抬升 ≤0.02 m、成功 =0 |
| 持续 / 冷却 | 12 / 50 个物理动作，与原 bundle 相同 |
| 张爪 | `opening=1`，唯一参数变化 `max_steps:20→60` |
| 保持 | 5 步，与原 bundle 相同 |
| 新鲜 VLA | 最多 4 个 chunk，与原 bundle 相同；当前接口每个 chunk 执行 16 步 |
| 恢复预算 | 最多 `60+5+4×16=129` 个物理动作，4 个工具调用 |
| 整轮预算 | 600 个物理动作，最多 4 次恢复，保持已有上限 |

执行动作顺序为 **张爪 → 保持 → VLA**。已有只读
`arx.review_reentry` 位于 VLA 前，核对新鲜健康观测、实测到位、夹爪张开、
无目标接触和持物以及 critic 清除，并签发重入令牌。它没有运动动作。
当前恢复期间，gateway 独立抑制同一触发规则的重复中断；无需扩大原 50 步
cooldown。重入 VLA 后恢复正常监测。

此前增加的 8 cm 距离条件已从本候选移除。远距离空闭爪满足原五个条件时
也会触发；没有将其预先定义为误报。没有增加几何、GraspGen 或自动抬升步骤。

60 步是张爪上限，达到目标可提前结束。依据是既有空夹爪验收的完整张爪
实测 53 步，以及原滤波/步长限制下的收敛测试。每步 0.08、滤波 0.35、
右夹爪 +0.9 补偿、关节界限和成功条件均保持原配置。

## 原始语义与预算记录

除张爪 `max_steps` 外，整个 `recovery_rules` 保持原样，包括 precondition、
stop_when、stop_condition、fallback 和安全说明。候选 ID、generation、parent
和 mechanism_change 记录这次人工预算适配；不能称为原文件逐字节 zero-shot。

原 prose 中仍有“20 步张爪、4 chunks 最多 20 个动作、50 步冷却覆盖恢复”
的旧算术。这些自然语言字段作为审计文本保留；编译器用工具参数计算实际
预算，由实测完成、成功优先和只读重入审核约束执行，未实现任意自然语言
stop_when 的逻辑解析。当前接口 4 chunks 上限为 64 步；尚无成功 sim 运行
轨迹证明其 chunk 长度或基础 VLA 权重与真机一致。

## 验证结果

- 55 项相关测试通过：原条件/恢复规则不变、远处空闭爪触发、12 步物理 dwell、
  50 步 cooldown、未知特征、恢复预算与顺序、恢复中重复事件抑制及真实重入。
- dodo 配置预检通过：`eligible=true`、`hardware_opened=false`。
- 只读回放 trial03：原 critic 仍在第 37 步触发，目标距离约 0.418 m。
- 只读回放 trial06：原 critic 在第 37 步会触发，目标距离约 0.480 m；实际
  历史运行用 8 cm 版本、没有触发。后续回放事件仅来自原有轨迹，不能代表
  真实恢复后的轨迹或抓取结果。
- 本次没有启动控制器、发送机器人命令或执行新 rollout。机械臂保持停止。

小型证据位于 `evidence/`，SHA 索引为 `evidence/index.json`；大数据留在 dodo。

## 下一轮执行入口

在 dodo 完成现场确认、控制器使能、实测 staging 和相机检查后，用新输出目录
运行。模型仍是本机端口 5583 的 Task7 Model A iter5000。

```bash
cd /home/dodo/chenfu/Agentic-Embodied
bash /mnt/hdd16t/chenfu/grasp_recovery/adaptation-20261008/run-env31213.sh \
  scripts/deployment/prepare_arx_trial_storage.py \
  --output /mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/trial07 \
  --fast-root /mnt/nvme0/chenfu/arx_gateway_journals
bash docs/experiments/arx-real2sim2real-budgetonly-20261009/run-trial.sh \
  /mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/trial07
```

`run-trial.sh` 默认采用已有异步无损传感器归档配置；小型 journal 位于 NVMe，
大型 RGB-D 留在 HDD。trial07 是预留名字，本次未创建或运行。结束后使用已有
实测归位流程再失能；可能持物时先现场卸载。
