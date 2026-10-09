# 原 sim bundle 的真机部署适配，2026-10-09

父候选是原附件 `cand-005-alignment-restage-settle-v2`，SHA `7d922aeff6d64b77a7c0db938aad7052d4a80c29baa2f76a21b350ef306d6f7c`。原文件保存在相邻 `arx-real2sim2real-zeroshot-20261009`，未修改。

本候选 `cand-005-alignment-restage-settle-v2-real-v1` 为人工部署适配，generation=1、parent_sha256 指向父候选。不能计作原候选原样 zero-shot 或自动学习的成功。

## 改动

- critic 增加真机 RGB-D + FK 的粉色目标距离 `<=0.08m` 激活条件。保留闭爪、无接触、无抓取、抬升 `<=0.02m`、未成功和连续12个物理动作。两次原试验在目标约42cm远时误入该失败阶段；8cm是现有5cm接触范围外加3cm接近余量，属于待实测的部署阈值。
- 张爪 `max_steps` 从20到60。保留 `opening=1`、每步0.08、滤波0.35和+0.9命令补偿；覆盖实际31步限速下限及滤波收敛，不改变速度/单位。
- 保持5步，再新鲜观测审核，最多4个16步VLA chunk。恢复物理上限129步，4个工具调用。
- cooldown同步调整到129个物理动作；真机 monitor 在开爪激活条件变假后仍保留该窗口。只读重新观测不累计dwell，也不消耗cooldown；未知观测仍打断失败窗口。
- 开爪恢复的真实重入审核额外核对：实测张开、目标接触=false、grasped=false、success=false。不能因为距离门槛变假而让仍闭合或可能持物的夹爪获得重入令牌。
- bounded VLA 重入之后允许VLA再次闭爪抓取；不再要求整个4个chunk都保持开爪。成功终止仍优先于恢复完成。

基础模型、关节界限、控制滤波、夹爪补偿、600步任务预算和真实成功条件均使用既有冻结配置。本候选不包含GraspGen或自动抬升步骤。

## 验证

相关91项测试通过，涵盖远处空闭爪不触发、近处12个物理动作触发、未知观测打断dwell、同一步重复读取不增加计数、开爪之后cooldown保留、滤波下60步收敛、闭爪/持物状态禁止重入，以及空夹爪验收的准入条件。

真机试验结果另记在本目录的 `evidence/`，大数据写入 dodo `/mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/trial04/`。未触发恢复的试验不能证明恢复成功；单次试验也不能证明候选优于父版本。

## 本轮真机结果

| 验证 | 结果 | 能证明的范围 |
| --- | --- | --- |
| trial04，代码 b33b7e3 | 完整执行600步，600次发送均记录实测到位；最小目标距离0.25009m；接触、抓取、成功、恢复均为0 | 没有再因远处空闭爪提前触发原失败规则；VLA本轮没有接近到8cm触发范围，未完成抓取 |
| gripper-acceptance01，代码7e3d3de | 起始位空夹爪闭爪53步、张爪53步、保持5步；每步保留观测；工具均实测到位 | 60步预算覆盖本次真实夹爪收敛，右臂关节命令保持不变 |
| 只读重入审核 | 健康、新鲜观测、实测到位、张爪、无接触、无持物、未成功、critic清除全部通过 | 当前状态满足审核条件；未制造critic事件、未发重入令牌、未调用VLA，不计作恢复救回 |
| 两次收尾 | 均实测归位后关闭控制器 | 现场当前已归位、失能 |

trial04 的 runner 字段为 `status=completed / termination_reason=environment_ended`，代表600步预算耗尽；`task_success=null`，不能读作任务成功。观测记录中的成功特征始终为false。`trial04-summary.json` 保留最近目标观测、末帧特征、完整事件计数和命令间隔摘要。

本轮命令发送间隔中位数320.98ms（约3.12Hz），配置15Hz为上限。当前逐步同步观测并等待实测到位的执行方式仍有停顿；本次候选适配没有完成连续轨迹执行优化。

下一项任务级检查应核对初始RGB/状态与模型训练分布、目标颜色及任务提示是否匹配，解释VLA为何始终离目标至少25cm。当前证据不能确定是模型能力、部署输入分布还是标定偏差，也没有近目标critic触发并恢复抓取的现场证据。

## 空夹爪验收复现

先按既有流程使能控制器并实测进入PickTube起始位，在已确认空夹爪的受监护现场执行：

```bash
cd /home/dodo/chenfu/Agentic-Embodied
bash /mnt/hdd16t/chenfu/grasp_recovery/adaptation-20261008/run-env31213.sh \
  scripts/deployment/commission_arx_empty_gripper.py \
  --experiment docs/experiments/arx-real2sim2real-adapted-20261009 \
  --hardware-config docs/experiments/arx-graspgen-live-20261008/hardware-sdk-bounded.json \
  --output /mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/gripper-acceptance-NEW \
  --execute
```

输出目录必须为新目录。脚本不启动或关闭控制器；若失败，保持使能以便检查。验收成功后使用 `finish_arx_real_episode.py` 对该输出目录验证归位，再关闭控制器。工具请求、真实反馈、逐步观测和审核留在其 `private/gateway/journal.sqlite3`；仓库只保存小型结果、身份信息和摘要。
