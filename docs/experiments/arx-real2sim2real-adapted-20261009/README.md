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

相关87项测试通过，涵盖远处空闭爪不触发、近处12个物理动作触发、未知观测打断dwell、同一步重复读取不增加计数、开爪之后cooldown保留、滤波下60步收敛和闭爪/持物状态禁止重入。

真机试验结果另记在本目录的 `evidence/`，大数据写入 dodo `/mnt/hdd16t/chenfu/grasp_recovery/real2sim2real-20261009/trial04/`。未触发恢复的试验不能证明恢复成功；单次试验也不能证明候选优于父版本。
