# 粉色试管接触与抬升验证（2026-10-07）

目标：保持原有粉色目标识别与真实成功条件，完整执行 generation4 的 engage / close / contact review / 2cm lift / 五帧 hold。模型、相机、外参、bundle 与预算沿用第 13 轮，原始数据留机械硬盘。

第 13 轮现场确认已夹住粉色试管，但自动流程在闭爪阶段中断。离线原始深度显示腕部目标处于 79–82mm；旧的 80mm 下限丢弃了有效近距点。D405 官方范围从 70mm 开始（[Intel 规格](https://www.intel.com/content/www/us/en/products/sku/229218/intel-realsense-depth-camera-d405/specifications.html)）。新观察器只将已标定的 right_rgb D405 下限修正为 70mm，前置范围不变；保持至少 12 点、80% 标签深度支持、新帧、源身份、深度一致性和三维连续性检查。

`PickTubeGraspObserver` 同步使用样本的准入范围，避免 critic 能观测而 grasp review 仍丢弃同一批近距点。未复用旧深度或用末端 FK 构造目标抬升。SHA 与试验参数见 `protocol-14.json` / `frozen/`。

当前：静态与近距回归已通过，等待离线实际 RGB-D 回放和真机复测。无自动成功或候选晋级结论。

## Trial14 result

278 commands,278 verified arrivals. Verified engage completed at260; closure interrupted after18 steps at278. No lift executed. Terminal observer unknown; operator confirmed pink tube held. Controller remains enabled for audited closing continuation. Minimum observed distance13.138mm; maximum observed lift2.298mm. This trial did not achieve the five-frame success condition.
