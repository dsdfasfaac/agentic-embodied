# v1.0.0 交付验证

2026-09-21 在 H20 `h20-125` 的新解压目录实测；不是只做文档整理。

| 检查 | 结果 |
|---|---|
| 原冻结在线代码 | 10 个文件 SHA256 全部与原 v4.1 一致 |
| 包完整性 | cr.py verify 通过；MANIFEST 覆盖随包文件 |
| H20 doctor | client venv 保留，依赖可导入，3 相机场景编译通过，关键外部文件哈希无不匹配 |
| 单元测试 | 37 项通过：critic 7、门控及离线审计 24、交付/管道适配 6 |
| Skill | 官方 skill-creator 的 quick_validate.py 检查通过 |
| CPU 恢复 smoke | f723；29 个 RGB 检查点逐像素一致；pregrasp eligible=true；零新 Cosmos 请求 |
| live 入口 | 首个暂停 f183；直接 resume 被拒绝且帧未推进；正常 finish |
| replay 后真实接回 | Agent 重新看 f723 RGB、重新 gate 并审批；实际发起 2 次 Cosmos 请求，执行 32 步，到 f755 后按交付集成测试范围结束 |
| 可选管道客户端 | 在 H20 启动 live worker，收到 f183 观察，通过 stdin 发送显式 finish，收到保存确认并正常退出 |
| 离线审计 | smoke、live 入口、replay+handoff 三种模式均可生成报告；未将回放或接回标成任务成功 |
| 视频 | Linux 上使用 imageio-ffmpeg 后备二进制和 DejaVu 字体；生成 724 帧 smoke、756 帧接回检查视频；抽查回放/新 Cosmos 边界标签清楚 |
| 资源回收 | 交付测试的 GPU2 / 15611 server 已停止，端口关闭、GPU2 显存释放；未停止其他作业 |

结构化证据见 `provenance/validation.json`。完整本次测试输出保留在 H20：

`/data4/dingxin/cosmos_arx_cr_delivery_20260921/cosmos-arx-rgb-cr/runs/`

压缩包不含新测试的 `runs/` 大量中间输出，含三条原回归参考视频、CPU smoke 所需 fixture 及结构化验证记录。

## 与原代码的关系

在线 critic、EEF IK、pregrasp gate、Cosmos handoff 类未改动。新 `worker.py` 仅设置外部路径、提供明确模式、保留输入管道和关闭标记；`cr.py` 负责选择原有两套 Python 环境、检查路径/GPU/端口和启动。

离线视频工具增加了 Linux 字体、imageio-ffmpeg fallback，以及 recorded replay 明确标签。额外审计入口拒绝未关闭/异常关闭的 delivery session。可选管道客户端不含 LLM，不生成恢复动作，超时也不会自动重发。

打包过程中修正了发现的适配问题：不可对 venv Python 的符号链接做 realpath（会绕过 venv）、H20 没有系统 ffmpeg（使用环境已有二进制）、processor 实际随 checkpoint 导出而非不存在的单独目录。默认环境配置据真实 H20 文件核实，不修改共享环境。

## 不代表什么

- 不是完整新 C&R 评估；32 步 fresh Cosmos 只验证交付链路。原 3 个其他失败 seed 的完整结果仍为 2 个接回、0 个完整 pickup。
- 同学自己的 Agent 的恢复选择可能不同。记录动作回放、人工决定与新的 Agent 自主恢复必须分别标注。
- 没有在另一台干净 H20、另一账号权限环境或真机上验证；新账号必须自行 doctor 检查共享资源权限。
- 没有打包大模型权重/网格，也不保证不同版本 MuJoCo、渲染后端或模型软件能逐像素重现。
- 37 项测试与 gate 通过不能证明抓取安全、策略成功率或恢复状态已属于 Cosmos 的可靠接手分布。
