# 独立在线执行 Agent：开发与执行分离

## 两种角色，不共享开发历史

- 开发端负责维护 critic、工具和 recovery skill，冻结运行版本，建立环境。
- 执行端是新建且不继承开发对话的多模态 Agent。它只收到独立 skill、当前 RGB/critic 提案、工具接口；之后可以保留本次 episode 自己积累的观察/动作记忆。
- 评估端在执行进程正常关闭之后才读取仿真接触、位姿和成功判据。实验期间不向执行端提示动作或历史结论。

这里没有训练一个新的模型，也没有修改 Cosmos 权重；“知识交付”是显式文本 skill，而不是开发 Agent 的隐式聊天记忆。

## 给执行端的独立材料

`online-skill/cosmos-arx-online-recovery/` 是可单独交给新 Agent 的 skill。
其中包含判定抓取失败/未知、检查目标支撑、选择小步 EEF 调整、主动 RGB 观察、pregrasp 门控、接回后继续 Cosmos 和终止标准。

它不包含历史 seed、成功/失败结果、参考动作序列、原始轨迹文件、旧恢复帧号或私有评估。不要把根目录开发版 `SKILL.md`、README、fixtures、reference_results 一并作为在线 Agent 输入。

## 运行流程

1. 运行器检查 H20 环境，准备 `live` 模式，不使用 `replay-pregrasp`。同一原 Cosmos 失败前缀可以由后端重建初始失败状态，但不向执行 Agent提供该文件或旧恢复动作。
2. 在可 SSH 连接 H20 的电脑启动下方 bridge。它通过 SSH 控制后端 worker，将当前三视角 PNG 复制到独立 capsule，提供 loopback 的 `state` / `action` 接口。
3. 用你的 Agent 框架启动一个全新上下文，只给 capsule 内的 `SKILL.md`、`scripts/client.py` 路径和“完成当前 episode”的任务。Codex subagent 对应 `fork_turns="none"`；不要把父对话摘要填回 prompt。
4. 执行 Agent 自己看图、选工具、批准恢复与 handoff、继续策略，最后明确 `finish`。父 Agent 只做运行监控，不给动作建议。
5. 等 `closed=true` 和 worker 的 `CLOSED.json` 后进行离线审计，分别报告恢复、handoff 和完整任务成功。

通用启动参数（占位值需由运行器填写，**不是给在线 Agent 的输入**）：

```bash
python3 scripts/isolated_online_bridge.py \
  --host SSH_ALIAS \
  --remote-root /absolute/H20/package \
  --remote-run /absolute/H20/fresh_run \
  --seed SEED --cosmos-port PORT \
  --capsule /absolute/local/fresh_online_capsule \
  --private /absolute/local/fresh_evaluator_logs \
  --port 18791
```

Capsule 和 evaluator 目录必须是新目录。后端必须存在对应 seed 的原始策略前缀。bridge 不启动或终止模型服务，不自动选择恢复动作，也不内置特定 LLM/API key。它保留了原 `cr.py live` 入口；新的 Agent 通过受限接口驱动这个入口。

## 可审计的输入边界

bridge 对 critic 提案做字段白名单，只提供 RGB 派生证据；不提供 reward、contact、真实物体位姿、measured joints 或审计文件。几何工具只允许引用本次已交付给执行端的观察帧。记录：

- `frozen_input_manifest.json`：skill 文件哈希、预先声明的输入范围。
- `agent_exposure.jsonl`：在线接口返回的状态和工具结果。
- `agent_decisions.jsonl`：独立执行 Agent 提交的每个动作/测量及理由。
- H20 run：原视频、控制来源、恢复与 Cosmos 请求记录，关闭后才开放给评估端。

每次只允许一条在途请求，拒绝重复 ID、过期 frame、未见过的历史图像与接口外工具；100 次在线决策后仅允许 `finish`。一轮恢复和 1400 仿真步数为实验协议；一轮恢复限制仍需执行 Agent 遵守。

注意这是**上下文和接口输入隔离**，不是操作系统级权限沙箱。同一 Codex 工作区的 subagent 仍具有通用工具能力，需遵守不读 capsule 外材料的约定。若需要可强制验证的防泄漏实验，应在独立容器/用户权限下只挂载 capsule，并只暴露工具服务，而不是声称共享工作区已具备这种隔离。

同一开发 seed 上的独立 Agent 试验可以验证上下文分离和 skill 是否足够自包含，但不是 held-out seed 泛化，也不是统计成功率。原策略前缀回放与新 Agent 恢复必须分别标注。
