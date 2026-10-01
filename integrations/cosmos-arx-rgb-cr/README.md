# Cosmos ARX C&R — H20 复现包 v1.0.0

交付日期：2026-09-21。用于 MuJoCo ARX test-tube pickup 的 **RGB critic → Agent 选择 EEF 工具 → pregrasp 检查 → 新 Cosmos 推理**。

这不是“一条命令自动成功抓取”的模型。当前 3 个回归 seed 中，2 条完成接回，0 条完整稳定 pickup。包保留了已验证的 v4.1 在线逻辑和失败证据，适合复现、研究和继续开发；**不用于真机**。

## 交付了什么

- `cr.py`：统一命令行，配置检查、服务启动、在线执行、确定性恢复回放、离线审计、视频标注。
- `vendor/`：10 个冻结在线文件字节不变，外加测试和离线工具。保留 v1/v2/v3/v4 相对目录是为了兼容历史 import，不是要求按四套方法运行。
- `SKILL.md`、`docs/agent_protocol.md`：可交给同学自己的多模态 Agent，规定观察边界、工具参数与恢复决策流程。
- `fixtures/seed_183173/`：原失败轨迹前缀、已记录恢复到 f723 的动作、Agent 理由和 RGB 检查点。
- `reference_results/`：43935、183173、344246 的完整控制来源标注视频和离线汇总；在线新实验时不要读取离线评价来选动作。
- `provenance/`、`MANIFEST.json`：冻结哈希、依赖与外部代码记录。

**不包含** Cosmos 大权重、VAE、processor、Agentic-Embodied 全仓库/场景资源或整个 Python 环境；同一台 H20 复用现存资源。也不包含内置 LLM/API key。恢复决策由外部 Agent（或人工操作者）提供；接上任意支持看 PNG、调用终端并输出 JSON 的 Agent 即可。

## H20 快速开始

登录已经能访问这些共享资源的 H20 账号，把压缩包解压到**自己的可写目录**，不要覆盖原实验目录。

```bash
tar -xzf cosmos-arx-rgb-cr-h20-v1.0.0.tar.gz
cd cosmos-arx-rgb-cr
python3 cr.py verify
python3 cr.py doctor
python3 cr.py test
```

`python3` 仅负责标准库启动器，仿真实际使用 `config.h20.json` 的 client Python 3.10，模型服务使用独立的 server Python。不要把两个环境合并或向共享 venv 直接 pip install。

默认路径：

| 配置项 | H20 默认值 |
|---|---|
| Agentic-Embodied | `/data4/zhengyikai/Agentic-Embodied` |
| 场景 | 上述仓库的 `runs/arx_pickup_test_tube/pickup_test_tube_initial_scene` |
| client Python | `/data4/dingxin/cosmos3_edge_arx_task7_runtime/client_py310/bin/python` |
| Cosmos bundle | `/data4/zhengyikai/cosmos3-edge-arx5-inference` |
| server Python | 上述 bundle 的 `.venv/bin/python` |
| server 依赖 overlay | `/data4/dingxin/loop2_stage2_c2_phase_v3/server_deps` |
| 微调 checkpoint | `/data4/zhengyikai/ckpt1/model` |
| 模型配置 | `/data4/zhengyikai/ckpt1/config/config.yaml` |
| GPU / 端口 | `2` / `15591`，可命令行覆盖 |

同学账号需要这些目录的读/执行权限；本包不会 chmod 他人的目录。路径不同就复制并编辑配置，所有命令都支持 `python3 cr.py --config /绝对路径/my-config.json ...`。先用 `doctor` 定位缺失资源。

### 1. 先做不加载 Cosmos 的恢复回归

```bash
python3 cr.py smoke --output runs/smoke_183173 \
  --approval "我批准重放已记录的 seed183173 恢复，仅作环境与门控复现"
python3 cr.py audit runs/smoke_183173
python3 cr.py annotate runs/smoke_183173
```

预期：从环境 reset 开始重放已记录动作，到 **f723**；每个已保存 RGB 检查点逐像素相同，pregrasp gate eligible=true；然后主动关闭。`smoke_result.json` 记录通过结果。它不需要 GPU 模型服务，不做新的 Agent 决策，也不会自动接回 Cosmos。它不表示 pickup 成功。

### 2. 启动 Cosmos（终端 A）

```bash
nvidia-smi
python3 cr.py server --gpu 2 --port 15591 --output runs/server_15591
```

前台运行，等待 `Server is ready and listening` 或 `runs/server_15591/ready.json`。启动器拒绝复用已有输出目录、被占端口和显存占用超过 1 GiB 的 GPU；仍应确认调度/使用权限。完成后在此终端 Ctrl-C，仅停止自己的服务。

### 3a. 真正的新 Agent 恢复（终端 B，推荐用于评估）

```bash
python3 cr.py live --seed 183173 --port 15591 --output runs/live_183173
```

会从原失败轨迹动作前缀精确复现到首个 critic 提案，然后暂停。把 `SKILL.md` 和 `docs/agent_protocol.md` 提供给 Agent；读取输出 JSON 指向的三张 RGB PNG，由 Agent 决定并向同一进程 stdin 写入一行 JSON。每个动作结束都重新读图，不能批量照抄旧 seed 的动作。

输入必须保持打开。不要用 `echo ... | ...` 期待持续交互；stdin EOF 会关闭本次仿真。自动化客户端用 `subprocess.Popen(..., stdin=PIPE, stdout=PIPE)` 持有进程，SSH 则使用保持交互的终端。避免在终端粘贴很长 JSON，推荐 Agent 客户端通过管道写入完整行。

提供了可选的标准库薄客户端 `scripts/agent_client.py`：自己的 Agent 先 `wait()` 取观察、加载 PNG，再显式 `send(decision)`。它不包含 LLM，也不选择恢复动作；超时后继续 wait，不自动重发。replay 的中间检查点不是可发送动作的边界，应等待 `approved RGB-verified state reproduction complete`。

其他 seed 需要原轨迹：

```bash
python3 cr.py live --seed 43935 --output runs/live_43935 \
  --prefix /data4/dingxin/cosmos3_edge_arx_task7_runtime/rollout_40_same_env_20260917_osmesa_v2/batch_20260917T074634_770167Z/seed_43935/trajectory.npz
```

### 3b. 快速复现到已有 pregrasp，再做新 Cosmos 接回

```bash
python3 cr.py replay-pregrasp --seed 183173 --port 15591 \
  --output runs/replay_then_cosmos_183173 \
  --approval "我批准重放已记录恢复到 f723；接回前由 Agent 重新审阅 RGB"
```

到 f723 后暂停，Agent 重新看图、请求 `review_pregrasp`、引用新提案 ID 和五项视觉检查后才能 `resume_vla`。后续 Cosmos 是新推理；此前的恢复是回放，不能计作新 seed 的独立恢复实验。具体 JSON 见协议文档。

### 4. 结束、审计、视频

先发送 `finish`，等输出保存完成。批量实验必须等**全部在线 trial 关闭**再读内部评估。

```bash
python3 cr.py audit runs/live_183173
python3 cr.py annotate runs/live_183173
```

主要文件：`three_view_controller_labels.mp4`（原 Cosmos / Agent C&R / 接回后 Cosmos 标签），`agent_decisions.jsonl`、`cosmos_requests.jsonl`、`validated_reentries.jsonl`、`delivery_audit.json`。原视频不覆盖。`CLOSED.json` 表示完整关闭；异常关闭会拒绝正常验收。

## 对照结果与限制

| Seed | EEF 工具次数 | Gate frame | 新 Cosmos 步数 | 本批结果 |
|---|---:|---:|---:|---|
| 43935 | 45 | 1088 | 312 | 1400 步预算结束，未抓起 |
| 183173 | 26 | 723 | 144 | 再次空抓，单次恢复结束 |
| 344246 | 0 | 无 | 0 | 已翻架，当前 pregrasp 恢复不覆盖 |

参照视频在 `reference_results/seed_<SEED>/three_view_controller_labels.mp4`。完整任务必须同时满足视觉验收和离线稳定完整拔出检查，不能把 10 mm 抬升或 gate 通过当作任务完成。

critic 的颜色/指尖模板和 pregrasp 阈值只针对这个场景，存在检测延迟、遮挡和分布适配问题。EEF 工具不是碰撞规划器；恢复可以很慢。当前证据不能把失败完全归因于 Cosmos，也不能宣称 C&R 无关。

详见 [Agent 协议](docs/agent_protocol.md)、[复现与故障排查](docs/reproduction.md)、[交付验证记录](docs/validation.md)。
