# 复现边界与故障排查

## 复现等级

1. **verify/test**：文件完整性和逻辑单测，不加载场景或模型。
2. **doctor**：路径、client import、MuJoCo 场景编译及外部关键文件哈希；不做模型推理。
3. **smoke**：reset + 已记录 seed183173 前缀与恢复，逐像素验证 29 个保存的 RGB 检查点（实际数量以 `approved_replay.jsonl` 为准），f723 gate 检查后结束；无新 Agent 决策、无 Cosmos 推理。
4. **replay-pregrasp + Agent 审核 + resume_vla**：相同已记录恢复后做新的 Cosmos 接手。模型版本/软件/GPU 改变可能导致动作不同，不保证后续像素逐帧相同。
5. **live**：原失败前缀后由同学自己的 Agent 实时选择恢复。不同 Agent 可能选择不同合法动作，因此不承诺复现相同恢复轨迹或成功率。

观察文件中的 RGB 路径会写当前输出绝对路径。fixture 中历史 JSON 保留历史路径作溯源，回放器按 fixture 的相对 `frame_XXXX` 读取 PNG，不依赖旧输出目录。代码不依赖旧 v1/v2/v3/v4 安装目录，仅 H20 外部仓库/模型/环境引用现有配置。

## 常见问题

| 现象 | 处理 |
|---|---|
| doctor 缺文件/权限 | 确认同一台 H20，联系资源拥有者授予读/执行权限，或复制配置指向自己的资源；不要更改他人目录权限 |
| python import / libstdc++ 错误 | 用 `cr.py` 调度 client_python；它设置 OSMesa、预加载配置中的系统 libstdc++。不要用 server Python 跑仿真 |
| pandas/dateutil 等 server import 失败 | 配置现有 `server_deps` overlay；包不自动安装或修改共享 venv。完整 server 环境不能由简化 client 依赖文件替代 |
| 原目录已存在 | 换新的 `--output`，不会覆盖旧实验。不要删除他人的输出 |
| RGB reproduction diverged | 停止，不接回 Cosmos；比较外部文件哈希、OSMesa/MuJoCo/numpy、原场景资源。严格像素对照不得忽略 |
| gate 返回 unknown/不通过 | 查看原因和 RGB；不能复制老 proposal ID 或修改门槛凑通过。可选择有证据的小步观察或结束 |
| stale frame / duplicate ID | 等上一请求结束，读最新观察，用新 ID 和真实 frame；review 不增加 frame |
| JSON 粘贴后没响应 | PTY 行长度可能截断；使用保持 stdin 的进程管道发送完整 JSON 行，或缩短 ASCII 理由。不要在旧请求未结束时重复发动作 |
| Cosmos timeout | 检查同一端口服务是否 ready、自己的 server 日志；不要自动重复可能已部分执行的动作。检查最新实际 frame/request log，由 Agent 决定下一步 |
| 没有 policy 请求文件 | 可能尚未接回、只做 smoke，或翻架停止，属于有效结果而不是假造一次请求 |
| audit 拒绝 | 要求 session 正常关闭、存在 CLOSED.json；批量还需人工确保全部 trial 都已关闭。不要伪造关闭文件 |
| annotate 找不到 ffmpeg | 优先系统 ffmpeg，否则自动使用 client 环境的 imageio-ffmpeg 二进制；doctor 会显示实际路径 |
| 显示 eligible=true 但抓不住 | 符合当前已知限制：候选 pregrasp 检查并非抓取/策略可接手的充分条件 |

## 环境依赖

默认直接复用 H20 已验证的两套解释器。`provenance/environment.json` 记录交付时实际版本；`external_files.json` 记录关键仓库代码、场景文本、模型配置的 SHA256。模型大权重和网格没有复制进包，也不是全量依赖锁文件；移到全新机器不属于本包已验证范围。

`config.h20.json` 是随包默认配置，受 MANIFEST 保护；自定义时复制成新文件，再通过全局 `--config` 选择，避免修改受校验的默认文件。external hashes 不匹配意味着不是同一已验证环境，应查明后建立新的对照，不要直接删除检查。

启动器覆盖 PYTHONPATH 为任务需要的仓库/overlay，不继承不明路径；不改系统配置、共享 Python 环境或 GPU 上其他进程。CUDA 服务只绑定 localhost，可经同机 client 访问。

## 文件与数据使用

恢复 fixture 是对本次实验自己记录的动作和 RGB 的裁剪，不读取私有 trace 来恢复状态；执行器从 reset 正向模拟。原 prefix NPZ 保留原运行数据供执行器做离线对照，Agent 不应在线分析其中状态数组。模型/外部仓库遵守各自许可与组内访问授权；本包不授予重新分发模型权重的许可。

最小分享单元是整个压缩包，而不是只拷贝 `SKILL.md`。Agent 可直接读取这个 SKILL.md，无需安装插件；若其工具没有持续 stdin 功能，写一个持有子进程管道的薄客户端即可，不能把工具层改成自行选恢复动作的脚本。
