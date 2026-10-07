# 整理验证记录（2026-10-07）

整理前基线：`91c6370`。本次仅整理源码和文档；没有启动机器人、拍摄新数据或执行新的 rollout。

## 本地验证

- 六组相关测试共 **45 passed**：continuation、pregrasp commissioning、bundle 执行、真机输入契约、收尾、真机进化审计。
- 与基线源码做 AST 核对：`RealCoreFactory`、continuation `run`/`main` 的运行逻辑等价；文件移动只调整仓库根目录的 `parents` 深度。
- `grasp_continuation.py` 仅格式化，整文件 AST 与基线相同。
- 九个统一入口的子命令 `--help` 均退出 0。
- 十份当前冻结文件的实际 SHA 与部署索引一致，当前指南中的 Markdown 文件链接存在。
- 冻结的 provider/bundle/hardware/标定等没有改写；旧实验的 source SHA 继续代表旧提交。

本地 macOS 的完整配置预检遇到硬件配置中已冻结的 dodo 绝对标定路径，因此在文件解析阶段拒绝。该预检需要在 dodo 执行，不能通过改写冻结路径来冒充同一配置。

## dodo 验证

源码 `2ed25fb` 同步到 dodo 后，使用当前部署页中的完整命令执行 `arx_real.py check`：退出 0，`eligible=true`，`hardware_opened=false`。注册/编译了同一个15调用恢复程序，bundle/provider/输入契约 SHA 与当前冻结文件匹配。见[预检原始输出](evidence/dodo-preflight-2ed25fb.json)。

控制器管理脚本 `status` 返回退出码 1、`not running`。机械臂控制保持停止，见[状态记录](evidence/dodo-controller-status-2ed25fb.json)。代码与记录同步到 aigc31、dodo 和用户 GitHub 仓库。

## 复查命令

```bash
python -m pytest -q \
  tests/test_arx_grasp_continuation.py \
  tests/test_arx_pregrasp_commission.py \
  tests/test_arx_bundle_execution.py \
  tests/test_arx_real_input.py \
  tests/test_arx_real_post_episode.py \
  tests/test_arx_real_evolution_audit.py
python scripts/deployment/arx_real.py --help
git diff --check
```

验证范围是源码整理与冻结配置的可加载性。真实机器人运行能力以原始实验记录为依据；当前整理不增加连续自主成功、泛化或晋升证据。
