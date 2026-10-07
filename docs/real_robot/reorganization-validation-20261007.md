# 整理验证记录（2026-10-07）

整理前基线：`91c6370`。本次仅整理源码和文档；没有启动机器人、拍摄新数据或执行新的 rollout。

## 本地验证

- 六组相关测试共 **45 passed**：continuation、pregrasp commissioning、bundle 执行、真机输入契约、收尾、真机进化审计。
- 与基线源码做 AST 核对：`RealCoreFactory`、continuation `run`/`main` 的运行逻辑等价；文件移动只调整仓库根目录的 `parents` 深度。
- `grasp_continuation.py` 仅格式化，整文件 AST 与基线相同。
- 九个统一入口的子命令 `--help` 均退出 0。
- 冻结的 provider/bundle/hardware/标定等没有改写；旧实验的 source SHA 继续代表旧提交。

本地 macOS 的完整配置预检遇到硬件配置中已冻结的 dodo 绝对标定路径，因此在文件解析阶段拒绝。该预检需要在 dodo 执行，不能通过改写冻结路径来冒充同一配置。

## dodo 验证

待源码同步后执行 `arx_real.py check`；仅校验文件、工具注册和输入契约，不打开硬件。

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
