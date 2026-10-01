# 主视角 + 腕部 critic 更新（2026-09-22）

## 修改范围

当前 `PhaseAwareRgbCritic` 仅使用 `front_rgb`（主固定视角）和 `right_rgb`（腕部）。允许直接传两张图；若传三张，`left_rgb` 不参与初始化基准、特征提取或判定。proposal 的 `critic_cameras` 字段记录实际使用的视角。

抓取尝试仍需腕部接近目标、可靠夹指模板匹配及连续闭合证据。随后，腕部目标面积缩小且中心移动，同时主视角确认目标相对黄色架子的初始关系未明显变化、连续稳定，才能提交 `attempted_grasp_target_left_behind`。去除了原来必须由左视角再次确认的要求；其他数值阈值保持不变。少一个固定视角意味着交叉验证减少，需要重新评估误报/漏报，不能沿用旧版成功率。

抓取尝试后腕部目标缺失、主视角也无法确认目标在架子上，持续 30 帧会提交 `grasp_outcome_unknown`，不把遮挡直接当失败。critic 只提交 proposal；恢复工具与接回 Cosmos 仍由 Agent 决定。

还修复了 Session 初始化顺序：现在通过子类的 `critic_type` 直接创建阶段 critic，不再先构造会检查左视角的旧 `EarlyRgbCritic`。

## 保留的约束

- reset 时主视角和腕部各需至少 20 个满足粉色阈值的像素；不足时异常明确列出对应相机。
- 两路有效输入为 `uint8 H×W×3` RGB，尺寸不能在运行中变化；腕部必须为 320×240，夹指模板仍与原 ARX 相机布局绑定。
- 黄色架子是主视角确认目标仍受支撑的参照。它不可见或基准无效时，不能可靠确认该失败条件。
- **不是整个系统改成两相机**：环境加载、Cosmos、录像、已记录的三视角逐像素回放，以及 recovery 几何工具保留三路接口。可保留左相机但允许它看不到目标；删除相机本身仍需要另行适配。
- pregrasp gate 原本就允许任意一个固定视角提供稳定支撑证据，左视角无目标不会单独造成初始化异常。深度仍需足够的有效双视角证据，或 Agent 显式选择且通过检查的时序 RGB；未放松恢复门控。
- 本补丁不修改场景、模型权重、EEF 工具参数或自动选择恢复动作。

## 验证方式与历史证据

`python3 cr.py test` 包含左图缺失/全黑/错误格式不影响 critic、仅两图输入、必需视角缺失与格式保护、运行中遮挡，以及抓取尝试/离开/未知的时序测试。

2026-09-22 本地与 H20 验证均通过 47 项单元测试（17 项 critic、24 项 gate/离线审计、6 项交付接口）。H20 另用 `scripts/check_front_wrist_reset.py` 在实际 MuJoCo 环境将左路观测持续置黑，完成 `PregraspSession` 初始化、6 步 hold 和正常关闭；两视角 critic 正常，左视角 gate 基准为缺失，Cosmos 请求为 0。运行记录位于交付目录的 `runs/front_wrist_black_left_reset_20260922`。这仅验证初始化和接口回归，尚未完成两视角版完整抓取 rollout 或误报率评估。

`fixtures/`、`reference_results/` 和 `provenance/` 中的旧实验记录仍是原 v1.0.0/v4.1 的历史证据，不是本补丁的完整 rollout 验证。`MANIFEST.json` 才是当前包的文件完整性清单。旧的 `scripts/build_payload.py` 是原始打包流程，不应用它覆盖当前补丁。
