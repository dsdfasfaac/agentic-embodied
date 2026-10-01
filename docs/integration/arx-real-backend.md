# ARX 真机后端

入口是 scripts/deployment/serve_arx_real_gateway.py；实现位于 robots/arx/gateway/real_backend.py。它沿用 gateway Backend 的 reset/step/close 边界，并把机械臂与相机分别抽象为 ArmDevice 和 CameraSource。新增其他机械臂时，只需适配到 14D 模型状态/命令和带时间戳的反馈；gateway 与同步/到位逻辑不依赖 ARX SDK。

## dodo 的当前连接

2026-10-01 只读检查显示 can1、can3 在线，两个 ARX ROS2 控制器分别作为 /arm_slave_l、/arm_slave_r 运行，状态话题是 /arm_slave_l_status、/arm_slave_r_status，命令话题是 /arm_master_l_status、/arm_master_r_status。因此 dodo 应使用 arm_transport=arx_ros2。arx_sdk 适配器供独占 CAN、没有 ROS 控制器的场景使用；两个控制栈不能同时控制同一 CAN 设备。官方 SingleArm 在该机的 /home/dodo/chenfu/ARX_X5/py/arx_x5_python/bimanual 中。

dodo 的 pyrealsense2 实时枚举确认了手册映射：front=260422272500、left=260422271945、right=260422275847。/dev/v4l/by-id 还列出另一组三台 261123… 设备，不能据此替换 RealSense SDK 使用的映射。仓库保存了从上述三台设备的 640x480@15 RGB stream profile 只读获取的三份内参快照（robots/arx/manifests/real/）。相机外参另见本仓库 docs/arx_camera.md：主相机到左臂局部基座的矩阵为实测、左腕手眼外参为实测，右腕手眼外参是由左腕同构推导、尚未经右腕独立验收。已把 dodo 运行时 front 原始标定复制为 dodo_front_d405_rgbd_calibration_BL_source.json，保留原文件 SHA 854854c1d0e512ccfe6411c1d3ebf8b74394f9138e1a713f4660169ab6cded10。原文件的 T_B_from_C 实际指向左臂局部基座 BL，不能当成整机基座 BA；如需在整机坐标计算，须采用 docs/arx_camera.md 中已验收的 BL→BA 迁移。当前后端只消费 RGB 内参和 14D 状态，不消费外参；不能仅凭这些数据声称真机目标到夹爪距离已经可用。启动会核对标定文件 SHA、内部 camera.logical_name、camera.serial、camera.width/height 和运行时内参；不匹配直接拒绝。模型契约中的 calibration_id 是训练接口标签，真实内参的身份由这份文件的 SHA 与物理序列号共同确定。

## 官方 AC one URDF 关节范围

从 [ARX_Model 官方 AC one URDF 压缩包](https://github.com/ARXroboticsX/ARX_Model/blob/1857d3b5796f3a8b11a5d86d66be6762963d8d12/AC%20one/URDF/AC%20one.7z) 中的 acone.urdf 读取了 6 个转动关节的范围，并把来源 commit、压缩包及 URDF SHA 连同数值存入 robots/arx/manifests/real/ac_one_urdf_limits.json。左右臂相同，按关节 1～6 顺序，单位 rad：

| 关节 | 下界 | 上界 |
| --- | ---: | ---: |
| 1 | -2.094 | 3.1416 |
| 2 | 0 | 3.665 |
| 3 | 0 | 3.24 |
| 4 | -1.671 | 1.671 |
| 5 | -1.671 | 1.671 |
| 6 | -2.094 | 2.094 |

URDF 的左右夹爪各有两个直动手指关节，范围均为 0～0.044 m。这不是 ROS2 RobotStatus.joint_pos 中夹爪原生坐标的范围；当前没有经验证的二者转换。dodo 控制器自带的 x5_2025.urdf 对六关节写的是宽泛的 -10～10 rad，不能据此覆盖 AC one CAD 范围。AC one URDF 与当前 SDK/ROS2 的关节零位和符号尚未在真机逐轴核对，所以这些值是配置参考，不自动写入命令界限；部署配置仍需明确冻结控制器坐标下的边界。

## 硬件契约

启动需要冻结的 JSON 硬件配置和从独立部署记录取得的文件 SHA-256。schema_version 是 arx.real.hardware.v1。主要字段如下：

| 字段 | 含义 |
| --- | --- |
| arm_transport | arx_ros2 或 arx_sdk；dodo 用 arx_ros2 |
| left/right | CAN 口、SDK 型号、6 个关节的弧度上下界、SDK 原生夹爪范围及到模型夹爪坐标的仿射变换 |
| command_left/right | 是否实际发送该臂；必须与任务清单的锁臂设置一致 |
| cameras | 按 front_rgb、left_rgb、right_rgb 顺序给出序列号、标定 ID/文件/SHA、模型输出尺寸、采集尺寸和帧率 |
| timing.control_hz | gateway 接受新关节目标的频率，必须等于模型契约的 15 Hz |
| timing.max_sensor_skew_ms / max_sensor_age_ms | 三路 RGB 和 14D 状态的同步与新鲜度上限 |
| timing.observation_timeout_s / arrival_timeout_s / feedback_poll_s | 观测等待、实测到位等待及轮询间隔 |
| timing.position_tolerance | 14D 到位容差；12 个关节以弧度计，两个夹爪以模型坐标计 |
| right_gripper_closed_policy/open_policy | 恢复工具 opening=0/1 的两个已标定模型坐标端点 |

模型前 6 + 后 6 个关节以 rad 表示；两个夹爪在 14D 模型中的单位是 policy_gripper 原始位置坐标。ARX SDK 对 2023/2025 夹爪给出的原生范围不同，实际机械开口宽度单位未由当前接口提供。部署必须显式填写 gripper_policy_scale 和 gripper_policy_offset，满足 policy_gripper = native * scale + offset，不会猜测米数或默认把原生值等同模型值。

D405 RGB 采集可使用 640x480，再按配置缩放为模型要求的 320x240 uint8 RGB。时间戳是主机 monotonic 时钟下的取帧/ROS 消息接收时间，不冒充硬件曝光时间。RealSenseCameraSource 会按序列号打开三台相机并选取偏差不超过配置上限的帧组；有硬件触发的相机可替换 CameraSource。

## 命令与反馈

reset 只读取同步观测，不回零或移动。每个 step 先经过现有 ActionProcessor 的任务锁定、平滑和步长限制，再以配置的 15 Hz 上限发送。事件日志依次记录 command_dispatch_started、command_sent、arrival_observed 或 arrival_unverified。命令回执只表示 SingleArm setter 或 ROS2 publish 已返回，不表示 CAN 已确认。到位结论由命令之后的新鲜 14D 反馈与 14 维容差比较得出。未实测到位、相机不同步、反馈陈旧或设备读数异常时，gateway 进入 EXECUTION_UNCERTAIN 并停止继续派发目标。

公开观测包含主机时间戳、三路相机标定 SHA、传感器年龄/偏差和设备健康。当前 SDK/ROS2 RobotStatus 没有可信的电机故障位或硬件 CAN ACK，因此 health 只报告“新鲜响应”及 diagnostics_available=false；外部硬件急停仍须独立存在。ROS2 适配器在命令发出后以 30 Hz 重发同一目标，租约默认 0.25 秒，到期停止重发；它不会把重发误记为实测到位。

## 启动范围

服务入口复用现有 EpisodeWorker、HTTP gateway、CosmosPredictor 和工具目录；当前只启用纯 VLA baseline。带 CandidateBundle 的真机 critic provider 和 reentry evaluator 尚未实现，不能通过这个入口启用示例 privileged bundle。实现这些 provider 后须先通过 real_input.py 的 bundle 输入预检。

先填写硬件配置中的实际关节上下界、夹爪原生范围和模型坐标变换，再计算配置文件 SHA。用 --check-config 校验冻结配置；此模式不打开 CAN、ROS2 或相机，也不需要 runtime-config/output。真正运行前还需在 dodo 的 Python 3.12 环境中加载 ARX ROS2 工作空间，并把此 Git 分支的代码部署到 dodo。当前没有向真机发送过运动指令；测试使用假机械臂、假相机和假 ROS2 话题。
