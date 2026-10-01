# ARX X5 三相机坐标关系交付说明

生成日期：2026-09-07  
适用设备：AC one 双臂 ARX X5，三台 Intel RealSense D405  
用途：向下游明确交付主相机、左右腕相机、整机基座、左右末端基座与 TCP 之间的坐标关系。

> 重要：本文严格区分“实测标定”“由机械同构关系推导”和“随关节角变化的动态变换”。
> 右腕手眼外参由左腕结果复制推导，尚未经过右腕独立标定或验证，不能标记为实测
> `accepted: true`。
>
> **基座更正：**主相机标定时读取的是左臂控制器 `/arm_master_l_status`，因此原始文件中笼统
> 写成 `B` 的坐标系实际是左臂局部根 `BL = arx_x5_left_base_link`，不是 AC one 整机中心
> `BA = acone_base_link`。本文同时给出原始实测矩阵和迁移到 `BA` 后的矩阵，二者不得混用。

## 1. 统一记号

本文使用列向量和齐次变换：

```text
p_A = T_A_from_B · p_B
```

因此，`T_A_from_B` 把 B 坐标系中的点转换到 A 坐标系。组合顺序为：

```text
T_A_from_C = T_A_from_B · T_B_from_C
```

全部平移单位为米，矩阵最后一行为 `[0, 0, 0, 1]`。

坐标系名称：

| 符号 | 坐标系 |
|---|---|
| `BA` | AC one 整机中心基座 `acone_base_link`；全身模型、MuJoCo world 的统一基座 |
| `BL` | 左臂局部基座 `arx_x5_left_base_link`；主相机标定时左臂控制器所用根坐标系 |
| `C0` | 固定主相机 RGB optical frame |
| `GL` | 左臂 `gripper_base` |
| `GR` | 右臂 `gripper_base` |
| `TL` | 左臂模型 TCP site |
| `TR` | 右臂模型 TCP site |
| `CL` | 左腕相机 RGB optical frame |
| `CR` | 右腕相机 RGB optical frame |

所有 RealSense RGB optical frame 均采用：

```text
x：图像向右
y：图像向下
z：光轴向前
```

## 2. 相机身份和标定状态

| 相机 | 数据/模型名称 | 运行时别名 | RealSense 序列号 | 状态 |
|---|---|---|---|---|
| 固定主相机 `C0` | `base_0_rgb` | `front_rgb` | `260422272500` | 内参、`C0 → BL` 外参均为实测且 accepted；`C0 → BA` 由已验收的基座迁移关系换算 |
| 左腕相机 `CL` | `left_wrist_0_rgb` | `left_rgb` | `260422271945` | 内参、手眼外参均为实测；独立验证 accepted |
| 右腕相机 `CR` | `right_wrist_0_rgb` | `right_rgb` | `260422275847` | 手眼外参由左腕同构复制，未独立验证；独立内参未找到 |

名称和序列号的代码来源：

```text
data_collect/compare_live_to_raw_frame.py
dodo-threetask-mixed-runtime-v2/scripts/downstream/dodo/live_pi05_dodo_h64_joint.py
```

## 3. 相机内参

### 3.1 固定主相机 `C0`

分辨率：`640 × 480`

```text
K_C0 =
[[395.3048095703125,   0,                 318.5152587890625],
 [  0,                 395.0485534667969, 239.1984100341797],
 [  0,                   0,                 1               ]]
```

畸变模型：`inverse_brown_conrady`

```text
D_C0 =
[-0.05327928811311722,
  0.05796921253204346,
  0.0021578031592071056,
  0.00018020032439380884,
 -0.019658749923110008]
```

来源：`inference/assets/ac_one/front_d405_rgbd_calibration.json`

### 3.2 左腕相机 `CL`

手眼标定的 15 个样本均记录了同一组内参：

```text
K_CL =
[[396.2483825683594,   0,                319.63616943359375],
 [  0,                 395.9893798828125, 238.58644104003906],
 [  0,                   0,                 1                ]]
```

畸变系数：

```text
D_CL =
[-0.05257856473326683,
  0.0549410842359066,
  0.0014854990877211094,
 -0.00008131132199196145,
 -0.017545683309435844]
```

来源：`/home/dodo/skill_factory/artifacts/hardware_calibration/dataset/sample_*/sample.json`

### 3.3 右腕相机 `CR`

仓库中没有找到序列号 `260422275847` 的独立内参记录。右腕虽然也是 D405，但设备级
`fx/fy/cx/cy` 和畸变系数不能由左腕或主相机可靠推导。使用 RGB-D 几何前，应从右腕设备的
librealsense profile 读取并固化对应分辨率下的内参。

## 4. 固定主相机与两个基座

### 4.1 原始实测：相机到左臂局部基座 `T_BL_from_C0`

原标定文件将目标系简写为 `B`，但标定程序订阅 `/arm_master_l_status`，控制器按左臂单臂
URDF 求解 `RobotStatus.end_pos`。因此该矩阵的准确语义是 `T_BL_from_C0`：

```text
T_BL_from_C0 =
[[ 0.021281523621, -0.681502371290,  0.731506400982,  0.013180862163],
 [-0.999695047109, -0.023672815373,  0.007029267195, -0.237032725735],
 [ 0.012526353713, -0.731432919507, -0.681798338752,  0.128294381911],
 [ 0,               0,               0,               1             ]]
```

```text
p_BL = T_BL_from_C0 · p_C0
```

其反方向为：

```text
T_C0_from_BL = inverse(T_BL_from_C0) =
[[ 0.021281523621, -0.999695047109,  0.012526353713, -0.238848011556],
 [-0.681502371290, -0.023672815373, -0.731432919507,  0.097210291184],
 [ 0.731506400982,  0.007029267195, -0.681798338752,  0.079495177779],
 [ 0,               0,               0,               1             ]]
```

原始标定质量不因坐标系名称更正而改变：

```text
config/arx_x5_front_main_camera_extrinsic.json
样本：15，总内点：12
平移残差 RMS：0.001263245277 m
旋转残差 RMS：1.421836282 deg
accepted：true
```

### 4.2 左臂局部基座到整机基座 `T_BA_from_BL`

`skill_factory` 的正式 frame contract 定义：

```text
T_BA_from_BL = translation([-0.01602, +0.25, -0.0388])

T_BA_from_BL =
[[1, 0, 0, -0.01602],
 [0, 1, 0,  0.25   ],
 [0, 0, 1, -0.0388 ],
 [0, 0, 0,  1      ]]
```

此迁移关系已经用 15 组姿态对单臂模型和 AC one 全身模型做过对照，最大平移差
`0.002787219628 m`，最大旋转差 `0.000841835 deg`，验收结果为 `accepted: true`。
由于两版模型的几何参数存在毫米级差异，这是一条**已验收的模型迁移约定**，不应描述成两套
URDF 在所有关节姿态下严格恒等。

### 4.3 正确交付给 AC one 全身模型：`T_BA_from_C0`

组合关系：

```text
T_BA_from_C0 = T_BA_from_BL · T_BL_from_C0
```

数值为：

```text
T_BA_from_C0 =
[[ 0.021281523621, -0.681502371290,  0.731506400982, -0.002839137837],
 [-0.999695047109, -0.023672815373,  0.007029267195,  0.012967274265],
 [ 0.012526353713, -0.731432919507, -0.681798338752,  0.089494381911],
 [ 0,               0,               0,               1             ]]
```

```text
p_BA = T_BA_from_C0 · p_C0
```

即主相机光心在 `acone_base_link` 中的位置为：

```text
[-0.002839137837, 0.012967274265, 0.089494381911] m
```

### 4.4 整机基座到主相机 `T_C0_from_BA`

```text
T_C0_from_BA = inverse(T_BA_from_C0) =
[[ 0.021281523621, -0.999695047109,  0.012526353713, 0.011902702753],
 [-0.681502371290, -0.023672815373, -0.731432919507, 0.063831229762],
 [ 0.731506400982,  0.007029267195, -0.681798338752, 0.063002817980],
 [ 0,               0,               0,              1             ]]
```

上述固定外参只在相机支架、相机本体和机器人基座的物理安装关系不变时有效。

### 4.5 运行时文件警告

`inference/assets/ac_one/front_d405_rgbd_calibration.json` 当前仍保存 4.1 的旧数值，即
`T_BL_from_C0`，不能仅凭字段名将其解释成 `T_BA_from_C0`。本文此次只修订交付说明，未修改
运行配置。若要更新运行链路，还需同时审查 `scene_reconstruction.py` 的基座解释以及现有
`simulation_xy_from_base_xy` 的 Y 轴翻转，避免重复换轴或重复平移。

## 5. 左腕相机与左臂末端

### 5.1 左腕相机到左 `gripper_base`：`T_GL_from_CL`

这是实测手眼标定结果：

```text
T_GL_from_CL =
[[-0.036274539936, -0.502952849196,  0.863552308339, 0.084796927438],
 [-0.998958456762, -0.005684868872, -0.045273435150, 0.035831938409],
 [ 0.027679584839, -0.864295154303, -0.502222786053, 0.070030470897],
 [ 0,               0,               0,              1             ]]
```

```text
p_GL = T_GL_from_CL · p_CL
```

相机光心在 `GL` 中的位置为：

```text
[0.084796927438, 0.035831938409, 0.070030470897] m
```

### 5.2 左 `gripper_base` 到左腕相机：`T_CL_from_GL`

```text
T_CL_from_GL = inverse(T_GL_from_CL) =
[[-0.036274539936, -0.998958456762,  0.027679584839,  0.036932173066],
 [-0.502952849196, -0.005684868872, -0.864295154303,  0.103379552779],
 [ 0.863552308339, -0.045273435150, -0.502222786053, -0.036433449287],
 [ 0,               0,               0,               1             ]]
```

### 5.3 左腕相机与模型 TCP

当前 `skill_factory` 模型定义：

```text
T_GL_from_TL = translation([0.145, 0, 0])
```

这里 `0.145 = tool_offset 0.110 + site offset 0.035`，TCP 与 `gripper_base` 姿态相同。

因此：

```text
T_TL_from_CL = inverse(T_GL_from_TL) · T_GL_from_CL

T_TL_from_CL =
[[-0.036274539936, -0.502952849196,  0.863552308339, -0.060203072562],
 [-0.998958456762, -0.005684868872, -0.045273435150,  0.035831938409],
 [ 0.027679584839, -0.864295154303, -0.502222786053,  0.070030470897],
 [ 0,               0,               0,               1             ]]
```

反方向：

```text
T_CL_from_TL =
[[-0.036274539936, -0.998958456762,  0.027679584839, 0.031672364776],
 [-0.502952849196, -0.005684868872, -0.864295154303, 0.030451389646],
 [ 0.863552308339, -0.045273435150, -0.502222786053, 0.088781635422],
 [ 0,               0,               0,              1             ]]
```

左腕外参来源与质量：

```text
/home/dodo/skill_factory/config/calibration/left_wrist_handeye.yaml
相机序列号：260422271945
求解方法：HORAUD
求解样本：15
求解平移残差 RMS：0.009599782706 m
求解旋转残差 RMS：0.717981373 deg
独立验证样本：5
独立验证平移一致性 RMS：0.007230149437 m
独立验证旋转一致性 RMS：0.942532607 deg
accepted：true
```

## 6. 右腕相机与右臂末端

左右机械臂模型的末端局部原点和轴定义一致。根据设备负责人确认，两台腕相机相对各自
末端为同向、同孔位安装。因此在各自局部坐标系中采用：

```text
T_GR_from_CR := T_GL_from_CL
T_TR_from_CR := T_TL_from_CL
```

这是同构复制，不是右腕实测手眼标定。

### 6.1 右腕相机到右 `gripper_base`：`T_GR_from_CR`

```text
T_GR_from_CR =
[[-0.036274539936, -0.502952849196,  0.863552308339, 0.084796927438],
 [-0.998958456762, -0.005684868872, -0.045273435150, 0.035831938409],
 [ 0.027679584839, -0.864295154303, -0.502222786053, 0.070030470897],
 [ 0,               0,               0,              1             ]]
```

### 6.2 右 `gripper_base` 到右腕相机：`T_CR_from_GR`

```text
T_CR_from_GR =
[[-0.036274539936, -0.998958456762,  0.027679584839,  0.036932173066],
 [-0.502952849196, -0.005684868872, -0.864295154303,  0.103379552779],
 [ 0.863552308339, -0.045273435150, -0.502222786053, -0.036433449287],
 [ 0,               0,               0,               1             ]]
```

### 6.3 右腕相机与模型 TCP

```text
T_TR_from_CR =
[[-0.036274539936, -0.502952849196,  0.863552308339, -0.060203072562],
 [-0.998958456762, -0.005684868872, -0.045273435150,  0.035831938409],
 [ 0.027679584839, -0.864295154303, -0.502222786053,  0.070030470897],
 [ 0,               0,               0,               1             ]]

T_CR_from_TR =
[[-0.036274539936, -0.998958456762,  0.027679584839, 0.031672364776],
 [-0.502952849196, -0.005684868872, -0.864295154303, 0.030451389646],
 [ 0.863552308339, -0.045273435150, -0.502222786053, 0.088781635422],
 [ 0,               0,               0,              1             ]]
```

状态：`derived_from_left / unvalidated_on_right`。

如果后续确认右相机是左右镜像安装而非同向复制，本节矩阵无效；镜像候选应按实际安装轴定义
重新推导并用实物验证，不能仅凭相机型号判断。

## 7. 随关节姿态变化的完整关系

主相机是固定相机，腕相机是 eye-in-hand。腕相机相对机器人基座、主相机以及另一腕相机的
变换都随左右臂关节角变化，不存在一组永久固定的数值矩阵。

以下动态链统一以 `BA = acone_base_link` 为世界/整机基座。设 AC one 全身模型正运动学输出：

```text
T_BA_from_GL(qL)：左 gripper_base 到整机基座
T_BA_from_GR(qR)：右 gripper_base 到整机基座
```

则腕相机到整机基座：

```text
T_BA_from_CL(qL) = T_BA_from_GL(qL) · T_GL_from_CL
T_BA_from_CR(qR) = T_BA_from_GR(qR) · T_GR_from_CR
```

腕相机到固定主相机：

```text
T_C0_from_CL(qL) = T_C0_from_BA · T_BA_from_GL(qL) · T_GL_from_CL
T_C0_from_CR(qR) = T_C0_from_BA · T_BA_from_GR(qR) · T_GR_from_CR
```

右腕相机到左腕相机：

```text
T_CL_from_CR(qL, qR)
  = inverse(T_BA_from_CL(qL)) · T_BA_from_CR(qR)
```

主相机到左右 TCP：

```text
T_C0_from_TL(qL)
  = T_C0_from_BA · T_BA_from_GL(qL) · T_GL_from_TL

T_C0_from_TR(qR)
  = T_C0_from_BA · T_BA_from_GR(qR) · T_GR_from_TR
```

其中：

```text
T_GL_from_TL = T_GR_from_TR = translation([0.145, 0, 0])
```

## 8. TCP 定义边界

本文的 `TL/TR` 是 `/home/dodo/skill_factory` MuJoCo 模型中的 `tcp` site。其位置相对
`gripper_base` 为 `[0.145, 0, 0] m`。

ARX 左臂控制器发布的 `RobotStatus.end_pos = [x, y, z, roll, pitch, yaw]` 是在左臂局部根
`BL` 下表达的，并被主相机标定流程称为 TCP 位姿；这正是 4.1 原始外参属于 `BL` 而非 `BA`
的原因。厂商运动学共享库没有在可读配置中暴露 flange/gripper 到该控制器 TCP 的固定矩阵。
在把本文 TCP 矩阵与 `RobotStatus.end_pos` 混用前，必须确认控制器 TCP 与模型 TCP 是同一个
物理点、同一轴定义；否则应优先使用 `gripper_base` 手眼矩阵，或补充控制器 TCP 对齐标定。

## 9. 交付结论

1. 固定主相机 `C0 ↔ BL`：原始矩阵为实测并已验收；原文件中的 `B` 应明确解释为 `BL`。
2. 固定主相机 `C0 ↔ BA`：通过已验收的 `T_BA_from_BL` 迁移得到；全身模型应使用 4.3 的矩阵。
3. 左腕 `CL ↔ GL`：已有实测手眼外参，并通过独立姿态验证。
4. 右腕 `CR ↔ GR`：根据左右模型及安装同构关系复制左腕矩阵；当前为推导值，未独立验证。
5. 腕相机与整机基座/主相机/另一腕相机的关系：必须结合当前关节角和全身模型正运动学实时计算。
6. 右腕相机独立内参：当前交付中缺失，不能从其他 D405 数值直接复制。
7. 模型 TCP 与厂商控制器 TCP：尚需确认物理点和轴定义完全一致。
8. 当前运行时主相机 JSON 仍是 `T_BL_from_C0`；在另行修订运行链路前，不得当作 `T_BA_from_C0` 使用。

## 10. 原始来源

```text
# 主相机内参及运行时外参
/home/dodo/chenfu/inference/assets/ac_one/front_d405_rgbd_calibration.json

# 主相机完整双向外参、质量指标和 provenance
/home/dodo/chenfu/config/arx_x5_front_main_camera_extrinsic.json

# 主相机标定说明
/home/dodo/chenfu/docs/main-camera-extrinsic-calibration.md

# 主相机标定程序：订阅 /arm_master_l_status 并读取 RobotStatus.end_pos
/home/dodo/chenfu/data_collect/calibrate_main_camera_extrinsics.py

# 左臂控制器配置及其单臂 URDF
/home/dodo/chenfu/ARX_X5/ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/config/v2_collect.yaml
/home/dodo/chenfu/ARX_X5/ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/x5_2025.urdf

# AC one 全身模型与 BL -> BA 正式 frame contract
/home/dodo/skill_factory/generated/vendor/acone/urdf/acone.urdf
/home/dodo/skill_factory/src/skill_factory/core/frames.py

# 单臂模型迁移到 AC one 全身模型的验证证据
/home/dodo/skill_factory/skills/evidence/stage-05/acone_model_migration.json

# 左腕实测手眼外参
/home/dodo/skill_factory/config/calibration/left_wrist_handeye.yaml

# 左腕独立验证
/home/dodo/skill_factory/artifacts/hardware_calibration/validation/validation_result.json

# 左腕内参记录
/home/dodo/skill_factory/artifacts/hardware_calibration/dataset/sample_*/sample.json

# 模型 TCP 定义
/home/dodo/skill_factory/config/arx_x5.yaml
/home/dodo/skill_factory/src/skill_factory/sim/scene.py

# 三相机名称及序列号
/home/dodo/chenfu/data_collect/compare_live_to_raw_frame.py
/home/dodo/chenfu/dodo-threetask-mixed-runtime-v2/scripts/downstream/dodo/live_pi05_dodo_h64_joint.py
```
