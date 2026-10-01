当前 `PickUpTestTube` 的完整推理链路如下：

```text
启动程序
  ↓
读取双臂 RobotStatus 和三路相机
  ↓
左右臂移动到任务初始位姿
  ↓
保持初始位姿一小段时间
  ↓
关闭左臂推理阶段的 topic 发布
  ↓
循环：
  读取最新双臂状态和相机
    ↓
  构造模型输入
    ↓
  请求 Cosmos3-Edge 模型
    ↓
  得到 32×14 action chunk
    ↓
  取前 8 个动作
    ↓
  任务锁定、滤波、限幅、安全检查
    ↓
  插值成 60 Hz 控制命令
    ↓
  只发布右臂 ROS 命令
```

## 1. 启动和连接

入口是：

[`x5_cosmos3_edge_pick_tube.py:8`](/home/dodo/chenfu/inference/x5_cosmos3_edge_pick_tube.py:8)

默认参数：

- 模型服务：`tcp://127.0.0.1:5580`
- 模型输入图像：`320×240 RGB`
- 推理频率参数：15 Hz
- 低层发布频率：60 Hz
- 每次模型返回：32 个 14D 动作
- 每次实际执行：默认前 8 个动作

程序会等待以下两个状态 topic：

```text
/arm_slave_l_status
/arm_slave_r_status
```

状态拼接顺序为：

```text
[left arm 7D, right arm 7D]
```

对应代码：

[`runtime.py:65`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:65)

## 2. 初始化阶段

读取当前双臂状态后，构造任务初始状态：

```python
measured_state = 当前双臂状态
start_target = PickUpTestTube.start_state
```

对应：

[`runtime.py:270`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:270)

然后计算初始化所需的插值步数，默认至少持续 3 秒，并根据最大关节步长自动增加步数：

[`runtime.py:104`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:104)

初始化时：

- 左臂和右臂都会发布；
- 从当前姿态平滑移动到初始姿态；
- 发布频率为 60 Hz；
- 到达后默认保持 0.5 秒。

[`runtime.py:276`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:276)

此阶段发布：

```text
/arm_master_l_status
/arm_master_r_status
```

## 3. 进入推理阶段后关闭左臂发布

初始化完成后，PickUpTestTube 的配置为：

[`pickup_test_tube.py:16`](/home/dodo/chenfu/inference/cosmos3_edge/tasks/pickup_test_tube.py:16)

```python
publish_left_arm_during_inference=False
```

运行时切换发布开关：

[`runtime.py:302`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:302)

之后每次发送动作时：

- 左臂 topic 不发布；
- 右臂 topic 正常发布。

[`runtime.py:191`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:191)

注意：左臂物理上保持在初始化位姿，但不会再收到新的左臂控制命令。

## 4. 采集当前观测

每一轮循环会读取：

### 双臂状态

重新读取最新的左右臂 `RobotStatus`：

[`runtime.py:324`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:324)

### 三路相机

固定使用：

```text
front_rgb
left_rgb
right_rgb
```

如果原始图像不是 `320×240`，会通过 OpenCV 缩放到：

```text
320×240×3 uint8 RGB
```

[`runtime.py:340`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:340)

## 5. 构造模型 state

首先将实时双臂状态拼接成 14D：

```text
[left_joint_0 ... left_joint_6,
 right_joint_0 ... right_joint_6]
```

然后做夹爪值归一化，将夹爪编码映射到：

```text
[-2π, 0)
```

[`control.py:34`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:34)

对于 PickUpTestTube，左臂的 7D state 会全部清零：

```python
model_state[:7] = 0.0
```

[`control.py:44`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:44)

因此模型实际看到的 state 是：

```text
左臂： [0, 0, 0, 0, 0, 0, 0]
右臂：实时关节状态
```

这只是模型输入被置零，不会改变机械臂实际状态。

## 6. 请求 Cosmos3-Edge 模型

模型请求包含：

- 三路图像；
- 14D state；
- Instruction：

```text
Pick up test tube with the pink label.
```

请求 endpoint：

```text
get_action
```

模型输入图像形状会扩展为：

```text
(1, 1, 240, 320, 3)
```

state 形状为：

```text
(1, 14)
```

并指定：

```python
executed_action_steps = 32
```

对应：

[`protocol.py:114`](/home/dodo/chenfu/inference/cosmos3_edge/protocol.py:114)

模型返回：

```text
32×14
```

的绝对关节位置 action，而不是关节增量：

[`protocol.py:162`](/home/dodo/chenfu/inference/cosmos3_edge/protocol.py:162)

客户端会检查：

- shape 是否为 `(32, 14)`；
- 是否存在 NaN/Inf；
- 服务端是否为连续原始关节位置模型。

## 7. 只执行 action chunk 的前 8 步

虽然模型返回 32 步，但当前任务默认：

```python
execution_steps = 8
```

实际只执行：

```python
prediction.actions[:8]
```

[`runtime.py:350`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:350)

执行完这 8 步后，重新读取相机和 state，再向模型请求下一组 32 步动作。

## 8. 每个模型动作经过 ActionProcessor

每个 14D 模型动作依次经过以下操作。

### 8.1 任务锁定

PickUpTestTube 配置了：

```python
lock_left_arm=True
```

因此内部 action 的左臂部分会被替换为初始姿态：

[`control.py:49`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:49)

但因为推理阶段不发布左臂，这部分不会发送到左臂 topic。

右臂 action 保留模型输出。

### 8.2 低通滤波

默认参数：

```text
arm_filter_alpha = 0.35
gripper_filter_alpha = 0.35
```

新动作不会直接跳到模型目标，而是从上一次 command 平滑靠近当前目标：

[`control.py:78`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:78)

### 8.3 单步限幅

默认限制：

```text
普通关节：每步最多变化 0.035
夹爪：每步最多变化 0.08
```

[`control.py:91`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:91)

### 8.4 跳变安全检查

最终还会检查：

```text
普通关节最大跳变：0.5
夹爪最大跳变：1.0
```

超过限制就抛出异常，不发送危险命令：

[`control.py:106`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:106)

### 8.5 更新内部 previous action

处理后的动作会保存为下一步的基准：

```python
self.previous = result
```

[`control.py:120`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:120)

## 9. 从 15 Hz 动作插值到 60 Hz

每个策略动作会被拆成：

```text
60 / 15 = 4 个低层控制命令
```

普通关节使用五次 smoothstep 插值：

[`control.py:130`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:130)

夹爪则直接使用目标值，不进行同样的平滑插值：

[`control.py:145`](/home/dodo/chenfu/inference/cosmos3_edge/control.py:145)

因此一个模型动作的大致过程是：

```text
模型动作
  ↓
ActionProcessor 滤波和限幅
  ↓
拆成 4 个 60 Hz 命令
  ↓
每个命令发布到右臂 topic
```

## 10. 最终 ROS 发布

发送前会构造 `RobotStatus`：

```text
header.stamp：当前 ROS 时间
end_pos：6 个 0
joint_pos：7D 关节命令
joint_vel：7 个 0
joint_cur：7 个 0
```

[`runtime.py:182`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:182)

PickUpTestTube 推理阶段最终只发布：

```text
/arm_master_r_status
```

不会发布：

```text
/arm_master_l_status
```

另外有一个 30 Hz keepalive 线程。如果主循环暂时没有新动作，超过约 66 ms 后会重复发布上一条动作；PickUpTestTube 此时也只会重复发布右臂命令：

[`runtime.py:210`](/home/dodo/chenfu/inference/cosmos3_edge/runtime.py:210)

## 11. 一轮循环的核心数据变化

```text
实时 RobotStatus
    ↓
14D measured_state
    ↓
左臂前 7D 置零 + 夹爪归一化
    ↓
模型输入 state
    ↓
Cosmos3-Edge 输出 32×14 action
    ↓
取前 8×14
    ↓
任务锁定
    ↓
低通滤波
    ↓
单步限幅
    ↓
跳变检查
    ↓
4 个 60 Hz 插值命令
    ↓
只发送右臂 7D RobotStatus
```

当前代码没有额外的“抓取成功判断”或视觉完成判断。推理会持续运行，直到：

- 达到 `max_steps`；
- 使用 `--once` 执行完一组 action chunk；
- 按下 `Ctrl-C`；
- 相机、状态、模型或安全检查出错。