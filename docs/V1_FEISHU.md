# data_filter V1/V2 说明

## 1. 这是什么

`data_filter` 是一个数据质量筛查工具，用来检查 pika/UMI 和遥操数据里哪些 episode 质量好、哪些需要降权、哪些需要人工复核、哪些应该剔除。

它只做检查和筛选，不做数据转换：

- 不做坐标系转换。
- 不重写 HDF5。
- 不修改训练数据。
- 只输出报告、原因、分桶列表和采样权重。

## 2. 第一版处理哪两类数据

| 数据层级 | 输入 | 目的 |
|---|---|---|
| raw 数据 | 原始 pika/UMI、原始 teleop/NAS HDF5 | 在转换前发现明显坏采集 |
| processed 数据 | 已转换成 XVLA 训练格式的 HDF5 | 在训练前检查转换产物是否符合训练契约 |

整体流程：

```text
HDF5 数据
  -> 读取 episode
  -> 跑质量检查
  -> 给出 label 和原因
  -> 输出 report / list / sampling weights
```

## 3. 当前版本已经能查什么

### Raw 数据

报告里的 `check(flag)` 可以这样理解：

命名说明：raw teleop 里，`timestamp(...)` 对应 `observations/eef_left_time`，`eef_right_time(...)` 对应 `observations/eef_right_time`。所以左右臂时间轴都会查，只是左臂在当前报告里叫 `timestamp`。

| 报错类型 | 怎么检测 | 含义 |
|---|---|---|
| `load(...)` | HDF5 打不开，或必需字段读取失败 | 文件本身不可用，直接 `drop` |
| `finite(nonfinite:*)` | 对 pose/qpos/action/timestamp 检查 NaN/Inf | 数值里有非法值，直接 `drop` |
| `finite(nonnumeric:*)` | 字段 dtype 不是数值类型 | 数据格式不对，直接 `drop` |
| `min_length(too_short)` | episode 帧数 `< 1` | 空 episode，直接 `drop` |
| `modality(length_mismatch)` | 比较 qpos/action/pose/timestamp/image 的长度 | 多个模态帧数不同，说明同步或写入有问题 |
| `image_schema(missing:...)` | 检查三路相机 key 是否存在 | 缺 `cam_high`、`cam_left_wrist` 或 `cam_right_wrist` |
| `timestamp(missing)` | raw teleop 检查 `eef_left_time`；raw pika 检查 `timestamps` | 左臂/主时间轴缺失，直接 `drop` |
| `timestamp(non_monotonic)` | raw teleop 对 `eef_left_time` 检查 `time[i+1] - time[i] > 0`；raw pika 对 `timestamps` 检查 | 左臂/主时间轴倒退，直接 `drop` |
| `timestamp(dt_jump)` | raw teleop 对 `eef_left_time` 计算 `dt`；raw pika 对 `timestamps` 计算；若 `max(dt) > 3 * median(dt)` | 左臂/主时间轴可能丢帧、暂停、采集卡顿 |
| `eef_right_time(missing)` | raw teleop 检查右臂时间轴是否存在 | 右臂时间戳缺失，直接 `drop` |
| `eef_right_time(non_monotonic)` | raw teleop 对 `eef_right_time` 检查 `time[i+1] - time[i] > 0` | 右臂时间轴倒退，直接 `drop` |
| `eef_right_time(dt_jump)` | raw teleop 对 `eef_right_time` 计算 `dt`，若 `max(dt) > 3 * median(dt)` | 右臂时间轴有采集卡顿或跳变 |
| `timestamp_skew(clock_skew)` | 比较 `eef_left_time` 和 `eef_right_time`，默认最大差值 `> 0.05s` | 左右臂时间轴不同步 |
| `spike(spike)` | 对轨迹计算速度、加速度、jerk；超过鲁棒阈值的帧数达到阈值 | 轨迹有突变或不连续 |
| `tracking(teleport)` | pika 位姿单帧位移 `> teleport_m` | pika tracking 瞬移 |
| `tracking(frozen)` | pika 位姿连续多帧几乎不动 | pika tracking 可能冻结 |
| `arm_activity(right_arm_frozen)` | 右臂半区 unique row 太少或均值标准差太小 | 右臂整段信号缺失/冻结，进入 `review` |
| `arm_activity(left_arm_frozen)` | 左臂半区 unique row 太少或均值标准差太小 | 左臂整段信号缺失/冻结，进入 `review` |
| `video_quality(cam_*_black)` | 抽样解码相机图像，灰度均值低于 `black_luma` 的比例超过阈值 | 对应相机有明显黑帧 |
| `video_quality(cam_*_blur)` | 抽样图像计算 Laplacian 方差，低于 `blur_var` 的比例超过阈值 | 对应相机画面模糊 |
| `video_quality(cam_*_static)` | 计算相邻抽样帧平均差分，连续低于 `static_diff_eps` 的帧数超过阈值 | 对应相机画面长时间不变或卡住 |
| `video_quality(cam_*_decode_failed)` | 抽样 JPEG 解码失败比例超过阈值 | 图像损坏或编码异常 |
| `state_action(low_directional_agreement)` | raw teleop 对 `qpos` 一阶差分与 `action` 做 lag 对齐后，方向一致性低于阈值 | action 和 state change 趋势不一致 |
| `state_action(large_lag)` | 在 `[-max_lag_frames, max_lag_frames]` 内找最佳相关 lag，最佳 lag 贴近边界 | action/state 可能存在明显时序错位 |
| `state_action(low_correlation)` | 可选阈值：最佳相关系数低于配置阈值 | action/state 线性趋势相关性低 |

已覆盖的真实问题：

- `episode_2030` 这类右臂信号冻结，会输出 `arm_activity(right_arm_frozen)`，被标为 `review`。
- raw teleop 缺 `eef_right_time`，会输出 `eef_right_time(missing)`，被标为 `drop`。
- 左右臂时间轴不同步，会输出 `timestamp_skew(clock_skew)`。

### Processed 数据

报告里的 `check(flag)` 可以这样理解：

| 报错类型 | 怎么检测 | 含义 |
|---|---|---|
| `schema_shape(bad_shape)` | 检查 `observations/qpos` 是否是 `(T, 20)` | qpos 不是训练约定格式，直接 `drop` |
| `schema_shape(too_short)` | 检查 `T >= 1` | 空 episode，直接 `drop` |
| `finite(nonfinite:*)` | 检查 qpos/timestamp 是否有 NaN/Inf | 数值非法，直接 `drop` |
| `finite(nonnumeric:*)` | 字段 dtype 不是数值类型 | 数据格式不对，直接 `drop` |
| `modality(length_mismatch)` | 比较 qpos、timestamp、三路图像帧数 | 图像和状态没有对齐 |
| `image_schema(missing:...)` | 检查三路相机 key 是否存在 | 缺相机模态 |
| `timestamp(missing)` | 检查 processed 是否有时间戳 | 时间轴缺失，直接 `drop` |
| `timestamp(non_monotonic)` | 检查 `timestamp[i+1] - timestamp[i] > 0` | 时间戳倒退，直接 `drop` |
| `timestamp(dt_jump)` | 计算 `dt`，若 `max(dt) > 3 * median(dt)` | processed 时间轴有卡顿或跳变 |
| `rot6d_left(norm_a)` / `rot6d_right(norm_a)` | 检查 rot6d 第一列向量范数是否接近 1 | 旋转表示不合法 |
| `rot6d_left(norm_b)` / `rot6d_right(norm_b)` | 检查 rot6d 第二列向量范数是否接近 1 | 旋转表示不合法 |
| `rot6d_left(orthogonality)` / `rot6d_right(orthogonality)` | 检查两列向量点积是否接近 0 | 旋转两列不正交 |
| `gripper(out_of_range)` | 检查 gripper 是否在 `[0,1]` 附近 | 夹爪值域不合法 |
| `gripper(not_binary)` | 检查 gripper 是否接近 0/1 二值 | 夹爪没有被正确二值化 |
| `attrs(missing:*)` | 检查 domain、pose_frame、tip2base、time_alignment 等 attrs | 转换产物缺少来源/坐标系/时间对齐声明 |
| `attrs(wrong:*)` | attrs 存在但值不符合约定 | 转换产物声明和预期不一致 |
| `motion(speed_outlier)` | 对左右末端 xyz 计算速度，超过鲁棒阈值的帧数达到阈值 | 轨迹速度异常，可能不连续或过快 |
| `motion(jerk_outlier)` | 由速度差分得到加速度，再差分得到 jerk；异常帧数达到阈值 | 加速度/加加速度不连续 |
| `motion(long_static)` | 连续低速度帧数超过 `static_min_frames` | episode 有长时间静止段 |
| `motion(low_gripper_coverage)` | 统计左右夹爪变化次数，低于阈值 | 夹爪动作覆盖不足 |
| `video_quality(cam_*_black)` | 抽样解码相机图像，灰度均值低于 `black_luma` 的比例超过阈值 | 对应相机有明显黑帧 |
| `video_quality(cam_*_blur)` | 抽样图像计算 Laplacian 方差，低于 `blur_var` 的比例超过阈值 | 对应相机画面模糊 |
| `video_quality(cam_*_static)` | 计算相邻抽样帧平均差分，连续低于 `static_diff_eps` 的帧数超过阈值 | 对应相机画面长时间不变或卡住 |
| `video_quality(cam_*_decode_failed)` | 抽样 JPEG 解码失败比例超过阈值 | 图像损坏或编码异常 |

## 4. 输出标签是什么意思

每个 episode 会被分到四类之一：

| label | 含义 | 建议 |
|---|---|---|
| `keep_high_quality` | 没发现明显问题 | 正常进入训练 |
| `keep_with_downweight` | 有轻微质量问题 | 可以训练，但建议降低采样权重 |
| `review` | 有可疑问题 | 人工看一下再决定 |
| `drop` | 硬性检查失败 | 默认不进训练 |

当前规则比较简单：

- hard fail 直接 `drop`。
- 单臂冻结直接 `review`。
- 一个普通 quality 问题通常 `keep_with_downweight`。
- 多个 quality 问题通常 `review`。

## 5. 输出哪些文件

运行后会输出：

| 文件 | 用途 |
|---|---|
| `{prefix}_report.json` | 完整机器可读报告 |
| `{prefix}_report.md` | 人可读报告 |
| `{prefix}_drop_list.txt` | 建议剔除的数据 |
| `{prefix}_review_list.txt` | 建议人工复核的数据 |
| `{prefix}_downweight_list.txt` | 建议降权的数据 |
| `{prefix}_keep_high_quality_list.txt` | 高质量数据 |
| `{prefix}_episode_scores.jsonl` | 每条 episode 的标签和原因 |
| `{prefix}_sampling_weights.json` | 训练采样权重 |

raw 和 processed 的输出文件名都有 prefix，不会互相覆盖：

| gate | prefix |
|---|---|
| raw | `raw_quality` |
| processed | `processed_validity` |

## 6. 怎么运行

raw pika：

```bash
uv run python scripts/run_filter.py \
  --gate raw \
  --source pika \
  --root /path/to/raw_pika \
  --out /path/to/output/raw_pika
```

raw teleop：

```bash
uv run python scripts/run_filter.py \
  --gate raw \
  --source teleop \
  --root /path/to/raw_teleop \
  --out /path/to/output/raw_teleop
```

processed：

```bash
uv run python scripts/run_filter.py \
  --gate processed \
  --root /path/to/processed_xvla \
  --out /path/to/output/processed
```

如果不手动传 `--config`，会自动选择对应配置：

- raw pika -> `raw_pika.yaml`
- raw teleop -> `raw_teleop.yaml`
- processed -> `processed_xvla.yaml`

## 7. 当前状态

当前版本已经完成并通过测试：

```text
49 passed
```

对应代码提交：

- `d7ccca3 Fix first-batch data filter review issues`
- `66e08e3 Tighten first-batch data filter regressions`

## 8. 还没做什么

这些放到后续版本：

| 功能 | 说明 |
|---|---|
| C2 video-state consistency | 用重投影和分割 mask 检查视频与机器人状态是否一致 |
| rot6d 行/列语义错误 | 当前只能检查正交和范数，不能判断所有语义错误 |
| VLM/SAM/URDF 检查 | 依赖较重，后续再接 |

## 9. 当前结论

第一版可以作为轻量数据质量闸门使用：

- raw 层能挡掉明显坏采集。
- processed 层能检查训练数据基本契约。
- 输出结果可追溯，能直接给训练筛选列表和人工复核列表。
- 配置和报告文件已经修正为可信，不会 raw/processed 混用或互相覆盖。
