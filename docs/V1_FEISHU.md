# data_filter V2 简版说明

## 1. 目标

`data_filter` 用来筛查 pika/UMI 和遥操数据质量，把 episode 分成：

| label | 含义 | 用法 |
|---|---|---|
| `keep_high_quality` | 高质量 | 正常训练 |
| `keep_with_downweight` | 有轻微问题 | 可训练，建议降权 |
| `review` | 可疑 | 人工复核 |
| `drop` | 硬性失败 | 默认不进训练 |

工具只输出报告、分桶列表和采样权重，不修改 HDF5，不做数据转换。

## 2. 为什么分 raw 和 processed

| gate | 检查对象 | 关注点 |
|---|---|---|
| raw gate | 原始 pika/UMI、teleop/NAS HDF5 | 采集是否可靠 |
| processed gate | 已转换成 XVLA 训练格式的 HDF5 | 转换产物能否训练 |

简单说：

- raw gate 管“采得对不对”：时间戳、相机、tracking、左右臂信号、state-action 对齐。
- processed gate 管“转完能不能训练”：`(T,20)` qpos、rot6d、gripper、attrs、模态长度对齐。

## 3. V2 主要检查

### Raw gate

| 模块 | 主要 flag | 说明 |
|---|---|---|
| 基础可用性 | `load(...)`、`finite(...)`、`min_length(too_short)` | 文件打不开、非数值、NaN/Inf、空 episode |
| 模态对齐 | `modality(length_mismatch)`、`image_schema(missing:...)` | 状态、动作、图像长度或 key 不一致 |
| 时间轴 | `timestamp(non_monotonic)`、`timestamp(dt_jump)`、`timestamp_skew(clock_skew)` | 时间戳倒退、跳变、左右臂不同步 |
| PIKA tracking | `tracking(teleport)`、`tracking(frozen)` | 位姿瞬移或冻结 |
| 轨迹突变 | `spike(spike)` | 速度、加速度、jerk 异常 |
| 单臂信号 | `arm_activity(left/right_arm_frozen)` | 左/右臂整段缺失或冻结 |
| 视频质量 | `cam_*_black/blur/static/decode_failed/shape_mismatch` | 黑帧、模糊、静止、解码失败、分辨率不一致 |
| S2 state-action | `low_directional_agreement`、`large_lag`、`low_correlation` | teleop action 和 state change 趋势不一致 |

### Processed gate

| 模块 | 主要 flag | 说明 |
|---|---|---|
| 训练格式 | `schema_shape(bad_shape/too_short)` | qpos 不是 `(T,20)` 或空 episode |
| 数值合法 | `finite(nonfinite/nonnumeric:*)` | qpos/timestamp 非数值或 NaN/Inf |
| 模态对齐 | `modality(length_mismatch)`、`image_schema(missing:...)` | 图像、qpos、timestamp 长度或 key 不一致 |
| 时间轴 | `timestamp(non_monotonic/dt_jump)` | 时间戳倒退或跳变 |
| rot6d | `norm_a`、`norm_b`、`orthogonality` | 旋转 6D 非单位或不正交 |
| gripper | `out_of_range`、`not_binary` | 夹爪值域错误或未二值化 |
| attrs | `missing:*`、`wrong:*` | 缺坐标系、domain、time alignment 等声明 |
| motion | `speed_outlier`、`jerk_outlier`、`long_static`、`low_gripper_coverage` | 轨迹过快、jerk 异常、长静止、夹爪覆盖不足 |
| 视频质量 | `cam_*_black/blur/static/decode_failed/shape_mismatch` | 同 raw gate |

## 4. V2 新增点

| 功能 | 当前实现 |
|---|---|
| video quality | 每路相机抽 4 帧做黑帧/模糊/解码检查；额外抽 1 个 45 帧连续窗口做静止检查 |
| S2 state-action | teleop 中用 `action - qpos` 近似命令方向，和 `qpos` 一阶差分做 lag 对齐、DA、corr 检查 |
| HDF5 图像布局 | 支持 vlen dataset、group + sibling index、group 内 index |
| 报告可读性 | Markdown 输出 Top Reasons、Top Check Flags、每条 episode 的关键 metric/threshold |
| 安全性 | check 异常进 `review`，不静默进入 `drop`；CLI 会校验 gate/source/config |

关键阈值：

| 配置 | 当前值 |
|---|---:|
| `video_quality.sample_frames` | 4 |
| `video_quality.static_window_frames` | 45 |
| `video_quality.static_sample_windows` | 1 |
| `state_action.da_threshold` | 0.30 |
| `state_action.corr_threshold` | 0.50 |
| `state_action.max_lag_frames` | 15 |

## 5. 输出文件

每组数据会输出 9 个文件：

| 文件 | 用途 |
|---|---|
| `*_report.md` | 人看的报告 |
| `*_report.json` | 完整机器可读报告 |
| `*_episode_scores.jsonl` | 每条 episode 的 label/reason |
| `*_sampling_weights.json` | 训练采样权重 |
| `*_keep_high_quality_list.txt` | 高质量列表 |
| `*_downweight_list.txt` | 降权列表 |
| `*_review_list.txt` | 复核列表 |
| `*_drop_list.txt` | 剔除列表 |
| `run.log` | 本次运行日志 |

raw 输出前缀是 `raw_quality`，processed 输出前缀是 `processed_validity`。

## 6. 最终验证结果

测试：

```text
58 passed
```

服务器最终报告：

```text
/data01/cc/data/xvla_market_bottle/processing/data_filter_v2_final_20260709_100059
```

真实数据结果：

| 数据 | total | keep_high_quality | downweight | review |
|---|---:|---:|---:|---:|
| `raw_pika_extra` | 376 | 290 | 78 | 8 |
| `raw_umi_scanqr_synced` | 171 | 111 | 50 | 10 |
| `raw_market_bottle_tele` | 884 | 520 | 280 | 84 |
| `raw_nas_teleop_full_raw` | 406 | 284 | 60 | 62 |
| `processed_all` | 953 | 428 | 369 | 156 |

## 7. 后续版本

| 功能 | 说明 |
|---|---|
| C2 video-state consistency | V3 做：URDF + 相机参数重投影，再和真实视频 mask 比 IoU |
| rot6d 语义错误 | 当前只能查范数和正交，不能判断行/列语义错误 |
| VLM/SAM/URDF 检查 | 依赖较重，放到 V3 |
