# data_filter 实现契约（spec）

> 本文件是**实现层契约**：检查判据、阈值、IO、决策策略。设计 rationale 与取舍见 `../PLAN.md`；术语见 `CONTEXT.md`。
> 阈值均为 **provisional**（默认值见 `configs/*.yaml`），待真实分布与 `review` 队列校准。

## 1. 目标与范围

- **目标**：筛选高质量数据。输出应能回答：哪些 episode 值得训练、哪些降权、哪些剔除、剔除原因、各源质量分布。
- **范围内**：raw + processed 两层的质量检查、打分、筛选；产出 mask/report/score/split-list。
- **非目标**：跨源对齐、坐标系/朝向/夹爪/schema 转换、重写训练数据（属 align/convert 阶段）。

## 2. 数据契约

字段/索引以 `data_filter/io/schema.py` 为单一真相源。

### 2.1 raw pika / UMI
- `observations/pose_left|right` (T,6) 欧拉 XYZ，**station frame（漂移）**；`observations/gripper_left|right` (T,) 距离(m)；3 路 JPEG；`timestamps` (T,)。
- **无独立 action、无 joint、无 language** → S2 不适用。

### 2.2 raw teleop / NAS
- `action` (T,14)、`observations/qpos|qvel|effort` (T,14)、`eef_6d` (T,20)、`eef_quaternion` (T,16)、`language_instruction`、3 路 JPEG、`eef_left|right_time`。
- 有 state+action → S2 适用。

### 2.3 processed XVLA
- `observations/qpos` (T,20) = `[left_pose9, left_grip(9), right_pose9, right_grip(19)]`，`pose9=[xyz(3), rot6d(6)]`，`rot6d=concat(R[:,0],R[:,1])`，gripper binary/positive=closed。
- attrs 契约（按来源）见 `schema.PROCESSED_ATTRS`：
  - pika_umi：`pose_frame=robot_base_tip2base_piper_tcp_config`、`tip2base_applied=True`、`relative_to_first_frame=False`、`domain_name∈{...}`、`time_alignment_status=verified_common_time_axis`。
  - nas_teleop：`domain_name=nas_real_teleop`、`source_key=observations/eef_6d`、`time_alignment_status=verified_common_time_axis`。

## 3. 四阶段

| 阶段 | 输入 | 目的 | 输出 |
|---|---|---|---|
| Raw quality gate | raw pika/teleop | 挡明显坏采集 | raw report + episode flags + exclude-list |
| Processed validity gate | processed XVLA | 验证转换契约 | validity report + hard-fail 列表 |
| Quality scoring | raw+processed metrics | 生成质量分数 | episode_scores.jsonl + source 分布 |
| Dataset selection | score + mask (+人工复核) | 出可训练列表 | keep/downweight/drop lists + sampling_weights |

> 四阶段 = 两道闸门 + 两个横切阶段（scoring/selection 复用 metrics，不是新闸门）。

## 4. 检查清单

统一契约：每个 check 是纯函数 → `CheckResult`（`checks/base.py`）。hard-validity 违反时 `severity="hard_fail"`；quality 类 `passed=True`，靠 `frame_mask`/`metrics`/`flags` 表达。

### 4.1 Raw gate
| 检查 | pika | 遥操 | 类型 | 判据（provisional） |
|---|:--:|:--:|---|---|
| spike (S1) | ✅ | ✅ | quality | 残差/加速度/jerk 三者联合超 k·σ |
| tracking (S1′) | ✅ | — | hard(NaN)+quality | NaN；单帧位移>teleport_m；连续 frozen_min_frames 不变 |
| state_action (S2) | — | ✅ | quality | 互相关时滞 + 方向一致性 DA<da_threshold **（v1 后置，见 §8）** |
| extreme (S3) | ✅(速度) | ✅(绝对) | quality | 超 [q01−α·IQR, q99+α·IQR]；夹爪豁免 |
| timestamp | ✅ | ✅ | hard(非单调)+quality | 单调；dt≤max_dt_ratio·median；时钟偏差 |
| modality | ✅ | ✅ | hard | 各模态 T 相等 |
| video_quality (C3) | ✅ | ✅ | quality | 黑帧/模糊/解码失败；长静止使用连续窗口采样 |

### 4.2 Processed gate
| 检查 | 类型 | 判据（provisional） |
|---|---|---|
| schema_shape | hard | qpos (T,20)；图像与 qpos 长度一致 |
| finite | hard | qpos/action/ts 无 NaN/Inf |
| rot6d | hard | finite、‖a‖≈‖b‖≈1、a·b≈0（tol~1e-3，仅 processed pose9；det=‖a×b‖²≥0 无法判手性，不作判据） |
| gripper | hard | 二值 ⊆{0,1}；positive=closed；无开合翻转 |
| attrs | hard | 按来源核对 `PROCESSED_ATTRS` |
| manifest | hard | 文件数/路径/dataset_path 权重与磁盘一致 |
| decode_contract | hard | raw→processed decode 约定一致（不做纯像素 BGR 判） |
| lazy_reader_smoke | hard(可选) | 经可插拔 adapter 抽 1–2 batch 能被训练消费 |
| motion | quality | 速度/jerk/长静止/过快 → score |
| video_quality/coverage/multicam | quality | 模糊·黑帧比例 / 操作·夹爪覆盖 / 多相机可用 |

v1 **不做**：C1 指令一致(VLM)、C2 video-state IoU(SAM3)、S4 FK(URDF)（重依赖）。

## 5. 决策与输出

- **hard-validity 失败** → `drop`（processed 为 block），不进打分；check 运行时异常进入 `review`，不混入 drop list。
- **quality** → 透明规则分级（第一版）：任一 hard fail → `drop`；quality flag ≥ `review_when_quality_flags_ge` → `review`；否则 `keep_high_quality`；轻微异常 → `keep_with_downweight`。**暂不塌成加权标量**。
- **输出**（不重写 HDF5）：
  - raw：`raw_quality_report.json/md`、episode flags、`raw_exclude_list`。
  - processed：`processed_validity_report.json/md`、`episode_scores.jsonl`、`keep/downweight/drop` lists、`sampling_weights.json`。

## 6. 配置

按源分离：`configs/raw_pika.yaml`、`raw_teleop.yaml`、`processed_xvla.yaml`。运行时建议用 `--root` 显式指定数据根；默认配置只保留检查开关和已校准阈值。

Raw teleop schema 里记录的是已知可用字段全集；V2 gate 的硬依赖是 `action`、`observations/qpos`、左右臂 `eef_*_time` 和三路相机。`qvel/effort/language` 等字段当前不作为 hard contract。

## 7. 模块接口

`CheckResult(name, passed, severity, frame_mask, metrics, flags)`（`checks/base.py`）。loader 统一产出 `EpisodeSignals`（`io/loaders.py`）。

## 8. 里程碑

**v1 范围**：processed validity（schema/finite/rot6d/gripper/attrs/manifest）+ raw（timestamp/modality/spike/tracking/extreme/video quality）+ scoring/report。
**S2（遥操 state-action 趋势对齐）后置为升级项，不阻塞 v1**（与 PLAN 一致）。

1. 脚手架 + spec + CONTEXT + io + selftest ← **当前（骨架已建）**
2. **Processed 最小闭环（P0）**：`load_processed_xvla` → schema/finite/rot6d/gripper/attrs/modality → `score_episode` → `write_report` + 5–8 合成单测 —— /tdd
3. Raw 信号类（spike/tracking/extreme/timestamp/modality）—— /tdd
4. 视频类 + decode 约定一致性 + 速度诊断
5. scoring / pipeline / report 完善 + 真数据跑通调参
6. **（升级，v1 后）** 遥操 S2、manifest/lazy-reader、VLM/SAM/FK 等重依赖检查

## 9. 待校准 / 开放

- 全部阈值 provisional；无人工校准集，靠 `review` 队列后量 precision/recall 校准。
- `data_roots`：teleop/NAS 根、processed XVLA 根（一个或多个目录）待填。
- rot6d 检查须在**真 univis-processed 文件**上验证在好数据上 pass（防假阳）。
