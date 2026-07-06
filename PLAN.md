# data_filter 方案设计

> 对多源机器人操作数据做 **data quality 检查与高质量数据筛选**，方法依据 Qwen-RobotManip 论文（arXiv:2606.17846）的数据 curation 管线（§2.4 五阶段 state-action 过滤 + 三项跨模态检查），并结合当前 xVLA / market_bottle 数据中真实踩过的问题（rot6d、夹爪语义、time/domain attrs、RGB/BGR、manifest/lazy HDF5）。
> 当前数据源：**pika/UMI**（手持采集设备）与 **遥操**（teleoperation/NAS）两类，均为 HDF5。检查对象分两层：raw HDF5 与 processed XVLA HDF5。
>
> 状态：方案草案，待确认四处后再建脚手架与实现。日期 2026-07-06。

---

## 0. 目标与定位

- **核心目标**：筛选高质量数据，而不只是保护某一次训练流程。`data_filter` 的输出应能回答：哪些 episode 值得进入训练、哪些只能降权、哪些应剔除、剔除原因是什么、不同数据源的质量分布如何。
- **做什么**：对 pika/UMI 与遥操两类数据做质量检查、质量打分与筛选，标记坏 episode / 坏帧，产出质量报告、mask、score 与 split/exclude-list。检查分两层：
  - **Raw quality gate**：对原始采集 HDF5 做轻量质量闸门，尽早发现坏 JPEG、长度不一致、timestamp 异常、NaN/Inf、PIKA tracking dropout、明显跳变。
  - **Processed XVLA quality gate**：对转换后的训练 HDF5 做质量分层与筛选，重点检查 qpos/action schema、rot6d 合法性、夹爪语义、domain/time attrs、坐标系 attrs、manifest 覆盖、RGB/BGR 一致性、运动质量与视觉质量。
- **不做什么（边界）**：不在 `data_filter` 内执行跨源对齐或重写训练数据。坐标系转换、夹爪语义统一、canonical schema 生成、速度匹配属于 align/convert 阶段；`data_filter` 只验证输入或产物是否满足预期，并输出 mask/report。
- **成功标准**：能对 raw 与 processed 两层数据跑出 per-source / per-dim / per-episode 的 flag 率、质量分数与筛选建议；坏样例（跳变、跟踪丢失、黑帧、时序错位、rot6d 错误、夹爪语义错误、domain/time attrs 缺失）能被稳定抓出；边界样例能进入人工复核队列；阈值和打分权重可按源配置。

### 0.1 一眼看懂：要做什么

> 四阶段 = **两道质量闸门**（Raw / Processed）+ **两个横切阶段**（Quality scoring / Dataset selection）。后两者跨闸门复用 metrics/flags，不是新增闸门。

| 阶段 | 输入 | 主要目的 | 关键检查 | 输出 | 决策 |
|---|---|---|---|---|---|
| **Raw quality gate** | 原始 pika/UMI HDF5、原始 teleop/NAS HDF5 | 先筛掉明显坏采集，避免垃圾数据进入后续转换 | HDF5 schema、长度一致、timestamp、NaN/Inf、JPEG decode、黑帧、PIKA tracking dropout、pose/action 突变、速度分布 | `raw_quality_report.json/md`、frame mask、episode flags、raw exclude-list | 明显坏数据 `drop`；轻微异常 `review`；正常进入 align/convert |
| **Align/convert sanity check** | 对齐/转换脚本输出的 processed XVLA HDF5 | 验证转换产物是否符合训练契约，同时发现转换引入的问题 | qpos shape、rot6d 合法性、gripper 二值与语义、pose_frame/tip2base attrs、domain/time attrs、manifest coverage、RGB/BGR audit | `processed_validity_report.json/md`、hard-fail 列表 | hard validity 失败直接 `drop/block`；通过后进入质量评分 |
| **Quality scoring** | raw + processed 的 metrics/flags | 给每个 episode 生成质量分数，用于筛选高质量数据 | jerk/速度异常、长静止、模糊/黑帧比例、操作阶段覆盖、夹爪动作覆盖、时间稳定性、多相机可用性 | `episode_scores.jsonl`、source-level 分布、阈值建议 | `keep_high_quality` / `keep_with_downweight` / `review` / `drop` |
| **Dataset selection** | score、mask、人工复核结果 | 生成最终可训练数据列表和采样权重 | 按源/任务/质量分桶，控制坏样本比例，保留多样性 | `train_keep_list.txt`、`downweight_list.txt`、`drop_list.txt`、`sampling_weights.json` | 高质量进入训练；边界样本降权或复核；坏样本剔除 |

第一版优先级：

| 优先级 | 模块 | 为什么先做 |
|---|---|---|
| P0 | processed validity + quality scoring 基础版 | 最贴近当前训练数据，能立刻筛出 rot6d、夹爪、time/domain、运动异常等问题 |
| P1 | raw schema/timestamp/JPEG/tracking checks | 能在转换前挡掉明显坏采集，减少后续处理浪费 |
| P2 | video quality + speed/jerk 分布调参 | 需要看真实分布后定阈值，先报告再逐步用于筛选 |
| P3 | S2 state-action trend alignment | 遥操数据可用，用 cross-correlation + DA 抓 state-action 时序错位；先作为升级项，不阻塞 v1 |
| P4 | VLM/SAM/URDF/任务语义检查 | 重依赖，成本高，等轻量规则稳定后再做 |

---

## 1. 数据现实（实测 schema —— 全盘设计的地基）

| | **pika**（`~/data/0703_lyd_01_synced/*.hdf5`，attr 自述 "UMI Pika raw, time-synced"） | **遥操**（`tele_*.hdf5`，ALOHA 风格） |
|---|---|---|
| 位姿 | `observations/pose_left\|right` (T,6) 欧拉 XYZ，**station frame（基站系，逐 session 漂移）** | `observations/eef_quaternion` (T,16) / `eef_6d` (T,20)，robot base |
| 关节 | ❌ 无 | `qpos` (T,14) / `qvel` (T,14) / `effort` (T,14) |
| **动作流** | ❌ **无独立 action**（手持设备，只有观测位姿） | ✅ `action` (T,14) + `base_action` (T,2) |
| 夹爪 | `gripper_left\|right` (T,)，单位=距离(m) | 嵌在 action/eef 内，单位/量程/开合约定不同 |
| 语言 | ❌ 无 | ✅ `language_instruction` (1,) |
| 相机 | 3 路 JPEG（`cam_high` / `cam_left_wrist` / `cam_right_wrist`） | 3 路（同名） |
| 时间戳 | `timestamps` (T,)，rebased 到首帧 | `eef_left_time` / `eef_right_time` (T,) |

**关键不对称**：raw pika 没有 action、没有 joint、没有 language，位姿在 station frame（基站系，逐 session 漂移）里；遥操 raw 通常已有 robot-base 下的 eef/qpos/action。→ 论文的检查**不能照抄**，要按 raw schema 与 processed schema 分层落地（见 §3）。

### 1.1 当前 processed XVLA 训练数据现实

当前 market_bottle 训练使用的 processed HDF5 已统一到 20D `observations/qpos`：

```text
left_pose9,left_gripper,right_pose9,right_gripper
pose9 = xyz + rot6d
rot6d = concat(R[:,0], R[:,1])
gripper = binary, positive=closed
```

PIKA/UMI processed 数据应满足：

```text
pose_frame = robot_base_tip2base_piper_tcp_config
tip2base_applied = True
relative_to_first_frame = False
domain_name in {pika_umi_tip2base_abs, pika_extra_tip2base_abs, pika_camera_wrong_tip2base_abs}
time_alignment_status = verified_common_time_axis
```

NAS/teleop processed 数据应满足：

```text
domain_name = nas_real_teleop
source_key = observations/eef_6d
rot6d = converted_from_row_major_R[:,:2]_to_concat_cols
time_alignment_status = verified_common_time_axis
```

这些 processed 检查不是 raw filter 的替代，而是高质量数据筛选的第二视角；它能覆盖 raw 阶段看不到的问题，例如转换后运动是否平滑、rot6d row-major/column-concat 错误、夹爪二值化语义错误、domain/time attrs 漏写、跨源 schema 是否稳定。

---

## 2. 对齐 vs 过滤的边界（含「tip2base 之外还要对齐什么」）

`~/tip2base` 与当前转换脚本已用于把 PIKA/UMI processed 数据转换到 Piper/teleop 使用的 robot-base 绝对表达。当前训练产物中，PIKA/UMI 的 `observations/qpos` attrs 已要求：

```text
pose_frame = robot_base_tip2base_piper_tcp_config
tip2base_applied = True
relative_to_first_frame = False
```

因此，文档里不应再把“tip2base 只做位置、朝向未对齐”作为当前训练数据事实。更准确的边界是：

1. **data_filter 不执行对齐**：它只检查 raw 数据是否值得进入转换，以及 processed 数据是否符合转换后的契约。
2. **坐标系/朝向/夹爪/schema 的实现属于 align/convert**：例如 tip2base、pose frame adjustment、rot6d 格式转换、gripper threshold/positive 统一。
3. **data_filter 必须验证 align/convert 的结果，并给出质量分层**：检查 `pose_frame`、`tip2base_applied`、`relative_to_first_frame`、rot6d 合法性、gripper 二值化、domain/time attrs；同时输出 episode-level quality score。
4. **速度分布是筛选信号，不只是诊断**：PIKA 手持速度与遥操速度可能不同，是否下采样/重采样由 align/training 策略决定；filter 阶段应标记过快、过慢、长时间静止、jerk 异常等 episode，默认进入降权或复核，而不是简单 hard drop。
5. **视觉/视角域对齐先不做重依赖处理**：v1 只做 JPEG decode、黑帧、模糊、静止段、decode 约定一致性检查；不做 VLM/SAM/render/inpaint。

---

## 3. 检查矩阵（按 schema 落地，不照抄论文）

第一版覆盖范围：**轻量子集**（纯信号 + 图像处理，零重依赖）。

### 3.1 Raw quality gate

| 检查 | pika | 遥操 | 说明 |
|---|:--:|:--:|---|
| **S1 突变检测**（残差 / 加速度 / jerk 三阈值联合） | ✅ | ✅ | 用导数，与基站漂移无关 |
| **S1′ 跟踪丢失 / 瞬移**（lighthouse dropout） | ✅ 关键 | — | pika 特有：基站丢跟踪 → NaN / 位置瞬移 |
| **S2 state-action 趋势对齐** | ❌ 无 action | ✅ | `qpos` vs `action` 互相关估时序滞后 + 方向一致性 |
| **S3 极值 / 分位带** `[q01,q99]→[−1,1]` | △ 用速度/delta | ✅ 绝对值可用 | pika 站系漂移 → 避免用绝对位置；夹爪豁免（双峰分布） |
| **时间戳 / 丢帧** | ✅ | ✅ | dt 规整性、跳变、同步偏差 |
| **模态长度一致** | ✅ | ✅ | 帧数 == pose == timestamps，廉价抓 sync bug |
| **C3 视频质量**（黑帧 / 模糊 / 长静止） | ✅ 三相机 | ✅ 三相机 | 保留夹爪闭合等关键帧 |
| **速度分布**（→ quality score，非 hard drop） | ✅ | ✅ | 过快/过慢/长静止/jerk 异常 → 默认降权或复核（见 §2 第 4 点）；分布亦供速度对齐定下采样比例 |
| C1 指令一致(VLM) / C2 video-state IoU(SAM3) / S4 FK(URDF) | — | — | v1 **不做**（重依赖，本版选轻量子集） |

论文原始映射：S1–S5 = 五阶段信号过滤；C1–C3 = 三项跨模态检查。v1 优先取 S1/S3 + C3，并按 pika 特性新增 S1′（跟踪丢失）与「模态长度 / 时间戳」两项廉价一致性检查；S2 与 C2 放入待升级版本。

#### 待升级：S2 State-Action Trend Alignment

S2 用于检查 `action` 与 `state change` 的因果趋势是否对齐，主要抓 **state-action 时序错位**，不是视频-state 对齐。基本假设是 action 应领先或同步于 state 变化。升级版做法：

1. 对 state/action 每个共享维度做平滑。
2. 用 cross-correlation 估计最佳 lag。
3. 在 lag 对齐后的一阶差分上计算 directional agreement（DA）。
4. DA 低于阈值（通常 `0.6-0.7`）则标记或剔除 episode。

该检查只适用于有独立 action 的遥操/NAS 数据；pika/UMI raw 无独立 action，默认不跑 S2。

#### 待升级：C2 Video-State Consistency

C2 是更接近“视频和 state 是否对齐”的跨模态检查，但依赖 URDF、相机参数和分割模型，v1 不做。升级版做法：

1. 用 URDF + 记录的 robot joint/state，把机器人重投影到图像平面。
2. 用 fine-tuned SAM3 或等价分割模型，从真实视频图像中分割机器人 mask。
3. 比较重投影 robot mask 与真实视频 robot mask 的 IoU。
4. IoU 低说明 video-state 不一致；若主要由相机参数导致，先优化相机参数，否则剔除 episode。

### 3.2 Processed XVLA quality gate

| 检查 | PIKA/UMI processed | NAS/teleop processed | 说明 |
|---|:--:|:--:|---|
| **schema/shape** | ✅ | ✅ | `observations/qpos` shape `(T,20)`；图像与 qpos 长度一致 |
| **finite** | ✅ | ✅ | qpos/action/timestamps 无 NaN/Inf |
| **rot6d 合法性** | ✅ | ✅ | `||a||≈1`、`||b||≈1`、`a·b≈0`、`det([a,b,a×b])≈1` |
| **rot6d 格式契约** | ✅ | ✅ | PIKA/UMI: `concat(R[:,0],R[:,1])`；NAS: 已从 row-major 转为 concat-cols |
| **夹爪语义** | ✅ | ✅ | `unique(qpos[:,9/19]) subset {0,1}`；attrs `gripper_positive=closed`、threshold 合理 |
| **坐标系 attrs** | ✅ 关键 | △ | PIKA/UMI 必须 `pose_frame=robot_base_tip2base_piper_tcp_config`、`tip2base_applied=True`、`relative_to_first_frame=False` |
| **domain attrs** | ✅ | ✅ | `domain_id/domain_name` 存在且符合配置 |
| **time alignment attrs** | ✅ | ✅ | `time_alignment_status=verified_common_time_axis`；timestamp 单调，dt 分布合理 |
| **decode 约定一致性**（原 RGB/BGR audit） | ✅ | ✅ | 验证 raw→processed 用同一 decode contract：核对 decode 路径/attrs，或对**固定参考帧**比对通道统计；只能验证约定一致性，不能从单张图像本身可靠判断 RGB/BGR 语义对错。TB 可视化只做显示转换，不改训练输入 |
| **manifest coverage** | ✅ | ✅ | manifest 文件数、路径、训练配置 `dataset_path` 权重与实际 HDF5 一致 |
| **运动质量评分** | ✅ | ✅ | 位姿速度、加速度、jerk、长静止段、过快段；输出 score/flag，不默认硬删 |
| **任务有效性 proxy** | △ | △ | 是否有接近目标/夹爪动作/明显操作阶段；v1 先用启发式，后续可接 VLM |
| **lazy reader smoke test**（可插拔 adapter） | ✅ | ✅ | 用当前训练 config 抽 1–2 batch 确认数据可被训练消费；经**可选 adapter / 子进程**调用训练 loader，data_filter **不硬依赖训练仓库**；工程可用性检查，不直接代表质量 |

Processed 检查要作为训练前的质量筛选与分层依据。它不只是防止训练报错，还要决定 episode 的使用策略：

```text
keep_high_quality    -> 默认进入训练
keep_with_downweight -> 可进入训练，但采样权重降低
review               -> 输出样例给人工看
drop                 -> 默认剔除
```

其中 shape/finite/rot6d/domain/time 这类属于 hard validity；运动平滑性、模糊、长静止、速度分布属于 quality score，不宜一刀切 hard drop。

---

## 4. 项目结构（拟）

```
data_filter/
├── PLAN.md                          # 本文件
├── CLAUDE.md                        # 待建
├── docs/{spec.md, CONTEXT.md}       # CONTEXT 术语: station frame / frame0(P0) / dwell / canonical / source ...
├── data_filter/
│   ├── io/{schema.py, loaders.py}   # raw pika/teleop + processed xvla loaders → 统一内部表示
│   ├── checks/                      # 每个检查 = 可独立测试的纯函数
│   │   ├── spike.py                 # S1 突变
│   │   ├── tracking.py              # S1′ pika lighthouse dropout
│   │   ├── state_action.py          # S2（遥操）
│   │   ├── extreme.py               # S3 极值/分位带
│   │   ├── timestamp.py             # 时间戳/丢帧
│   │   ├── modality.py              # 模态长度一致
│   │   ├── video.py                 # C3 视频质量
│   │   ├── rot6d.py                 # processed XVLA rot6d 合法性
│   │   ├── gripper.py               # processed XVLA 夹爪二值/语义
│   │   ├── attrs.py                 # domain/time/pose_frame/source_kind attrs
│   │   └── manifest.py              # manifest coverage + dataset_path 一致性
│   ├── adapters/                    # 可选：训练 loader 适配器（lazy-reader smoke test 用；可插拔，不硬依赖训练仓库）
│   ├── report/                      # per-episode / per-dim JSON + 可读 markdown/html
│   ├── config.py                    # 阈值（按源分离，论文默认打底）
│   ├── scoring.py                   # flags/metrics → quality score / keep-review-drop 决策
│   └── pipeline.py                  # 遍历 episode → 跑 checks → 决策 → 写 mask/score/report/splits
├── configs/{raw_pika.yaml, raw_teleop.yaml, processed_xvla.yaml}
├── tests/                           # 合成信号注入(spike/dropout/静止段) 断言被抓
└── scripts/run_filter.py
```

### 设计原则
1. 每个 check 是**纯函数**（输入信号 → 返回 flag mask + 指标），可独立测试，走 `/tdd`。
2. 阈值**按源分离**、可配置；论文默认值打底。
3. 决策**三级**：frame mask + episode score + dataset split/exclude-list；**输出 mask + score + 报告，不物理删除原数据**（可追溯）。Processed XVLA 阶段默认生成 `keep_high_quality / keep_with_downweight / review / drop` 列表，不重写 HDF5。
4. 张量操作**标注 shape**。
5. 文档与注释用中文。

---

## 5. 模块里程碑（每次对话一个模块）

1. 脚手架 + `spec.md` + `CONTEXT.md` + `io`（raw pika/raw teleop/processed xvla 三类 loader）+ selftest
2. **Processed XVLA checks 优先**：schema/shape、finite、rot6d、gripper、attrs、manifest、motion quality、visual quality、lazy-reader smoke test
3. Raw 信号类 checks（S1 / S1′ / S3 / timestamp / modality）—— `/tdd`
4. 视频类 check（C3）+ decode 约定一致性 + 速度诊断
5. 遥操 S2：state-action trend alignment（cross-correlation + DA）
6. 升级版 C2：URDF 重投影 + 分割 mask IoU 的 video-state consistency
7. `scoring` + `pipeline` 编排 + `report` + config
8. 真数据跑通 + 阈值调参 + review

---

## 6. 待确认（动手前拍板）

1. **流水线顺序**：是否确认采用两层闸门？
   - raw filter：对齐/转换前，过滤明显坏采集。
   - processed XVLA filter：训练前，验证转换产物契约。
2. **数据根目录与规模**：
   - raw pika/UMI 根目录。
   - raw NAS/teleop 根目录。
   - processed XVLA 根目录：一个或多个 processed XVLA 数据根目录，例如 `/data/xvla_market_bottle/...`。
3. **输出形态**：
   - raw 阶段：mask + quality report + optional exclude-list。
   - processed 阶段：quality report + score file + keep/review/drop lists + optional sampling weights；默认不重写 HDF5。
4. **硬失败 vs 质量降权标准**：
   - hard fail：shape/finite/rot6d/gripper/domain/time/manifest。
   - quality score：速度、jerk、长静止、模糊、黑帧比例、操作阶段覆盖率。
   - review：处于阈值边界或多项轻微异常叠加的 episode。

---

## 附：参考

- 论文：Qwen-RobotManip Technical Report（arXiv:2606.17846），§2.3 human-to-robot 合成、§2.4 数据 curation（S1–S5 + C1–C3）、§3.2 canonical state-action 表示。
- 相关项目：`~/tip2base`；当前 market_bottle 转换脚本已将 PIKA/UMI processed qpos 写为 `robot_base_tip2base_piper_tcp_config` 绝对 pose，并统一 rot6d/夹爪语义。
- 精读笔记：`~/claude-papers/papers/qwen-robotmanip/`。
