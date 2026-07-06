# CONTEXT — 项目术语表

随对话持续补充。每个术语一行定义，命名（变量/函数/文件）统一用这套词。

## 数据源
- **pika / UMI**：手持采集夹爪设备，lighthouse 基站定位；只有观测位姿+夹爪+图像，**无 action/joint/language**。
- **遥操 / teleop / NAS**：teleoperation 采集的真机数据（ALOHA 风格），有 action + qpos + eef + language。
- **raw**：原始采集 HDF5（未经 align/convert）。
- **processed / processed XVLA**：经 align/convert 后用于训练的 HDF5（统一 20D qpos）。

## 坐标系 / 对齐
- **station frame（基站系）**：pika lighthouse 基站的坐标系，原点**逐 session 漂移**。
- **frame0 / P0**：每条录制起步的固定工装位姿，tip2base 用它消基站漂移。
- **tip2base**：把 pika 尖端轨迹搬进松灵 robot base 的项目/步骤（`~/tip2base`）。
- **align / convert**：对齐+转换阶段（坐标系/朝向/夹爪/schema），**不属 data_filter**。

## 表示
- **canonical / ArmFrame**：统一的单臂表示 = xyz + rot6d + gripper。
- **pose9**：`[xyz(3), rot6d(6)]`，单臂 9 维。
- **rot6d**：6D 旋转表示 = `concat(R[:,0], R[:,1])`（真旋转矩阵前两列，列拼接）。约定 A=构造即正交。
- **qpos (20D)**：processed 训练向量 `[left_pose9, left_grip, right_pose9, right_grip]`。
- **domain_name / source_kind / time_alignment_status**：processed HDF5 的来源与时间对齐 attrs 契约。

## 检查 / 决策
- **Raw quality gate**：对 raw 数据的质量闸门。
- **Processed (validity) gate**：对 processed 产物的契约验证闸门。
- **hard-validity**：客观二值、必须过的检查（schema/finite/rot6d/gripper/attrs/manifest/decode）；失败即 drop/block。
- **quality-score**：proxy 质量指标（速度/jerk/静止/模糊/黑帧/覆盖），加权/规则聚合成分数，不一刀切。
- **CheckResult**：所有 check 的统一输出（name/passed/severity/frame_mask/metrics/flags）。
- **EpisodeSignals**：三类 loader 产出的统一内部信号表示。
- **S1/S1′/S2/S3**：论文式检查——突变 / 跟踪丢失(pika 新增) / state-action 趋势对齐 / 极值分位带。
- **decode 约定一致性**：验证 raw→processed 用同一图像 decode contract（非纯像素 BGR 检测）。
- **tracking dropout**：pika 基站丢跟踪导致的 NaN/瞬移/冻结。
- **keep_high_quality / keep_with_downweight / review / drop**：episode 四级决策标签。
- **exclude-list / split-list / sampling_weights**：筛选产物——剔除名单 / 分桶名单 / 采样权重。
- **provisional 阈值**：暂定阈值，待真实分布 + review 队列校准。
- **lazy-reader smoke test**：经可插拔 adapter 用训练 config 抽 batch，验证数据可被训练消费。
