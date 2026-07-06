# CLAUDE.md — data_filter

进入本目录先读这份，再读 `docs/spec.md`（实现契约）与 `docs/CONTEXT.md`（术语）。设计 rationale 见 `PLAN.md`。

## 一句话

对多源机器人操作数据（**pika/UMI 手持** + **遥操/NAS**，均 HDF5）做**质量检查 + 高质量数据筛选**，方法依据 Qwen-RobotManip §2.4 curation 管线。

## 边界（重要）

- **只检查/筛选，不对齐、不重写训练数据**。坐标系/朝向/夹爪/canonical schema 的对齐属于 align/convert 阶段；本项目只**验证**输入或产物是否满足契约，输出 mask/report/score/split-list。
- 两道闸门：**Raw quality gate**（原始采集）+ **Processed XVLA quality gate**（转换产物）；外加两个横切阶段：quality scoring、dataset selection。

## 关键约定

- **hard-validity vs quality-score 严格分开**：
  - hard（失败即 drop/block）：schema/finite/rot6d/gripper/domain·time attrs/manifest/decode 约定。
  - score（不一刀切）：速度/jerk/长静止/模糊/黑帧/覆盖 → keep / downweight / review / drop。
- **输出不改原数据**：只写 mask + report + score + split-list。
- **阈值全部 provisional**：没有人工校准集，靠 `review` 队列后校准；第一版决策用**透明规则分级**，不塌成手调权重标量。
- **rot6d 契约**（约定 A，见 `io/schema.py`）：`concat(R[:,0], R[:,1])`，真 R 前两列，构造即正交 → 判据 `finite、‖a‖≈‖b‖≈1、a·b≈0`（det 无法判手性，不作判据）。**只对 processed pose9 用，勿用于 raw eef_6d**。

## 工程规范

- 环境用 **uv**（`uv sync` / `uv run pytest`）。
- 每个 check 是**纯函数**（输入信号 → `CheckResult`），可独立单测，走 **/tdd**。
- 张量操作**注释 shape**；文档与注释用**中文**。
- 每次对话只做一个模块，流程：读 spec → /plan → 确认 → 实现 → 测试 → /simplify → review → /commit。

## 目录导航

```
data_filter/
├── io/{schema.py, loaders.py}   # schema 声明（单一真相源）+ 三类 loader
├── checks/                      # 纯函数检查：spike/tracking/state_action/extreme/
│                                #   timestamp/modality/video + rot6d/gripper/attrs/manifest
│   └── base.py                  # CheckResult 契约
├── adapters/                    # 可插拔训练-loader 适配器（lazy-reader smoke test）
├── scoring.py  pipeline.py  report/  config.py
configs/{raw_pika,raw_teleop,processed_xvla}.yaml   # 按源分离，阈值 provisional
tests/   scripts/run_filter.py
```

## 当前状态

脚手架 + docs 完成（里程碑 1 的骨架）。下一步：io/loaders 实现 + selftest，然后 processed 检查（P0）。里程碑见 `docs/spec.md`。
