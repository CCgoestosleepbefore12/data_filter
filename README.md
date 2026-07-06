# data_filter

多源机器人操作数据的**质量检查与高质量数据筛选**。方法依据 Qwen-RobotManip 论文（arXiv:2606.17846）§2.4 数据 curation 管线，并结合当前 xVLA / market_bottle 数据的真实踩坑。

数据源：**pika/UMI**（手持采集）与**遥操**（teleoperation/NAS），均为 HDF5。检查分两层：raw HDF5 与 processed XVLA HDF5。

## 边界

`data_filter` **只做质量检查 + 筛选**，不做跨源对齐、不重写训练数据。坐标系/朝向/夹爪/schema 的对齐属于 align/convert 阶段；本项目只**验证**输入或产物是否满足契约，并输出 mask / report / score / split-list。

## 文档

- [`PLAN.md`](PLAN.md) —— 方案设计与 rationale（为什么这么做）
- [`docs/spec.md`](docs/spec.md) —— 实现契约（检查判据、阈值、IO、决策策略）
- [`docs/CONTEXT.md`](docs/CONTEXT.md) —— 项目术语表
- [`CLAUDE.md`](CLAUDE.md) —— 进入本目录时的快速导航

## 环境

```bash
uv sync            # 安装依赖
uv run pytest      # 跑测试
```

## 状态

脚手架阶段。里程碑见 `docs/spec.md`。阈值均为 **provisional**，待真实分布与 `review` 队列校准。
