"""可插拔训练-loader 适配器（lazy-reader smoke test 用）。

通过可选 adapter / 子进程调用训练仓库的 lazy HDF5 reader，抽 1–2 batch，
确认数据能被训练消费。**data_filter 不硬依赖训练仓库**：未配置 adapter 时该检查跳过。

TODO(milestone 2+): 定义 adapter 协议 + 一个 xVLA 适配实现（可选依赖）。
"""

from __future__ import annotations

from typing import Protocol


class TrainLoaderAdapter(Protocol):
    """训练 loader 适配协议。实现方负责用训练 config 抽样 batch。"""

    def sample_batch(self, hdf5_path: str, n: int = 1) -> object:
        ...
