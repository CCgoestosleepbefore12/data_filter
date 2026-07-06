"""时间戳 / 丢帧检查（raw + processed 通用）。

检查时间戳单调递增、dt 规整（无异常大跳变 / 抖动）、多时钟同步偏差
（遥操 eef_left/right_time vs 主时钟）。质量+hard：非单调可 hard_fail。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_timestamp(ts: np.ndarray, cfg: dict) -> CheckResult:
    """ts: (T,) 秒。返回单调性 / dt 分布 / 丢帧指标。"""
    raise NotImplementedError
