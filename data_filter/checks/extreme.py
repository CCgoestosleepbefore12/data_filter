"""S3 极值 / 分位带过滤。

按维计算 [q01, q99]，超出 [q01−α(q99−q01), q99+α(q99−q01)] 的帧标记。
夹爪维豁免（双峰分布）。pika 位姿在漂移基站系 → 用速度/delta 而非绝对位置。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_extreme(signal: np.ndarray, cfg: dict, exempt_dims: tuple[int, ...] = ()) -> CheckResult:
    """signal: (T, D)。exempt_dims: 豁免维（如夹爪）。返回越界 mask 与分位带。"""
    raise NotImplementedError
