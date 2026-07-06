"""S1′ 跟踪丢失 / 瞬移（pika 特有）。

pika 用 lighthouse 基站定位，丢跟踪会产生 NaN / 位置瞬移 / 冻结重复帧。
检测：位姿 NaN、单帧位移超阈（teleport）、长段完全不变（frozen）。
质量+hard 混合：NaN 可 hard_fail，teleport/frozen 出 frame_mask + flags。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_tracking(pose: np.ndarray, cfg: dict) -> CheckResult:
    """pose: (T, 6) pika 位姿（station frame）。返回丢跟踪 mask 与指标。"""
    raise NotImplementedError
