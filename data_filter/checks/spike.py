"""S1 突变检测（raw + processed 通用）。

对每个信号维度提取平滑趋势，联合三个阈值判定突变帧：
残差（raw vs 平滑）、二阶差分（加速度）、三阶差分（jerk）。
用导数 → 与 pika 基站漂移无关。质量类：输出 frame_mask + metrics，不 hard_fail。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_spike(signal: np.ndarray, cfg: dict) -> CheckResult:
    """signal: (T, D) 位姿/关节/动作轨迹。返回逐帧突变 mask 与指标。"""
    raise NotImplementedError
