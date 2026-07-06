"""S2 state-action 趋势对齐（仅遥操；pika 无独立 action）。

在一段正确录制里，action 应时序上领先或同步于 state 变化。
对每个共享关节维：平滑 → 互相关估最优时滞 → 一阶差分方向一致性(DA)。
DA 低于阈值的维/episode 标记（论文里 RoboMIND UR 型 81% episode 因此被剔）。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_state_action(qpos: np.ndarray, action: np.ndarray, cfg: dict) -> CheckResult:
    """qpos/action: 各 (T, 14) 关节空间。返回时滞 + 方向一致性指标。"""
    raise NotImplementedError
