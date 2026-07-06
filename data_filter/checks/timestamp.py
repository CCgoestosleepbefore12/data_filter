"""时间戳检查（raw + processed 通用，hard-validity）。

判据：单调递增（dt>0）、无异常大跳变（max dt ≤ max_dt_ratio·中位 dt）。
多时钟同步偏差（遥操 eef_left/right_time）留待 raw teleop 阶段。

TODO(raw 阶段): 多时钟同步偏差。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_timestamp(ts: np.ndarray, cfg: dict, name: str = "timestamp") -> CheckResult:
    """ts: (T,) 秒。返回单调性 + dt 分布指标，非单调 / 大跳变 → hard_fail。"""
    t = np.asarray(ts, dtype=np.float64)
    if t.size < 2:
        return CheckResult(name, passed=True, severity="info", metrics={"n": int(t.size)})
    if not np.all(np.isfinite(t)):
        return CheckResult.hard(name, False, flags=["nan_inf"])

    dt = np.diff(t)                                        # (T-1,)
    med = float(np.median(dt))
    dt_max_ratio = float(np.max(dt) / med) if med > 0 else float("inf")
    metrics = {"dt_median": med, "dt_max_ratio": dt_max_ratio, "monotonic": bool(np.all(dt > 0))}

    flags = []
    if not np.all(dt > 0):
        flags.append("non_monotonic")
    if dt_max_ratio > cfg.get("max_dt_ratio", 3.0):
        flags.append("dt_jump")
    return CheckResult.hard(name, not flags, metrics=metrics, flags=flags)
