"""时间戳检查（raw + processed 通用）。

判据：单调递增（dt>0）为 hard-validity；异常大跳变作为 quality flag。
多时钟同步偏差（遥操 eef_left/right_time）留待 raw teleop 阶段。

TODO(raw 阶段): 多时钟同步偏差。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_timestamp(ts: np.ndarray, cfg: dict, name: str = "timestamp") -> CheckResult:
    """ts: (T,) 秒。非单调 hard_fail；dt 大跳变只进入 quality 分层。"""
    t = np.asarray(ts, dtype=np.float64)
    if t.size < 2:
        return CheckResult(name, passed=True, severity="info", metrics={"n": int(t.size)})
    if not np.all(np.isfinite(t)):
        return CheckResult.hard(name, False, flags=["nan_inf"])

    dt = np.diff(t)                                        # (T-1,)
    med = float(np.median(dt))
    dt_max_ratio = float(np.max(dt) / med) if med > 0 else float("inf")
    metrics = {"dt_median": med, "dt_max_ratio": dt_max_ratio, "monotonic": bool(np.all(dt > 0))}

    hard_flags = []
    quality_flags = []
    if not np.all(dt > 0):
        hard_flags.append("non_monotonic")
    if dt_max_ratio > cfg.get("max_dt_ratio", 3.0):
        quality_flags.append("dt_jump")
    if hard_flags:
        return CheckResult.hard(name, False, metrics=metrics, flags=hard_flags)
    return CheckResult(
        name=name,
        passed=True,
        severity="warn" if quality_flags else "info",
        metrics=metrics,
        flags=quality_flags,
    )
