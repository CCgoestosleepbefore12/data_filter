"""时间戳检查（raw + processed 通用）。

判据：单调递增（dt>0）为 hard-validity；异常大跳变作为 quality flag。
多时钟同步偏差用于 raw teleop 的 eef_left/right_time 质量检查。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_timestamp(ts: np.ndarray, cfg: dict, name: str = "timestamp") -> CheckResult:
    """ts: (T,) 秒。非单调 hard_fail；dt 大跳变只进入 quality 分层。"""
    try:
        t = np.asarray(ts, dtype=np.float64)
    except (TypeError, ValueError):
        return CheckResult.hard(name, False, flags=["nonnumeric"])
    if t.size < 2:
        if not np.all(np.isfinite(t)):
            return CheckResult.hard(name, False, metrics={"n": int(t.size)}, flags=["nonfinite"])
        return CheckResult(name, passed=True, severity="info", metrics={"n": int(t.size)})
    if not np.all(np.isfinite(t)):
        return CheckResult.hard(name, False, flags=["nonfinite"])

    dt = np.diff(t)                                        # (T-1,)
    med = float(np.median(dt))
    dt_max_ratio = float(np.max(dt) / med) if med > 0 else float("inf")
    max_dt_ratio = float(cfg.get("max_dt_ratio", 3.0))
    metrics = {
        "dt_median": med,
        "dt_max_ratio": dt_max_ratio,
        "max_dt_ratio": max_dt_ratio,
        "monotonic": bool(np.all(dt > 0)),
    }

    hard_flags = []
    quality_flags = []
    if not np.all(dt > 0):
        hard_flags.append("non_monotonic")
    if dt_max_ratio > max_dt_ratio:
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


def check_clock_skew(
    reference_ts: np.ndarray,
    other_ts: np.ndarray,
    cfg: dict,
    name: str = "timestamp_skew",
) -> CheckResult:
    """检查两路同源时钟是否同步。超阈值作为 quality flag，非数值/空数组 hard fail。"""
    try:
        ref = np.asarray(reference_ts, dtype=np.float64)
        other = np.asarray(other_ts, dtype=np.float64)
    except (TypeError, ValueError):
        return CheckResult.hard(name, False, flags=["nonnumeric"])
    n = min(ref.size, other.size)
    if n == 0:
        return CheckResult.hard(name, False, metrics={"n": int(n)}, flags=["missing"])
    if not np.all(np.isfinite(ref[:n])) or not np.all(np.isfinite(other[:n])):
        return CheckResult.hard(name, False, metrics={"n": int(n)}, flags=["nonfinite"])

    skew = np.abs(ref[:n] - other[:n])
    max_skew = float(np.max(skew)) if skew.size else 0.0
    threshold = float(cfg.get("max_clock_skew_s", 0.05))
    flags = ["clock_skew"] if max_skew > threshold else []
    return CheckResult(
        name=name,
        passed=True,
        severity="warn" if flags else "info",
        metrics={
            "n": int(n),
            "max_clock_skew_s": max_skew,
            "median_clock_skew_s": float(np.median(skew)) if skew.size else 0.0,
            "threshold_s": threshold,
        },
        flags=flags,
    )
