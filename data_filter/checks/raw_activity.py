"""Raw 双臂活动性检查。

用于抓 teleop/NAS 这类“某一臂整段信号缺失但数值有限”的情况。
典型表现是右臂 qpos/action/eef 半区几乎完全冻结，unique row 约等于 1。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_bimanual_activity(signal: np.ndarray, cfg: dict, name: str = "arm_activity") -> CheckResult:
    """signal: (T, 2D)，左右臂按维度前后半区划分。冻结臂作为 quality flag。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 2:
        return CheckResult(name=name, passed=True, severity="warn", flags=["too_short"])
    if not np.all(np.isfinite(x)):
        return CheckResult.hard(name, False, flags=["nonfinite"])

    mid = x.shape[1] // 2
    min_unique_rows = int(cfg.get("min_unique_rows", 3))
    min_mean_std = float(cfg.get("min_mean_std", 1e-5))
    decimals = int(cfg.get("unique_decimals", 6))

    metrics: dict[str, float | int] = {}
    flags: list[str] = []
    for arm, y in [("left", x[:, :mid]), ("right", x[:, mid:])]:
        unique_rows = int(len(np.unique(np.round(y, decimals), axis=0)))
        mean_std = float(np.mean(np.std(y, axis=0)))
        metrics[f"{arm}_unique_rows"] = unique_rows
        metrics[f"{arm}_mean_std"] = mean_std
        if unique_rows < min_unique_rows or mean_std < min_mean_std:
            flags.append(f"{arm}_arm_frozen")

    return CheckResult(
        name=name,
        passed=True,
        severity="warn" if flags else "info",
        metrics=metrics,
        flags=flags,
    )
