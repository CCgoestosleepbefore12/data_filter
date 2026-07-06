"""Processed qpos 运动质量检查（quality 类）。

第一版只做 episode 内部的鲁棒异常检测，不做跨数据集阈值归一：
- 位置速度过快段
- jerk 过大段
- 长静止段
- 夹爪没有动作覆盖

这些信号用于 review/downweight，不直接 hard drop。
"""

from __future__ import annotations

import numpy as np

from ..io import schema
from .base import CheckResult


def _robust_limit(values: np.ndarray, k: float) -> float:
    """median + k * MAD，MAD 退化时回退到 max。"""
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 1e-12:
        return float(values.max())
    return med + k * 1.4826 * mad


def _longest_true_run(mask: np.ndarray) -> int:
    longest = cur = 0
    for hit in mask.astype(bool):
        if hit:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)


def check_motion_quality(qpos: np.ndarray, cfg: dict, name: str = "motion") -> CheckResult:
    """qpos: (T,20)。返回速度/jerk/静止/夹爪覆盖质量 flags。"""
    if qpos.ndim != 2 or qpos.shape[1] < schema.QPOS_DIM or qpos.shape[0] < 4:
        return CheckResult(name=name, passed=True, severity="warn", flags=["too_short"])

    k = float(cfg.get("robust_sigma", cfg.get("jerk_sigma", 6.0)))
    static_min_frames = int(cfg.get("static_min_frames", 45))
    static_speed_eps = float(cfg.get("static_speed_eps", 1e-5))
    min_gripper_changes = int(cfg.get("min_gripper_changes", 1))
    min_speed_outlier_frames = int(cfg.get("min_speed_outlier_frames", 3))
    min_jerk_outlier_frames = int(cfg.get("min_jerk_outlier_frames", 3))

    xyz = np.concatenate(
        [qpos[:, schema.LEFT_XYZ], qpos[:, schema.RIGHT_XYZ]], axis=1
    )  # (T,6)
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)  # (T-1,)
    accel = np.diff(speed)                                # (T-2,)
    jerk = np.diff(accel)                                 # (T-3,)

    speed_limit = _robust_limit(speed, k)
    jerk_abs = np.abs(jerk)
    jerk_limit = _robust_limit(jerk_abs, k)
    speed_fast = speed > speed_limit if speed_limit > 0 else np.zeros_like(speed, dtype=bool)
    jerk_spike = jerk_abs > jerk_limit if jerk_limit > 0 else np.zeros_like(jerk_abs, dtype=bool)
    static_run = _longest_true_run(speed <= static_speed_eps)

    gripper = qpos[:, [schema.LEFT_GRIP, schema.RIGHT_GRIP]]
    gripper_changes = int(np.count_nonzero(np.any(np.diff(gripper, axis=0) != 0, axis=1)))

    flags: list[str] = []
    speed_outlier_frames = int(np.count_nonzero(speed_fast))
    jerk_outlier_frames = int(np.count_nonzero(jerk_spike))

    if speed_outlier_frames >= min_speed_outlier_frames:
        flags.append("speed_outlier")
    if jerk_outlier_frames >= min_jerk_outlier_frames:
        flags.append("jerk_outlier")
    if static_run >= static_min_frames:
        flags.append("long_static")
    if gripper_changes < min_gripper_changes:
        flags.append("low_gripper_coverage")

    metrics = {
        "speed_median": float(np.median(speed)) if speed.size else 0.0,
        "speed_max": float(speed.max()) if speed.size else 0.0,
        "speed_limit": float(speed_limit),
        "speed_outlier_frames": speed_outlier_frames,
        "jerk_abs_median": float(np.median(jerk_abs)) if jerk_abs.size else 0.0,
        "jerk_abs_max": float(jerk_abs.max()) if jerk_abs.size else 0.0,
        "jerk_limit": float(jerk_limit),
        "jerk_outlier_frames": jerk_outlier_frames,
        "longest_static_run": static_run,
        "gripper_changes": gripper_changes,
    }
    return CheckResult(
        name=name,
        passed=True,
        severity="warn" if flags else "info",
        metrics=metrics,
        flags=flags,
    )
