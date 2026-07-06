"""S1 突变检测（raw + processed 通用）。

对每个信号维度提取平滑趋势，联合三个阈值判定突变帧：
残差（raw vs 平滑）、二阶差分（加速度）、三阶差分（jerk）。
用导数 → 与 pika 基站漂移无关。质量类：输出 frame_mask + metrics，不 hard_fail。

第一版用 diff / accel / jerk 的 MAD 阈值检测突变；quality 类，不 hard drop。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_spike(signal: np.ndarray, cfg: dict) -> CheckResult:
    """signal: (T, D) 位姿/关节/动作轨迹。返回逐帧突变 mask 与指标。"""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 4:
        return CheckResult(name="spike", passed=True, severity="warn", flags=["too_short"])
    if not np.all(np.isfinite(x)):
        return CheckResult.hard("spike", False, flags=["nan_inf"])

    k = float(cfg.get("jerk_sigma", cfg.get("accel_sigma", cfg.get("residual_sigma", 6.0))))
    min_frames = int(cfg.get("min_spike_frames", 3))

    vel = np.diff(x, axis=0)
    accel = np.diff(vel, axis=0)
    jerk = np.diff(accel, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1)
    accel_mag = np.linalg.norm(accel, axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)

    def limit(v: np.ndarray) -> float:
        med = float(np.median(v)) if v.size else 0.0
        mad = float(np.median(np.abs(v - med))) if v.size else 0.0
        return med + k * 1.4826 * mad if mad > 1e-12 else float(v.max() if v.size else 0.0)

    accel_lim = limit(accel_mag)
    jerk_lim = limit(jerk_mag)
    accel_hit = accel_mag > accel_lim if accel_lim > 0 else np.zeros_like(accel_mag, dtype=bool)
    jerk_hit = jerk_mag > jerk_lim if jerk_lim > 0 else np.zeros_like(jerk_mag, dtype=bool)

    frame_mask = np.zeros(x.shape[0], dtype=bool)
    frame_mask[2:][accel_hit] = True
    frame_mask[3:][jerk_hit] = True
    hit_count = int(np.count_nonzero(frame_mask))
    flags = ["spike"] if hit_count >= min_frames else []
    return CheckResult(
        name="spike",
        passed=True,
        severity="warn" if flags else "info",
        frame_mask=frame_mask,
        metrics={
            "vel_median": float(np.median(vel_mag)) if vel_mag.size else 0.0,
            "vel_max": float(vel_mag.max()) if vel_mag.size else 0.0,
            "accel_max": float(accel_mag.max()) if accel_mag.size else 0.0,
            "accel_limit": float(accel_lim),
            "jerk_max": float(jerk_mag.max()) if jerk_mag.size else 0.0,
            "jerk_limit": float(jerk_lim),
            "spike_frames": hit_count,
        },
        flags=flags,
    )
