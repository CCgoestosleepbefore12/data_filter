"""S2 state-action 趋势对齐（仅遥操；pika 无独立 action）。

在一段正确录制里，action 应时序上领先或同步于 state 变化。
对每个共享关节维：平滑 → 互相关估最优时滞 → 一阶差分方向一致性(DA)。
DA 低于阈值的维/episode 标记（论文里 RoboMIND UR 型 81% episode 因此被剔）。

V2 轻量实现：用 qpos 一阶差分近似 state change，与 action 命令做时滞搜索和
方向一致性检查。该检查只作为 quality flag，不直接 hard drop。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_state_action(qpos: np.ndarray, action: np.ndarray, cfg: dict) -> CheckResult:
    """qpos/action: 各 (T, 14) 关节空间。返回时滞 + 方向一致性指标。"""
    q = np.asarray(qpos, dtype=np.float64)
    a = np.asarray(action, dtype=np.float64)
    if q.ndim != 2 or a.ndim != 2 or q.shape[0] < 4 or a.shape[0] < 4:
        return CheckResult(name="state_action", passed=True, severity="warn", flags=["too_short"])
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(a)):
        return CheckResult.hard("state_action", False, flags=["nonfinite"])

    n = min(q.shape[0] - 1, a.shape[0])
    d = min(q.shape[1], a.shape[1], int(cfg.get("dims", min(q.shape[1], a.shape[1]))))
    state_delta = np.diff(q[: n + 1, :d], axis=0)
    action_sig = a[:n, :d]

    max_points = int(cfg.get("max_points", 300))
    if max_points > 0 and n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        state_delta = state_delta[idx]
        action_sig = action_sig[idx]

    smooth_window = int(cfg.get("smooth_window", 5))
    state_delta = _smooth(state_delta, smooth_window)
    action_sig = _smooth(action_sig, smooth_window)

    max_lag = int(cfg.get("max_lag_frames", 15))
    active_eps = float(cfg.get("active_eps", 1e-4))
    min_active_ratio = float(cfg.get("min_active_ratio", 0.05))
    da_threshold = float(cfg.get("da_threshold", 0.65))
    corr_threshold = float(cfg.get("corr_threshold", -1.0))

    best_lag, best_corr = _best_lag(action_sig, state_delta, max_lag)
    aa, ss = _align_by_lag(action_sig, state_delta, best_lag)
    active = (np.abs(aa) > active_eps) | (np.abs(ss) > active_eps)
    active_ratio = float(np.count_nonzero(active) / active.size) if active.size else 0.0

    if np.count_nonzero(active) == 0:
        directional_agreement = 1.0
    else:
        directional_agreement = float(np.mean(np.sign(aa[active]) == np.sign(ss[active])))

    flags: list[str] = []
    if active_ratio >= min_active_ratio and directional_agreement < da_threshold:
        flags.append("low_directional_agreement")
    if abs(best_lag) >= max_lag and active_ratio >= min_active_ratio:
        flags.append("large_lag")
    if corr_threshold >= 0 and best_corr < corr_threshold and active_ratio >= min_active_ratio:
        flags.append("low_correlation")

    return CheckResult(
        name="state_action",
        passed=True,
        severity="warn" if flags else "info",
        metrics={
            "best_lag": int(best_lag),
            "best_corr": float(best_corr),
            "directional_agreement": directional_agreement,
            "active_frame_ratio": active_ratio,
            "dims": int(d),
        },
        flags=flags,
    )


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.shape[0] < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], kernel, mode="same")
    return out


def _best_lag(action: np.ndarray, state_delta: np.ndarray, max_lag: int) -> tuple[int, float]:
    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        aa, ss = _align_by_lag(action, state_delta, lag)
        if aa.size == 0:
            continue
        corr = _corrcoef_flat(aa, ss)
        if corr > best_corr:
            best_lag = lag
            best_corr = corr
    return best_lag, best_corr


def _align_by_lag(action: np.ndarray, state_delta: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        n = min(action.shape[0], state_delta.shape[0] - lag)
        if n <= 0:
            return action[:0], state_delta[:0]
        return action[:n], state_delta[lag: lag + n]
    offset = -lag
    n = min(action.shape[0] - offset, state_delta.shape[0])
    if n <= 0:
        return action[:0], state_delta[:0]
    return action[offset: offset + n], state_delta[:n]


def _corrcoef_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1)
    y = b.reshape(-1)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)
