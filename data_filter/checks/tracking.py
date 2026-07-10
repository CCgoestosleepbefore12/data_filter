"""S1′ 跟踪丢失 / 瞬移（pika 特有）。

pika 用 lighthouse 基站定位，丢跟踪会产生 NaN / 位置瞬移 / 冻结重复帧。
检测：位姿 NaN、单帧位移超阈（teleport）、长段完全不变（frozen）。
质量+hard 混合：NaN 可 hard_fail，teleport/frozen 出 frame_mask + flags。

NaN 属 hard_fail；瞬移 / 冻结作为 quality flag。
"""

from __future__ import annotations

import numpy as np

from ._stats import longest_true_run
from .base import CheckResult


def check_tracking(pose: np.ndarray, cfg: dict) -> CheckResult:
    """pose: (T, 6) pika 位姿（station frame）。返回丢跟踪 mask 与指标。"""
    x = np.asarray(pose, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return CheckResult.hard("tracking", False, flags=["bad_shape"])
    finite_rows = np.all(np.isfinite(x), axis=1)
    if not np.all(finite_rows):
        return CheckResult(
            name="tracking",
            passed=False,
            severity="hard_fail",
            frame_mask=~finite_rows,
            metrics={"nan_frames": int(np.count_nonzero(~finite_rows))},
            flags=["nonfinite"],
        )

    pos = x[:, :3]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    teleport_m = float(cfg.get("teleport_m", 0.10))
    frozen_min_frames = int(cfg.get("frozen_min_frames", 30))
    frozen_eps = float(cfg.get("frozen_eps", 1e-9))
    teleport = step > teleport_m
    frozen_step = step <= frozen_eps

    longest_frozen = longest_true_run(frozen_step)

    flags = []
    if np.any(teleport):
        flags.append("teleport")
    if longest_frozen >= frozen_min_frames:
        flags.append("frozen")

    frame_mask = np.zeros(x.shape[0], dtype=bool)
    frame_mask[1:][teleport] = True
    return CheckResult(
        name="tracking",
        passed=True,
        severity="warn" if flags else "info",
        frame_mask=frame_mask,
        metrics={
            "max_step_m": float(step.max()) if step.size else 0.0,
            "teleport_frames": int(np.count_nonzero(teleport)),
            "longest_frozen_run": int(longest_frozen),
        },
        flags=flags,
    )
