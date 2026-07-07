"""夹爪语义检查（processed，hard-validity）。

processed qpos 的夹爪维（LEFT_GRIP=9, RIGHT_GRIP=19）应为二值，positive=closed。
检查：finite、值域 [0,1]、二值（贴近 0/1）。正负翻转（positive=closed 反了）
需语义参照，v1 不查。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_gripper(gripper: np.ndarray, attrs: dict, cfg: dict, name: str = "gripper") -> CheckResult:
    """gripper: (T,) 或 (T, n_arm)。判据：finite、值域 [0,1]、二值（贴近 0/1）。

    正负翻转（positive=closed 反了）需语义参照，v1 不查。
    """
    g = np.asarray(gripper, dtype=np.float64)
    tol = cfg.get("binary_tol", 1e-3)
    if not np.all(np.isfinite(g)):
        return CheckResult.hard(name, False, flags=["nonfinite"])

    gmin, gmax = float(g.min()), float(g.max())
    in_range = gmin >= -tol and gmax <= 1.0 + tol
    binary_dev = float(np.abs(g - np.round(np.clip(g, 0.0, 1.0))).max())
    metrics = {"min": gmin, "max": gmax, "binary_dev": binary_dev}

    flags = []
    if not in_range:
        flags = ["out_of_range"]
    elif binary_dev > tol:
        flags = ["not_binary"]
    return CheckResult.hard(name, not flags, metrics=metrics, flags=flags)
