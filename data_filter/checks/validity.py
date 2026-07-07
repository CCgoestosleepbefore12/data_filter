"""结构性 hard-validity 检查：schema/shape 与 finite。

- check_schema_shape: qpos 必须 (T, 20)。
- check_finite:       qpos/action/timestamps 无 NaN/Inf。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

import numpy as np

from ..io import schema
from .base import CheckResult


def check_schema_shape(qpos: np.ndarray, cfg: dict, name: str = "schema_shape") -> CheckResult:
    """qpos 必须是 (T, 20)，且默认至少 1 帧。"""
    min_length = int(cfg.get("min_length", 1))
    ok = qpos.ndim == 2 and qpos.shape[1] == schema.QPOS_DIM and qpos.shape[0] >= min_length
    flags = []
    if not (qpos.ndim == 2 and qpos.shape[1] == schema.QPOS_DIM):
        flags.append("bad_shape")
    if qpos.ndim >= 1 and qpos.shape[0] < min_length:
        flags.append("too_short")
    return CheckResult.hard(
        name, ok, metrics={"shape": list(qpos.shape), "min_length": min_length}, flags=flags
    )


def check_finite(arrays, cfg: dict, name: str = "finite") -> CheckResult:
    """arrays: {名: array} 或单 array。任一含 NaN/Inf → hard_fail。"""
    items = arrays.items() if isinstance(arrays, dict) else [("array", arrays)]
    bad = []
    for nm, arr in items:
        if arr is None:
            continue
        try:
            a = np.asarray(arr)
            if not np.issubdtype(a.dtype, np.number):
                bad.append(f"nonnumeric:{nm}")
                continue
            if not np.all(np.isfinite(a.astype(np.float64, copy=False))):
                bad.append(f"nonfinite:{nm}")
        except (TypeError, ValueError):
            bad.append(f"nonnumeric:{nm}")
    ok = not bad
    return CheckResult.hard(name, ok, flags=bad)


def check_min_length(length: int, cfg: dict, name: str = "min_length") -> CheckResult:
    """episode 至少要有 min_length 帧。"""
    min_length = int(cfg.get("min_length", 1))
    ok = int(length) >= min_length
    return CheckResult.hard(
        name, ok, metrics={"length": int(length), "min_length": min_length}, flags=[] if ok else ["too_short"]
    )


def check_required_keys(present, required, name: str = "required_keys") -> CheckResult:
    """要求必需字段/模态存在。present/required 可为任意可迭代 key 集。"""
    present_set = set(present)
    missing = [k for k in required if k not in present_set]
    ok = not missing
    return CheckResult.hard(
        name,
        ok,
        metrics={"present": sorted(present_set), "required": list(required)},
        flags=[f"missing:{m}" for m in missing],
    )
