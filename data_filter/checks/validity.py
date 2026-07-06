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
    """qpos 必须是 (T, 20)。"""
    ok = qpos.ndim == 2 and qpos.shape[1] == schema.QPOS_DIM
    return CheckResult.hard(
        name, ok, metrics={"shape": list(qpos.shape)}, flags=[] if ok else ["bad_shape"]
    )


def check_finite(arrays, cfg: dict, name: str = "finite") -> CheckResult:
    """arrays: {名: array} 或单 array。任一含 NaN/Inf → hard_fail。"""
    items = arrays.items() if isinstance(arrays, dict) else [("array", arrays)]
    bad = []
    for nm, arr in items:
        if arr is None:
            continue
        if not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
            bad.append(nm)
    ok = not bad
    return CheckResult.hard(name, ok, flags=[f"nonfinite:{b}" for b in bad])
