"""rot6d 合法性检查（processed，hard-validity）。

契约（约定 A，见 io/schema.py）：rot6d = concat(R[:,0], R[:,1])，真 R 的前两列，
构造即正交。记 a=rot6d[:3]、b=rot6d[3:]，合法判据：
    finite、‖a‖≈1、‖b‖≈1、a·b≈0     （容差 ~1e-3，float32 + FK/euler→R roundtrip）

注：det([a,b,a×b]) = ‖a×b‖² ≥ 0 恒非负，**无法**判手性/镜像，故不作为判据
（只有语义参照如 eef_quaternion 才能查列/行互换这类保正交的错，超出 v1）。
非正交/非单位能抓到「取错 6 个数 / row-major→concat-cols 搞坏」这类破坏正交的 bug。

⚠️ 只对 **processed（适配后 pose9）** 用；raw eef_6d 的 rot 约定可能不同，勿误用。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_rot6d(rot6d: np.ndarray, cfg: dict, name: str = "rot6d") -> CheckResult:
    """rot6d: (T, 6) = concat(R[:,0], R[:,1])。返回正交性指标与 hard_fail。"""
    tol = cfg.get("tol", 1e-3)
    try:
        rot = np.asarray(rot6d, dtype=np.float64)
    except (TypeError, ValueError):
        return CheckResult.hard(name, False, flags=["nonnumeric"])
    if not np.all(np.isfinite(rot)):
        return CheckResult.hard(name, False, flags=["nonfinite"])

    a, b = rot[:, :3], rot[:, 3:]                       # 各 (T, 3)
    err_a = float(np.abs(np.linalg.norm(a, axis=1) - 1.0).max())
    err_b = float(np.abs(np.linalg.norm(b, axis=1) - 1.0).max())
    err_dot = float(np.abs(np.sum(a * b, axis=1)).max())
    metrics = {"norm_a_err": err_a, "norm_b_err": err_b, "dot_max": err_dot}

    flags = [
        f for f, hit in
        [("norm_a", err_a > tol), ("norm_b", err_b > tol), ("orthogonality", err_dot > tol)]
        if hit
    ]
    return CheckResult.hard(name, not flags, metrics=metrics, flags=flags)
