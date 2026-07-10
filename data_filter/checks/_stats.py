"""Small shared statistics helpers for quality checks."""

from __future__ import annotations

import numpy as np


def robust_limit(values: np.ndarray, k: float, *, fallback_sigma: float = 3.0, eps: float = 1e-12) -> float:
    """median + k * MAD, with std/epsilon fallback for degenerate signals."""
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad > eps:
        return med + k * 1.4826 * mad
    std = float(np.std(values))
    if std > eps:
        return med + fallback_sigma * std
    return med + eps


def longest_true_run(mask: np.ndarray) -> int:
    longest = cur = 0
    for hit in np.asarray(mask).astype(bool):
        if hit:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)
