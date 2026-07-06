"""模态长度一致检查（raw + processed 通用，廉价高收益）。

各相机帧数、pose/qpos、gripper、timestamps 的 T 必须一致；
不一致往往是采集/同步 bug。hard-validity：不一致 → hard_fail。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

from .base import CheckResult


def check_modality_lengths(lengths: dict[str, int], cfg: dict, name: str = "modality") -> CheckResult:
    """lengths: {模态名: T}。所有模态 T 必须相等，否则 hard_fail。"""
    ok = len(set(lengths.values())) <= 1
    return CheckResult.hard(
        name, ok, metrics={"lengths": dict(lengths)}, flags=[] if ok else ["length_mismatch"]
    )
