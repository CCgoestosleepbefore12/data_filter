"""坐标系 / domain / time attrs 检查（processed，hard-validity）。

按来源（pika_umi / nas_teleop）核对 PROCESSED_ATTRS 契约（见 io/schema.py）：
- pika_umi: pose_frame / tip2base_applied / relative_to_first_frame / domain_name / time_alignment_status
- nas_teleop: domain_name / source_key / time_alignment_status
缺失或不符 → hard_fail。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

from ..io import schema
from .base import CheckResult


def check_attrs(attrs: dict, source_kind: str, cfg: dict, name: str = "attrs") -> CheckResult:
    """按来源核对 PROCESSED_ATTRS 契约。attrs 缺失或不符 → hard_fail。

    契约值为 set 时表示「取值须属于该集合」（如 domain_name）；否则须精确相等。
    """
    contract = schema.PROCESSED_ATTRS.get(source_kind)
    if contract is None:
        return CheckResult.hard(name, False, flags=[f"unknown_source:{source_kind}"])

    missing, wrong = [], []
    for key, expected in contract.items():
        if key not in attrs:
            missing.append(key)
            continue
        v = attrs[key]
        bad = (v not in expected) if isinstance(expected, set) else (v != expected)
        if bad:
            wrong.append(key)

    flags = [f"missing:{k}" for k in missing] + [f"wrong:{k}" for k in wrong]
    return CheckResult.hard(name, not flags, metrics={"missing": missing, "wrong": wrong}, flags=flags)
