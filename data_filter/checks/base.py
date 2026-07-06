"""检查结果的统一契约。

所有 check 都是**纯函数**：输入信号 / 元数据，输出 `CheckResult`。
不做 IO、不改数据，便于独立单测（走 /tdd）。

两类检查共用同一契约：
- **hard-validity**（schema/finite/rot6d/gripper/domain/time/manifest）：违反时
  `severity="hard_fail"`、`passed=False`。
- **quality**（速度/jerk/静止/模糊/黑帧/覆盖）：`passed=True` 恒成立，靠
  `frame_mask` + `metrics` + `flags` 表达问题，交由 scoring 聚合成分数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Severity = str  # "info" | "warn" | "hard_fail"


@dataclass
class CheckResult:
    """单个 check 的输出。"""

    name: str                                   # 检查名，如 "spike" / "rot6d"
    passed: bool = True                         # hard-validity 是否通过；quality 恒 True
    severity: Severity = "info"                 # info | warn | hard_fail
    frame_mask: Optional[np.ndarray] = None     # (T,) bool，True=该帧命中问题；无逐帧概念时为 None
    metrics: dict = field(default_factory=dict) # 数值指标，如 {"max_jerk": ..., "a_dot_b_max": ...}
    flags: list[str] = field(default_factory=list)  # 命中的问题标签，如 ["nan", "teleport"]

    def hard_fail(self) -> bool:
        return self.severity == "hard_fail" or not self.passed

    @classmethod
    def hard(cls, name: str, ok: bool, *, metrics: Optional[dict] = None,
             flags: Optional[list] = None) -> "CheckResult":
        """构造 hard-validity 结果：ok→info/passed，否则→hard_fail。"""
        return cls(
            name=name,
            passed=ok,
            severity="info" if ok else "hard_fail",
            metrics=metrics or {},
            flags=flags or [],
        )
