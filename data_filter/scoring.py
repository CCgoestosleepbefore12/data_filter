"""质量打分与决策分级。

聚合各 quality check 的 metrics/flags → episode 质量分数与决策标签。
hard-validity 失败直接 drop/block（不进打分）。

决策标签（见 spec.md §决策与输出）：
    keep_high_quality | keep_with_downweight | review | drop

用**透明规则分级**（任一 hard fail → drop；命中 ≥N 个 quality 检查或 REVIEW_FLAGS
→ review；否则 keep），不塌成手调权重标量——待有校准数据后再引入加权分。
"""

from __future__ import annotations

from .checks.base import CheckResult

REVIEW_FLAGS = {
    "left_arm_frozen",
    "right_arm_frozen",
}


def score_episode(results: list[CheckResult], cfg: dict | None = None) -> dict:
    """results: 一个 episode 的所有 CheckResult。返回 {label, reasons}。

    任一 hard_fail → drop；quality flags 按透明规则分到 review/downweight。
    """
    cfg = cfg or {}
    hard = [r for r in results if r.hard_fail()]
    if hard:
        return {
            "label": "drop",
            "reasons": [{"check": r.name, "flags": r.flags} for r in hard],
        }

    quality = [r for r in results if r.flags and not r.hard_fail()]
    reasons = [{"check": r.name, "flags": r.flags} for r in quality]
    n_checks = len({r.name for r in quality})
    flag_set = {flag for r in quality for flag in r.flags}
    decision = cfg.get("decision", {})
    review_at = int(decision.get("review_when_quality_flags_ge", 2))

    if flag_set & REVIEW_FLAGS or n_checks >= review_at:
        return {"label": "review", "reasons": reasons}
    if n_checks > 0:
        return {"label": "keep_with_downweight", "reasons": reasons}
    return {"label": "keep_high_quality", "reasons": []}
