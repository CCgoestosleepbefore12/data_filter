"""质量打分与决策分级。

聚合各 quality check 的 metrics/flags → episode 质量分数与决策标签。
hard-validity 失败直接 drop/block（不进打分）。

决策标签（见 spec.md §决策与输出）：
    keep_high_quality | keep_with_downweight | review | drop

第一版建议用**透明规则分级**（任一 hard fail → drop；≥N 项 quality flag → review；
否则 keep），不塌成手调权重标量——待有校准数据后再引入加权分。

TODO(milestone 6): 实现规则分级 + source-level 分布 + 阈值建议。
"""

from __future__ import annotations

from .checks.base import CheckResult


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
    n_flags = sum(len(r.flags) for r in quality)
    decision = cfg.get("decision", {})
    review_at = int(decision.get("review_when_quality_flags_ge", 2))

    if n_flags >= review_at:
        return {"label": "review", "reasons": reasons}
    if n_flags > 0:
        return {"label": "keep_with_downweight", "reasons": reasons}
    return {"label": "keep_high_quality", "reasons": []}
