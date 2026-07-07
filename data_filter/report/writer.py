"""报告产出：per-episode JSON + 可读 markdown + split lists。

产物（见 spec.md §决策与输出）：
- processed 阶段: processed_validity_report.json/md、episode_scores.jsonl、
  keep/downweight/review/drop lists、sampling_weights.json。
"""

from __future__ import annotations

import json
import os


def write_report(report: dict, out_dir: str, prefix: str = "processed_validity") -> dict:
    """report: run_processed_gate 的返回。写 json/md/drop_list，返回写出的路径。"""
    os.makedirs(out_dir, exist_ok=True)
    summary = report.get("summary", {})
    episodes = report.get("episodes", [])

    json_path = os.path.join(out_dir, f"{prefix}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    drops = [e["path"] for e in episodes if e.get("label") == "drop"]
    drop_path = os.path.join(out_dir, f"{prefix}_drop_list.txt")
    with open(drop_path, "w", encoding="utf-8") as f:
        f.write("\n".join(drops) + ("\n" if drops else ""))

    list_paths = {"drop_list": drop_path}
    label_to_file = {
        "keep_high_quality": ("keep_high_quality_list", f"{prefix}_keep_high_quality_list.txt"),
        "keep_with_downweight": ("downweight_list", f"{prefix}_downweight_list.txt"),
        "review": ("review_list", f"{prefix}_review_list.txt"),
    }
    for label, (key, filename) in label_to_file.items():
        path = os.path.join(out_dir, filename)
        rows = [e["path"] for e in episodes if e.get("label") == label]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + ("\n" if rows else ""))
        list_paths[key] = path

    scores_path = os.path.join(out_dir, f"{prefix}_episode_scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(_score_row(e), ensure_ascii=False) + "\n")

    weights = {
        e["path"]: _sampling_weight(e.get("label", "drop"))
        for e in episodes
        if e.get("label") in {"keep_high_quality", "keep_with_downweight"}
    }
    weights_path = os.path.join(out_dir, f"{prefix}_sampling_weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(out_dir, f"{prefix}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_md(summary, episodes, title=_title_from_prefix(prefix)))

    return {
        "json": json_path,
        "md": md_path,
        "scores": scores_path,
        "sampling_weights": weights_path,
        **list_paths,
    }


def _sampling_weight(label: str) -> float:
    if label == "keep_high_quality":
        return 1.0
    if label == "keep_with_downweight":
        return 0.5
    if label == "review":
        return 0.0
    return 0.0


def _score_row(e: dict) -> dict:
    return {
        "path": e.get("path"),
        "source_kind": e.get("source_kind"),
        "label": e.get("label"),
        "reasons": e.get("reasons", []),
    }


def _title_from_prefix(prefix: str) -> str:
    if prefix.startswith("raw"):
        return "Raw quality report"
    if prefix.startswith("processed"):
        return "Processed validity report"
    return f"{prefix} report"


def _render_md(summary: dict, episodes: list, title: str = "Processed validity report") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- 总数: {summary.get('total', len(episodes))}")
    for label, n in sorted(summary.get("by_label", {}).items()):
        lines.append(f"- {label}: {n}")
    lines += ["", "| episode | source | label | 命中 |", "|---|---|---|---|"]
    for e in episodes:
        reasons = "; ".join(
            f"{r['check']}({','.join(r['flags'])})" for r in e.get("reasons", [])
        )
        lines.append(
            f"| {os.path.basename(e['path'])} | {e.get('source_kind', '-')} "
            f"| {e.get('label', '-')} | {reasons or '-'} |"
        )
    return "\n".join(lines) + "\n"
