"""报告产出：per-episode JSON + 可读 markdown + drop-list。

产物（见 spec.md §决策与输出）：
- processed 阶段: processed_validity_report.json/md、drop_list.txt。
  （episode_scores.jsonl / downweight / sampling_weights 待 quality 检查接入后补。）
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
    drop_path = os.path.join(out_dir, "drop_list.txt")
    with open(drop_path, "w", encoding="utf-8") as f:
        f.write("\n".join(drops) + ("\n" if drops else ""))

    md_path = os.path.join(out_dir, f"{prefix}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_md(summary, episodes))

    return {"json": json_path, "md": md_path, "drop_list": drop_path}


def _render_md(summary: dict, episodes: list) -> str:
    lines = ["# Processed validity report", ""]
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
