"""编排：遍历 episode → 跑 checks → 决策 → 写 report/drop-list。

两道闸门：
- Raw quality gate（待 milestone 3）：raw pika/teleop → 明显坏采集 drop / review。
- Processed XVLA quality gate：hard-validity → 通过后（quality scoring 待补）。

不重写 HDF5：只输出 report + drop-list。
"""

from __future__ import annotations

import glob
import os

from .checks.attrs import check_attrs
from .checks.base import CheckResult
from .checks.gripper import check_gripper
from .checks.modality import check_modality_lengths
from .checks.rot6d import check_rot6d
from .checks.validity import check_finite, check_schema_shape
from .io import schema
from .io.loaders import EpisodeSignals, load_processed_xvla
from .report.writer import write_report
from .scoring import score_episode


def _run_processed_checks(ep: EpisodeSignals, cfg: dict) -> list[CheckResult]:
    """对一个 processed episode 跑 v1 hard-validity 检查集。"""
    thr = cfg.get("thresholds", {})
    rot_cfg = thr.get("rot6d", {"tol": 1e-3})
    grip_cfg = thr.get("gripper", {"binary_tol": 1e-3})

    results = [
        check_schema_shape(ep.qpos, {}),
        check_finite({"qpos": ep.qpos, "timestamps": ep.timestamps}, {}),
    ]

    lengths = {"qpos": ep.length}
    if ep.timestamps is not None:
        lengths["timestamps"] = int(len(ep.timestamps))
    lengths.update(ep.image_lengths)
    results.append(check_modality_lengths(lengths, {}))

    # 形状合法才切 rot6d/gripper
    if ep.qpos.ndim == 2 and ep.qpos.shape[1] >= schema.QPOS_DIM:
        results.append(check_rot6d(ep.qpos[:, schema.LEFT_ROT6D], rot_cfg, name="rot6d_left"))
        results.append(check_rot6d(ep.qpos[:, schema.RIGHT_ROT6D], rot_cfg, name="rot6d_right"))
        if ep.gripper is not None:
            results.append(check_gripper(ep.gripper, ep.attrs, grip_cfg))

    results.append(check_attrs(ep.attrs, ep.source_kind, {}))
    return results


def run_processed_gate(root: str, cfg: dict | None = None) -> dict:
    """遍历 root 下的 *.hdf5，跑 processed 质量闸门，返回结构化报告（不写盘）。"""
    cfg = cfg or {}
    files = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    episodes: list[dict] = []

    for path in files:
        try:
            ep = load_processed_xvla(path)
        except Exception as e:  # 读失败也算 drop，附原因
            episodes.append({
                "path": path, "source_kind": "unknown", "label": "drop",
                "reasons": [{"check": "load", "flags": [f"{type(e).__name__}: {e}"]}],
                "checks": [],
            })
            continue

        results = _run_processed_checks(ep, cfg)
        score = score_episode(results, cfg)
        episodes.append({
            "path": path,
            "source_kind": ep.source_kind,
            "label": score["label"],
            "reasons": score["reasons"],
            "checks": [
                {"name": r.name, "passed": r.passed, "severity": r.severity,
                 "flags": r.flags, "metrics": r.metrics}
                for r in results
            ],
        })

    by_label: dict[str, int] = {}
    for e in episodes:
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
    summary = {"total": len(episodes), "by_label": by_label}
    return {"summary": summary, "episodes": episodes}


def main() -> None:
    """CLI 入口（见 scripts/run_filter.py）。"""
    import argparse

    ap = argparse.ArgumentParser(description="data_filter 质量闸门")
    ap.add_argument("--gate", required=True, choices=["processed", "raw"])
    ap.add_argument("--root", required=True, help="待检查的 HDF5 根目录")
    ap.add_argument("--out", default="outputs", help="报告输出目录")
    args = ap.parse_args()

    if args.gate != "processed":
        raise SystemExit("raw gate 待 milestone 3 实现")

    report = run_processed_gate(args.root)
    paths = write_report(report, args.out)
    print(f"total={report['summary']['total']} by_label={report['summary']['by_label']}")
    print(f"report: {paths['json']}")


if __name__ == "__main__":
    main()
