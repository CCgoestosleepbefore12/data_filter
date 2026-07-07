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
from .checks.motion import check_motion_quality
from .checks.raw_activity import check_bimanual_activity
from .checks.rot6d import check_rot6d
from .checks.spike import check_spike
from .checks.timestamp import check_clock_skew, check_timestamp
from .checks.tracking import check_tracking
from .checks.validity import check_finite, check_min_length, check_required_keys, check_schema_shape
from .config import load_config
from .io import schema
from .io.loaders import EpisodeSignals, load_processed_xvla, load_raw_pika, load_raw_teleop
from .report.writer import write_report
from .scoring import score_episode


def _enabled(cfg: dict, key: str) -> bool:
    """检查是否启用（cfg["hard_checks"][key]，缺省 True）。"""
    return cfg.get("hard_checks", {}).get(key, True)


def _quality_enabled(cfg: dict, key: str) -> bool:
    """quality 检查是否启用（cfg["quality_checks"][key]，缺省 False）。"""
    return cfg.get("quality_checks", {}).get(key, False)


def _raw_enabled(cfg: dict, key: str) -> bool:
    """raw 检查是否启用（cfg["checks"][key]，缺省 True）。"""
    return cfg.get("checks", {}).get(key, True)


def _run_processed_checks(ep: EpisodeSignals, cfg: dict) -> list[CheckResult]:
    """对一个 processed episode 跑 v1 hard-validity 检查集（按 cfg 启用开关）。"""
    thr = cfg.get("thresholds", {})

    # schema/finite 恒开（切片与后续检查的地基）
    results = [
        check_schema_shape(ep.qpos, {}),
        check_finite({"qpos": ep.qpos, "timestamps": ep.timestamps}, {}),
    ]

    if _enabled(cfg, "modality"):
        lengths = {"qpos": ep.length}
        if ep.timestamps is not None:
            lengths["timestamps"] = int(len(ep.timestamps))
        lengths.update(ep.image_lengths)
        results.append(check_modality_lengths(lengths, {}))
        results.append(check_required_keys(ep.image_keys, schema.RAW_TELEOP["images"], name="image_schema"))

    if _enabled(cfg, "timestamp"):
        if ep.timestamps is None:
            results.append(CheckResult.hard("timestamp", False, flags=["missing"]))
        else:
            results.append(check_timestamp(ep.timestamps, thr.get("timestamp", {})))

    # 形状和帧数合法才切 rot6d/gripper；结构性失败后短路，避免空数组 max/min 异常。
    if ep.qpos.ndim == 2 and ep.qpos.shape[1] >= schema.QPOS_DIM and ep.qpos.shape[0] >= 1:
        if _enabled(cfg, "rot6d"):
            rot_cfg = thr.get("rot6d", {})
            results.append(check_rot6d(ep.qpos[:, schema.LEFT_ROT6D], rot_cfg, name="rot6d_left"))
            results.append(check_rot6d(ep.qpos[:, schema.RIGHT_ROT6D], rot_cfg, name="rot6d_right"))
        if _enabled(cfg, "gripper") and ep.gripper is not None:
            results.append(check_gripper(ep.gripper, ep.attrs, thr.get("gripper", {})))
        if _quality_enabled(cfg, "motion"):
            results.append(check_motion_quality(ep.qpos, thr.get("motion", {})))

    if _enabled(cfg, "attrs"):
        results.append(check_attrs(ep.attrs, ep.source_kind, {}))
    return results


def _summarize(episodes: list[dict]) -> dict:
    by_label: dict[str, int] = {}
    for e in episodes:
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
    return {"total": len(episodes), "by_label": by_label}


def _episode_record(path: str, source_kind: str, results: list[CheckResult], cfg: dict) -> dict:
    score = score_episode(results, cfg)
    return {
        "path": path,
        "source_kind": source_kind,
        "label": score["label"],
        "reasons": score["reasons"],
        "checks": [
            {"name": r.name, "passed": r.passed, "severity": r.severity,
             "flags": r.flags, "metrics": r.metrics}
            for r in results
        ],
    }


def _raw_lengths(ep: EpisodeSignals) -> dict[str, int]:
    lengths = {}
    if ep.pose is not None:
        lengths["pose"] = int(ep.pose.shape[0])
    if ep.qpos is not None:
        lengths["qpos"] = int(ep.qpos.shape[0])
    if ep.action is not None:
        lengths["action"] = int(ep.action.shape[0])
    if ep.gripper is not None:
        lengths["gripper"] = int(ep.gripper.shape[0])
    if ep.timestamps is not None:
        lengths["timestamps"] = int(len(ep.timestamps))
    for name, ts in ep.extra_timestamps.items():
        lengths[name] = int(len(ts))
    lengths.update(ep.image_lengths)
    return lengths


def _run_raw_checks(ep: EpisodeSignals, cfg: dict) -> list[CheckResult]:
    thr = cfg.get("thresholds", {})
    results: list[CheckResult] = []
    arrays = {}
    if ep.pose is not None:
        arrays["pose"] = ep.pose
    if ep.qpos is not None:
        arrays["qpos"] = ep.qpos
    if ep.action is not None:
        arrays["action"] = ep.action
    if ep.gripper is not None:
        arrays["gripper"] = ep.gripper
    if ep.timestamps is not None:
        arrays["timestamps"] = ep.timestamps
    arrays.update(ep.extra_timestamps)
    results.append(check_finite(arrays, {}, name="finite"))
    results.append(check_min_length(ep.length, {}))

    if _raw_enabled(cfg, "modality"):
        results.append(check_modality_lengths(_raw_lengths(ep), {}))
        expected_images = schema.RAW_PIKA["images"] if ep.source_kind == "pika" else schema.RAW_TELEOP["images"]
        results.append(check_required_keys(ep.image_keys, expected_images, name="image_schema"))
    if _raw_enabled(cfg, "timestamp"):
        timestamp_cfg = thr.get("timestamp", {})
        if ep.timestamps is None:
            results.append(CheckResult.hard("timestamp", False, flags=["missing"]))
        else:
            results.append(check_timestamp(ep.timestamps, timestamp_cfg))
        for name, ts in ep.extra_timestamps.items():
            results.append(check_timestamp(ts, timestamp_cfg, name=name))
        if ep.source_kind == "teleop" and "eef_right_time" not in ep.extra_timestamps:
            results.append(CheckResult.hard("eef_right_time", False, flags=["missing"]))
        if ep.timestamps is not None and "eef_right_time" in ep.extra_timestamps:
            results.append(check_clock_skew(ep.timestamps, ep.extra_timestamps["eef_right_time"], timestamp_cfg))
    if _raw_enabled(cfg, "tracking") and ep.source_kind == "pika" and ep.pose is not None:
        results.append(check_tracking(ep.pose[:, :6], thr.get("tracking", {})))
        results.append(check_tracking(ep.pose[:, 6:12], thr.get("tracking", {})))
    if _raw_enabled(cfg, "spike"):
        signal = ep.pose if ep.pose is not None else ep.qpos
        if signal is not None:
            results.append(check_spike(signal, thr.get("spike", {})))
    if _raw_enabled(cfg, "arm_activity"):
        signal = ep.pose if ep.pose is not None else ep.qpos
        if signal is not None:
            results.append(check_bimanual_activity(signal, thr.get("arm_activity", {})))
    return results


def run_raw_gate(root: str, source: str, cfg: dict | None = None) -> dict:
    """递归遍历 root 下 raw *.hdf5，跑 raw 质量闸门。source: pika | teleop。"""
    cfg = cfg or {}
    root = os.path.expanduser(root)
    loader = load_raw_pika if source == "pika" else load_raw_teleop
    files = sorted(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True))
    episodes: list[dict] = []
    for path in files:
        try:
            ep = loader(path)
        except Exception as e:
            episodes.append(_exception_record(path, source, "load", e))
            continue
        try:
            episodes.append(_episode_record(path, ep.source_kind, _run_raw_checks(ep, cfg), cfg))
        except Exception as e:
            episodes.append(_exception_record(path, source, "check_exception", e))
    return {"summary": _summarize(episodes), "episodes": episodes}


def run_processed_gate(root: str, cfg: dict | None = None) -> dict:
    """递归遍历 root 下的 *.hdf5，跑 processed 质量闸门，返回结构化报告（不写盘）。"""
    cfg = cfg or {}
    root = os.path.expanduser(root)
    files = sorted(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True))
    episodes: list[dict] = []

    for path in files:
        try:
            ep = load_processed_xvla(path)
        except Exception as e:  # 读失败也算 drop，附原因
            episodes.append(_exception_record(path, "unknown", "load", e))
            continue

        try:
            episodes.append(_episode_record(path, ep.source_kind, _run_processed_checks(ep, cfg), cfg))
        except Exception as e:
            episodes.append(_exception_record(path, ep.source_kind, "check_exception", e))

    return {"summary": _summarize(episodes), "episodes": episodes}


def main() -> None:
    """CLI 入口（见 scripts/run_filter.py）。"""
    import argparse

    ap = argparse.ArgumentParser(description="data_filter 质量闸门")
    ap.add_argument("--gate", required=True, choices=["processed", "raw"])
    ap.add_argument("--config", default=None, help="configs/<name>.yaml 的 name；缺省按 gate/source 选择")
    ap.add_argument("--source", choices=["pika", "teleop"], default=None, help="raw gate 数据源")
    ap.add_argument("--root", nargs="*", default=None, help="覆盖 config 的 data_roots（可多个）")
    ap.add_argument("--out", default="outputs", help="报告输出目录")
    args = ap.parse_args()

    config_name = args.config
    if config_name is None:
        if args.gate == "processed":
            config_name = "processed_xvla"
        else:
            if args.source == "teleop":
                config_name = "raw_teleop"
            elif args.source == "pika":
                config_name = "raw_pika"
            else:
                raise SystemExit("raw gate 缺省配置需要 --source pika|teleop，或显式 --config")
    cfg = load_config(config_name)
    roots = args.root if args.root else [r for r in cfg.get("data_roots", []) if r]
    roots = [os.path.expanduser(r) for r in roots]
    if not roots:
        raise SystemExit("未提供 data_roots：用 --root 指定，或在 config 里填")
    missing_roots = [r for r in roots if not os.path.isdir(r)]
    if missing_roots:
        raise SystemExit(f"data_roots 不存在或不是目录: {missing_roots}")

    episodes: list[dict] = []
    for root in roots:
        if args.gate == "processed":
            episodes.extend(run_processed_gate(root, cfg)["episodes"])
        else:
            source = args.source or cfg.get("source_kind")
            if source not in {"pika", "teleop"}:
                raise SystemExit("raw gate 需要 --source pika|teleop，或 config.source_kind")
            episodes.extend(run_raw_gate(root, source, cfg)["episodes"])
    report = {"summary": _summarize(episodes), "episodes": episodes}

    prefix = "processed_validity" if args.gate == "processed" else "raw_quality"
    paths = write_report(report, args.out, prefix=prefix)
    print(f"roots={roots}")
    print(f"total={report['summary']['total']} by_label={report['summary']['by_label']}")
    print(f"report: {paths['json']}")


def _exception_record(path: str, source_kind: str, check: str, e: Exception) -> dict:
    return {
        "path": path,
        "source_kind": source_kind,
        "label": "drop",
        "reasons": [{"check": check, "flags": [f"{type(e).__name__}: {e}"]}],
        "checks": [],
    }


if __name__ == "__main__":
    main()
