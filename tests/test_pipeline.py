"""score_episode + run_processed_gate 端到端单测。"""

from __future__ import annotations

import json
import sys

import pytest
import numpy as np

import data_filter.pipeline as pipeline_mod
from data_filter.checks.base import CheckResult
from data_filter.io import schema
from data_filter.pipeline import main, run_processed_gate
from data_filter.pipeline import run_raw_gate
from data_filter.report.writer import write_report
from data_filter.scoring import score_episode

from ._fixtures import make_processed_hdf5, make_raw_pika_hdf5, make_raw_teleop_hdf5, valid_qpos


# ------------------------- score_episode -------------------------
def test_score_all_pass_keeps():
    ok = [CheckResult("a", passed=True), CheckResult("b", passed=True)]
    assert score_episode(ok)["label"] == "keep_high_quality"


def test_score_hard_fail_drops_with_reason():
    res = [
        CheckResult("rot6d_left", passed=True),
        CheckResult("attrs", passed=False, severity="hard_fail", flags=["missing:domain_name"]),
    ]
    out = score_episode(res)
    assert out["label"] == "drop"
    assert out["reasons"][0]["check"] == "attrs"
    assert "missing:domain_name" in out["reasons"][0]["flags"]


def test_score_quality_flags_downweights_or_reviews():
    one = [CheckResult("motion", passed=True, severity="warn", flags=["long_static"])]
    assert score_episode(one, {"decision": {"review_when_quality_flags_ge": 2}})["label"] == "keep_with_downweight"

    two = [CheckResult("motion", passed=True, severity="warn", flags=["long_static", "low_gripper_coverage"])]
    assert score_episode(two, {"decision": {"review_when_quality_flags_ge": 2}})["label"] == "review"

    two_checks = [
        CheckResult("motion", passed=True, severity="warn", flags=["long_static"]),
        CheckResult("timestamp", passed=True, severity="warn", flags=["dt_jump"]),
    ]
    assert score_episode(two_checks, {"decision": {"review_when_quality_flags_ge": 2}})["label"] == "review"

    frozen = [CheckResult("arm_activity", passed=True, severity="warn", flags=["right_arm_frozen"])]
    assert score_episode(frozen, {"decision": {"review_when_quality_flags_ge": 2}})["label"] == "review"


# ------------------------- 端到端 -------------------------
def test_processed_gate_end_to_end(tmp_path):
    # 1 个好 episode
    make_processed_hdf5(tmp_path / "good.hdf5", T=8, source="pika_umi")

    # 1 个坏 episode：把左臂 rot6d 破坏成非正交
    bad_qpos = valid_qpos(8)
    bad_qpos[:, schema.LEFT_ROT6D] = np.tile([1, 0, 0, 1, 0, 0], (8, 1))  # a=b
    make_processed_hdf5(tmp_path / "bad.hdf5", qpos=bad_qpos, source="pika_umi")

    report = run_processed_gate(str(tmp_path))

    assert report["summary"]["total"] == 2
    assert report["summary"]["by_label"].get("keep_high_quality") == 1
    assert report["summary"]["by_label"].get("drop") == 1

    labels = {e["path"].split("/")[-1]: e["label"] for e in report["episodes"]}
    assert labels["good.hdf5"] == "keep_high_quality"
    assert labels["bad.hdf5"] == "drop"

    # 坏 episode 的 drop 原因应含 rot6d_left 正交性
    bad = next(e for e in report["episodes"] if e["path"].endswith("bad.hdf5"))
    assert any(r["check"] == "rot6d_left" and "orthogonality" in r["flags"] for r in bad["reasons"])


def test_write_report_outputs(tmp_path):
    make_processed_hdf5(tmp_path / "good.hdf5", T=6, source="nas_teleop")
    bad_qpos = valid_qpos(6)
    bad_qpos[:, schema.LEFT_ROT6D] = np.tile([1, 0, 0, 1, 0, 0], (6, 1))
    make_processed_hdf5(tmp_path / "bad.hdf5", qpos=bad_qpos, source="nas_teleop")

    report = run_processed_gate(str(tmp_path))
    out = write_report(report, str(tmp_path / "out"))

    # json 可解析且内容一致
    loaded = json.load(open(out["json"], encoding="utf-8"))
    assert loaded["summary"]["total"] == 2

    # drop_list 含且仅含坏 episode
    drops = open(out["drop_list"], encoding="utf-8").read().split()
    assert any(d.endswith("bad.hdf5") for d in drops)
    assert not any(d.endswith("good.hdf5") for d in drops)

    # v1 输出 score / split / sampling weight
    assert "scores" in out and "sampling_weights" in out
    assert (tmp_path / "out" / "processed_validity_keep_high_quality_list.txt").exists()
    assert (tmp_path / "out" / "processed_validity_review_list.txt").exists()
    assert (tmp_path / "out" / "processed_validity_downweight_list.txt").exists()
    md = open(out["md"], encoding="utf-8").read()
    assert "## Top Reasons" in md
    assert "## Top Check Flags" in md
    assert "| episode | source | label | 命中 | 证据 |" in md
    assert "rot6d_left(orthogonality)" in md


def test_raw_gate_pika_end_to_end(tmp_path):
    make_raw_pika_hdf5(tmp_path / "raw.hdf5", T=20)
    report = run_raw_gate(str(tmp_path), "pika", {"checks": {"tracking": True, "spike": True}})
    assert report["summary"]["total"] == 1
    assert report["episodes"][0]["label"] in {"keep_high_quality", "keep_with_downweight", "review"}
    names = {c["name"] for c in report["episodes"][0]["checks"]}
    assert {"finite", "modality", "timestamp", "tracking", "spike"}.issubset(names)


def test_raw_gate_teleop_end_to_end(tmp_path):
    make_raw_teleop_hdf5(tmp_path / "raw.hdf5", T=20)
    report = run_raw_gate(
        str(tmp_path), "teleop", {"checks": {"tracking": False, "spike": True, "arm_activity": True}}
    )
    assert report["summary"]["total"] == 1
    names = {c["name"] for c in report["episodes"][0]["checks"]}
    assert {"finite", "modality", "timestamp", "spike", "arm_activity"}.issubset(names)


def test_raw_gate_teleop_checks_right_arm_time_axis(tmp_path):
    right_time = np.arange(20, dtype=np.float32) / 30.0
    right_time[10:] += 0.2
    make_raw_teleop_hdf5(tmp_path / "raw.hdf5", T=20, right_time=right_time)

    report = run_raw_gate(str(tmp_path), "teleop", {"checks": {"timestamp": True, "spike": False}})
    ep = report["episodes"][0]
    names = {c["name"] for c in ep["checks"]}
    assert "eef_right_time" in names
    assert any(c["name"] == "timestamp_skew" and "clock_skew" in c["flags"] for c in ep["checks"])


def test_raw_gate_teleop_requires_right_arm_time_axis(tmp_path):
    make_raw_teleop_hdf5(tmp_path / "raw.hdf5", T=20, with_right_time=False)

    report = run_raw_gate(str(tmp_path), "teleop", {"checks": {"timestamp": True, "spike": False}})
    ep = report["episodes"][0]
    assert ep["label"] == "drop"
    assert any(r["check"] == "eef_right_time" and "missing" in r["flags"] for r in ep["reasons"])


def test_raw_gate_teleop_reviews_frozen_right_arm_regression(tmp_path):
    make_raw_teleop_hdf5(tmp_path / "episode_2030.hdf5", T=20, freeze_arm="right")

    report = run_raw_gate(
        str(tmp_path),
        "teleop",
        {"checks": {"timestamp": True, "spike": False, "arm_activity": True}},
    )
    ep = report["episodes"][0]
    assert ep["label"] == "review"
    assert any(r["check"] == "arm_activity" and "right_arm_frozen" in r["flags"] for r in ep["reasons"])


def test_processed_gate_keeps_running_after_check_exception_shape_edge(tmp_path):
    make_processed_hdf5(tmp_path / "empty.hdf5", T=0, source="pika_umi")
    make_processed_hdf5(tmp_path / "good.hdf5", T=6, source="pika_umi")

    report = run_processed_gate(str(tmp_path))
    labels = {e["path"].split("/")[-1]: e["label"] for e in report["episodes"]}
    assert labels["empty.hdf5"] == "drop"
    assert labels["good.hdf5"] == "keep_high_quality"
    empty = next(e for e in report["episodes"] if e["path"].endswith("empty.hdf5"))
    assert any(r["check"] == "schema_shape" and "too_short" in r["flags"] for r in empty["reasons"])
    assert not any(r["check"] == "check_exception" for r in empty["reasons"])


def test_processed_gate_nonnumeric_timestamp_keeps_specific_reason(tmp_path):
    make_processed_hdf5(tmp_path / "bad_ts.hdf5", T=6, source="pika_umi", timestamps=np.array(["bad"] * 6))

    report = run_processed_gate(str(tmp_path), {"hard_checks": {"timestamp": True}})
    ep = report["episodes"][0]
    assert ep["label"] == "drop"
    assert any(r["check"] == "finite" and "nonnumeric:timestamps" in r["flags"] for r in ep["reasons"])
    assert not any(r["check"] == "check_exception" for r in ep["reasons"])


def test_processed_gate_check_exception_reviews_not_drops(tmp_path, monkeypatch):
    make_processed_hdf5(tmp_path / "ep.hdf5", T=6, source="pika_umi")

    def boom(_ep, _cfg):
        raise RuntimeError("synthetic check bug")

    monkeypatch.setattr(pipeline_mod, "_run_processed_checks", boom)
    report = run_processed_gate(str(tmp_path), {})
    ep = report["episodes"][0]
    assert ep["label"] == "review"
    assert any(r["check"] == "check_exception" for r in ep["reasons"])


def test_processed_gate_video_bad_frame_is_quality_flag_not_check_exception(tmp_path):
    make_processed_hdf5(tmp_path / "ep.hdf5", T=6, source="pika_umi", image_layout="chunked_index")

    report = run_processed_gate(
        str(tmp_path),
        {
            "quality_checks": {"video_quality": True},
            "thresholds": {"video_quality": {"sample_frames": 4, "blur_var": -1.0, "max_decode_failure_ratio": 0.0}},
        },
    )
    ep = report["episodes"][0]
    assert not any(r["check"] == "check_exception" for r in ep["reasons"])
    assert ep["label"] == "keep_high_quality"


def test_processed_gate_video_static_uses_continuous_window(tmp_path):
    make_processed_hdf5(tmp_path / "static.hdf5", T=50, source="pika_umi", image_layout="vlen")

    report = run_processed_gate(
        str(tmp_path),
        {
            "quality_checks": {"video_quality": True},
            "thresholds": {
                "video_quality": {
                    "sample_frames": 4,
                    "black_luma": -1.0,
                    "blur_var": -1.0,
                    "enable_static": True,
                    "static_min_frames": 45,
                    "static_window_frames": 45,
                    "static_sample_windows": 1,
                    "static_diff_eps": 1.0,
                }
            },
        },
    )
    ep = report["episodes"][0]
    assert any(
        check["name"] == "video_quality" and "cam_high_static" in check["flags"]
        for check in ep["checks"]
    )


def test_cli_empty_directory_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["data-filter", "--gate", "processed", "--root", str(tmp_path), "--out", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit, match="未在 data_roots 中找到"):
        main()


def test_cli_rejects_mismatched_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["data-filter", "--gate", "processed", "--config", "raw_teleop", "--root", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="processed gate 只能使用"):
        main()


def test_cli_rejects_source_for_processed_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["data-filter", "--gate", "processed", "--source", "pika", "--root", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="processed gate 不接受 --source"):
        main()
