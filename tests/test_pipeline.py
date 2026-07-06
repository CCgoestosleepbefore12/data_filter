"""score_episode + run_processed_gate 端到端单测。"""

from __future__ import annotations

import json

import numpy as np

from data_filter.checks.base import CheckResult
from data_filter.io import schema
from data_filter.pipeline import run_processed_gate
from data_filter.report.writer import write_report
from data_filter.scoring import score_episode

from ._fixtures import make_processed_hdf5, valid_qpos


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
    assert (tmp_path / "out" / "keep_high_quality_list.txt").exists()
    assert (tmp_path / "out" / "review_list.txt").exists()
    assert (tmp_path / "out" / "downweight_list.txt").exists()
