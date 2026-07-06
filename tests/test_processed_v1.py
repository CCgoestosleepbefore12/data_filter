"""步骤 1：processed v1 硬检查补齐——timestamp / 递归扫描 / 配置接入。"""

from __future__ import annotations

import numpy as np

from data_filter.checks.timestamp import check_timestamp
from data_filter.config import load_config
from data_filter.pipeline import run_processed_gate

from ._fixtures import make_processed_hdf5, valid_qpos


# ------------------------- timestamp -------------------------
def test_timestamp_good():
    ts = np.arange(20) / 30.0
    assert check_timestamp(ts, {"max_dt_ratio": 3.0}).passed


def test_timestamp_non_monotonic():
    ts = np.arange(20, dtype=float) / 30.0
    ts[10] = ts[9] - 0.01  # 回退
    r = check_timestamp(ts, {"max_dt_ratio": 3.0})
    assert r.hard_fail() and "non_monotonic" in r.flags


def test_timestamp_dt_jump():
    ts = np.arange(20, dtype=float) / 30.0
    ts[10:] += 1.0  # 制造一个 ~30x 中位 dt 的大跳变
    r = check_timestamp(ts, {"max_dt_ratio": 3.0})
    assert r.passed and not r.hard_fail()
    assert r.severity == "warn" and "dt_jump" in r.flags


# ------------------------- 递归扫描 -------------------------
def test_recursive_scan(tmp_path):
    (tmp_path / "sub").mkdir()
    make_processed_hdf5(tmp_path / "a.hdf5", T=6, source="pika_umi")
    make_processed_hdf5(tmp_path / "sub" / "b.hdf5", T=6, source="pika_umi")

    report = run_processed_gate(str(tmp_path))
    assert report["summary"]["total"] == 2
    names = {e["path"].split("/")[-1] for e in report["episodes"]}
    assert names == {"a.hdf5", "b.hdf5"}


# ------------------------- 配置接入 -------------------------
def test_config_load():
    cfg = load_config("processed_xvla")
    assert "thresholds" in cfg and "rot6d" in cfg["thresholds"]
    assert "hard_checks" in cfg


def test_gate_honors_disabled_check(tmp_path):
    # 造一个 attrs 缺失的 episode
    make_processed_hdf5(tmp_path / "ep.hdf5", T=6, source="pika_umi", drop_attr="domain_name")

    # 默认：attrs 检查开 → drop
    on = run_processed_gate(str(tmp_path))
    assert on["episodes"][0]["label"] == "drop"

    # 关掉 attrs 检查 → 不因 attrs 被 drop
    cfg = {"hard_checks": {"attrs": False}}
    off = run_processed_gate(str(tmp_path), cfg)
    assert off["episodes"][0]["label"] == "keep_high_quality"


def test_timestamp_wired_into_gate(tmp_path):
    bad_ts = np.arange(8, dtype=float) / 30.0
    bad_ts[4] = bad_ts[3] - 0.1  # 非单调
    make_processed_hdf5(tmp_path / "ep.hdf5", qpos=valid_qpos(8), source="pika_umi", timestamps=bad_ts)

    report = run_processed_gate(str(tmp_path))
    ep = report["episodes"][0]
    assert ep["label"] == "drop"
    assert any(r["check"] == "timestamp" and "non_monotonic" in r["flags"] for r in ep["reasons"])
