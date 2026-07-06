"""processed 质量检查的单测（schema/finite/modality/rot6d/gripper/attrs）。"""

from __future__ import annotations

import numpy as np

from data_filter.checks.attrs import check_attrs
from data_filter.checks.gripper import check_gripper
from data_filter.checks.modality import check_modality_lengths
from data_filter.checks.motion import check_motion_quality
from data_filter.checks.rot6d import check_rot6d
from data_filter.checks.validity import check_finite, check_schema_shape
from data_filter.io import schema

from ._fixtures import DEFAULT_ATTRS, valid_qpos, valid_rot6d

CFG = {"tol": 1e-3, "binary_tol": 1e-3}


# ------------------------- schema / finite -------------------------
def test_schema_shape():
    assert check_schema_shape(valid_qpos(8), CFG).passed
    r = check_schema_shape(np.zeros((8, 14), np.float32), CFG)
    assert r.hard_fail() and "bad_shape" in r.flags


def test_finite():
    q = valid_qpos(8)
    assert check_finite({"qpos": q}, CFG).passed
    q[3, 5] = np.nan
    r = check_finite({"qpos": q}, CFG)
    assert r.hard_fail() and any("qpos" in f for f in r.flags)


# ------------------------- modality -------------------------
def test_modality_lengths():
    assert check_modality_lengths({"qpos": 8, "ts": 8, "cam": 8}, CFG).passed
    r = check_modality_lengths({"qpos": 8, "ts": 8, "cam": 7}, CFG)
    assert r.hard_fail() and "length_mismatch" in r.flags


# ------------------------- rot6d -------------------------
def test_rot6d_valid():
    assert check_rot6d(valid_rot6d(8), CFG).passed


def test_rot6d_non_orthogonal():
    bad = np.tile([1, 0, 0, 1, 0, 0], (8, 1)).astype(np.float32)  # a=b → a·b=1
    r = check_rot6d(bad, CFG)
    assert r.hard_fail() and "orthogonality" in r.flags


def test_rot6d_non_unit():
    bad = np.tile([2, 0, 0, 0, 1, 0], (8, 1)).astype(np.float32)  # ‖a‖=2
    r = check_rot6d(bad, CFG)
    assert r.hard_fail() and "norm_a" in r.flags


def test_rot6d_nan():
    bad = valid_rot6d(8)
    bad[2, 0] = np.nan
    assert check_rot6d(bad, CFG).hard_fail()


# ------------------------- gripper -------------------------
def test_gripper_binary_ok():
    g = np.tile([[0.0, 1.0]], (8, 1)).astype(np.float32)
    assert check_gripper(g, {}, CFG).passed


def test_gripper_not_binary():
    g = np.full((8, 2), 0.5, np.float32)  # 连续值
    r = check_gripper(g, {}, CFG)
    assert r.hard_fail() and "not_binary" in r.flags


def test_gripper_out_of_range():
    g = np.full((8, 2), 2.0, np.float32)
    r = check_gripper(g, {}, CFG)
    assert r.hard_fail() and "out_of_range" in r.flags


# ------------------------- attrs -------------------------
def test_attrs_pika_complete():
    assert check_attrs(dict(DEFAULT_ATTRS["pika_umi"]), "pika_umi", CFG).passed


def test_attrs_pika_missing():
    a = dict(DEFAULT_ATTRS["pika_umi"])
    a.pop("domain_name")
    r = check_attrs(a, "pika_umi", CFG)
    assert r.hard_fail() and any("missing:domain_name" in f for f in r.flags)


def test_attrs_pika_wrong_domain():
    a = dict(DEFAULT_ATTRS["pika_umi"])
    a["domain_name"] = "not_a_valid_domain"
    r = check_attrs(a, "pika_umi", CFG)
    assert r.hard_fail() and any("wrong:domain_name" in f for f in r.flags)


def test_attrs_nas_complete():
    assert check_attrs(dict(DEFAULT_ATTRS["nas_teleop"]), "nas_teleop", CFG).passed


# ------------------------- motion quality -------------------------
def test_motion_quality_ok_on_regular_motion():
    q = valid_qpos(80)
    r = check_motion_quality(q, {"static_min_frames": 45, "min_gripper_changes": 1})
    assert r.passed
    assert not r.flags


def test_motion_quality_flags_static_and_low_gripper_coverage():
    q = valid_qpos(80)
    q[:, :3] = q[0, :3]
    q[:, 10:13] = q[0, 10:13]
    q[:, [schema.LEFT_GRIP, schema.RIGHT_GRIP]] = 1.0
    r = check_motion_quality(q, {"static_min_frames": 10, "min_gripper_changes": 1})
    assert r.passed
    assert "long_static" in r.flags
    assert "low_gripper_coverage" in r.flags
