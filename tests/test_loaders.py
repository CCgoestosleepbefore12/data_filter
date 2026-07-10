"""load_processed_xvla 的单测。"""

from __future__ import annotations

import h5py

from data_filter.io.loaders import load_processed_xvla, load_raw_pika, load_raw_teleop, sample_image_frames

from ._fixtures import make_processed_hdf5, make_raw_pika_hdf5, make_raw_teleop_hdf5


def test_load_processed_good_pika(tmp_path):
    p = make_processed_hdf5(tmp_path / "ep.hdf5", T=8, source="pika_umi")
    ep = load_processed_xvla(p)

    assert ep.length == 8
    assert ep.qpos.shape == (8, 20)
    assert ep.gripper.shape == (8, 2)          # [LEFT_GRIP, RIGHT_GRIP]
    assert ep.timestamps.shape == (8,)
    assert ep.source_kind == "pika_umi"
    assert ep.attrs["tip2base_applied"] is True
    assert ep.attrs["domain_name"] == "pika_umi_tip2base_abs"
    assert len(ep.image_keys) == 3
    assert all(v == 8 for v in ep.image_lengths.values())


def test_load_processed_infers_nas_source(tmp_path):
    p = make_processed_hdf5(tmp_path / "ep.hdf5", T=6, source="nas_teleop")
    ep = load_processed_xvla(p)

    assert ep.source_kind == "nas_teleop"
    assert ep.attrs["domain_name"] == "nas_real_teleop"
    assert ep.attrs["source_key"] == "observations/eef_6d"


def test_load_processed_chunked_image_group_uses_index_length(tmp_path):
    p = make_processed_hdf5(
        tmp_path / "ep.hdf5",
        T=8,
        source="pika_umi",
        n_img_frames=7,
        image_layout="chunked_index",
    )
    ep = load_processed_xvla(p)

    assert len(ep.image_keys) == 3
    assert all(v == 7 for v in ep.image_lengths.values())
    assert all(not k.endswith("_index") for k in ep.image_keys)

    with h5py.File(p, "r") as h:
        frames = sample_image_frames(h, "observations/images/cam_high", 4, window_frames=3, num_windows=1)
    assert len(frames) >= 4
    assert all(frame and frame.startswith(b"\xff\xd8") for frame in frames)


def test_load_raw_pika(tmp_path):
    p = make_raw_pika_hdf5(tmp_path / "raw_pika.hdf5", T=10)
    ep = load_raw_pika(p)

    assert ep.source_kind == "pika"
    assert ep.length == 10
    assert ep.pose.shape == (10, 12)
    assert ep.gripper.shape == (10, 2)
    assert ep.timestamps.shape == (10,)
    assert all(v == 10 for v in ep.image_lengths.values())


def test_load_raw_teleop(tmp_path):
    p = make_raw_teleop_hdf5(tmp_path / "raw_teleop.hdf5", T=11)
    ep = load_raw_teleop(p)

    assert ep.source_kind == "teleop"
    assert ep.length == 11
    assert ep.qpos.shape == (11, 14)
    assert ep.action.shape == (11, 14)
    assert ep.timestamps.shape == (11,)
    assert ep.extra_timestamps["eef_right_time"].shape == (11,)
