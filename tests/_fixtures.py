"""合成 processed XVLA HDF5，供单测注入故障。

默认造一个「好」文件；通过参数注入故障（非法 rot6d、非二值夹爪、缺 attrs、
模态长度不一致、NaN 等）。只用 numpy + h5py，不依赖真数据。
"""

from __future__ import annotations

import h5py
import numpy as np

# 与 schema.PROCESSED_ATTRS 对应的合法默认 attrs
DEFAULT_ATTRS = {
    "pika_umi": {
        "pose_frame": "robot_base_tip2base_piper_tcp_config",
        "tip2base_applied": True,
        "relative_to_first_frame": False,
        "domain_name": "pika_umi_tip2base_abs",
        "time_alignment_status": "verified_common_time_axis",
    },
    "nas_teleop": {
        "domain_name": "nas_real_teleop",
        "source_key": "observations/eef_6d",
        "time_alignment_status": "verified_common_time_axis",
    },
}


def valid_rot6d(T: int) -> np.ndarray:
    """(T,6) 合法 rot6d = concat(R[:,0], R[:,1])，每帧绕 z 轴旋转 → 构造即正交。"""
    theta = np.linspace(0.0, 0.5, T)
    c, s, z = np.cos(theta), np.sin(theta), np.zeros(T)
    # R_z 列: col0=[c,s,0], col1=[-s,c,0]
    return np.stack([c, s, z, -s, c, z], axis=1).astype(np.float32)  # (T,6)


def valid_qpos(T: int = 8) -> np.ndarray:
    """(T,20) 合法 processed qpos: [L_xyz3,L_rot6,L_grip, R_xyz3,R_rot6,R_grip]。"""
    rot = valid_rot6d(T)                                             # (T,6)
    idx = np.arange(T)
    xyz_l = np.stack([0.30 + 0.01 * idx, np.full(T, 0.1), np.full(T, 0.2)], axis=1)  # (T,3)
    xyz_r = np.stack([-0.30 - 0.01 * idx, np.full(T, 0.1), np.full(T, 0.2)], axis=1)
    grip_l = (idx % 2).astype(np.float32)[:, None]                   # (T,1) 二值 0/1
    grip_r = ((idx + 1) % 2).astype(np.float32)[:, None]
    left = np.concatenate([xyz_l, rot], axis=1)                      # (T,9)
    right = np.concatenate([xyz_r, rot], axis=1)                     # (T,9)
    return np.concatenate([left, grip_l, right, grip_r], axis=1).astype(np.float32)  # (T,20)


def make_processed_hdf5(
    path,
    T: int = 8,
    source: str = "pika_umi",
    *,
    qpos: np.ndarray | None = None,
    attrs: dict | None = None,
    drop_attr: str | None = None,
    n_img_frames: int | None = None,
    with_timestamps: bool = True,
    timestamps: np.ndarray | None = None,
    image_layout: str = "vlen",
    cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"),
) -> str:
    """写一个 processed XVLA HDF5。返回路径字符串。

    knobs:
      qpos          -- 直接指定 (T,20)（注入非法 rot6d/夹爪/NaN 用）
      attrs         -- 覆盖 attrs（默认取 DEFAULT_ATTRS[source]）
      drop_attr     -- 删掉某个 attr（测缺失）
      n_img_frames  -- 图像帧数（≠T 用于测模态长度不一致）
      with_timestamps -- 是否写 timestamps
      image_layout  -- "vlen" 逐帧 dataset；"chunked_index" 分块 group + *_index
    """
    if qpos is None:
        qpos = valid_qpos(T)
    T = qpos.shape[0]
    a = dict(DEFAULT_ATTRS[source]) if attrs is None else dict(attrs)
    if drop_attr:
        a.pop(drop_attr, None)

    with h5py.File(path, "w") as h:
        d = h.create_dataset("observations/qpos", data=qpos.astype(np.float32))
        for k, v in a.items():
            d.attrs[k] = v
        if with_timestamps:
            ts = timestamps if timestamps is not None else (np.arange(T, dtype=np.float32) / 30.0)
            h.create_dataset("timestamps", data=np.asarray(ts, dtype=np.float32))
        nfr = T if n_img_frames is None else n_img_frames
        if image_layout == "vlen":
            vlen = h5py.vlen_dtype(np.uint8)
            for cam in cameras:
                ds = h.create_dataset(f"observations/images/{cam}", (nfr,), dtype=vlen)
                jpeg = np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)  # 占位最小 JPEG 标记
                for i in range(nfr):
                    ds[i] = jpeg
        elif image_layout == "chunked_index":
            for cam in cameras:
                group = h.create_group(f"observations/images/{cam}")
                split = max(1, nfr // 2)
                group.create_dataset("chunk_000000", data=np.zeros((split, 4), dtype=np.uint8))
                group.create_dataset("chunk_000001", data=np.zeros((nfr - split, 4), dtype=np.uint8))
                h.create_dataset(f"observations/images/{cam}_index", data=np.arange(nfr, dtype=np.int64))
        else:
            raise ValueError(f"unknown image_layout: {image_layout}")
    return str(path)


def _write_vlen_images(h, T: int, cameras=("cam_high", "cam_left_wrist", "cam_right_wrist")) -> None:
    vlen = h5py.vlen_dtype(np.uint8)
    jpeg = np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)
    for cam in cameras:
        ds = h.create_dataset(f"observations/images/{cam}", (T,), dtype=vlen)
        for i in range(T):
            ds[i] = jpeg


def make_raw_pika_hdf5(path, T: int = 12, *, pose: np.ndarray | None = None) -> str:
    idx = np.arange(T, dtype=np.float32)
    if pose is None:
        left = np.stack([0.01 * idx, np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)], axis=1)
        right = np.stack([-0.01 * idx, np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)], axis=1)
    else:
        left, right = pose[:, :6], pose[:, 6:12]
    with h5py.File(path, "w") as h:
        h.attrs["desc"] = "UMI Pika raw, time-synced (raw values, no fusion transform)"
        h.create_dataset("observations/pose_left", data=left.astype(np.float32))
        h.create_dataset("observations/pose_right", data=right.astype(np.float32))
        h.create_dataset("observations/gripper_left", data=np.zeros(T, dtype=np.float32))
        h.create_dataset("observations/gripper_right", data=np.ones(T, dtype=np.float32) * 0.09)
        h.create_dataset("timestamps", data=np.arange(T, dtype=np.float32) / 30.0)
        _write_vlen_images(h, T)
    return str(path)


def make_raw_teleop_hdf5(path, T: int = 12) -> str:
    idx = np.arange(T, dtype=np.float32)
    qpos = np.stack([0.01 * idx for _ in range(14)], axis=1).astype(np.float32)
    action = qpos.copy()
    with h5py.File(path, "w") as h:
        h.attrs["sim"] = False
        h.create_dataset("action", data=action)
        h.create_dataset("base_action", data=np.zeros((T, 2), dtype=np.float32))
        h.create_dataset("observations/qpos", data=qpos)
        h.create_dataset("observations/qvel", data=np.zeros((T, 14), dtype=np.float32))
        h.create_dataset("observations/effort", data=np.zeros((T, 14), dtype=np.float32))
        h.create_dataset("observations/eef_6d", data=np.zeros((T, 20), dtype=np.float32))
        h.create_dataset("observations/eef_quaternion", data=np.zeros((T, 16), dtype=np.float32))
        h.create_dataset("observations/eef_left_time", data=np.arange(T, dtype=np.float32) / 30.0)
        h.create_dataset("observations/eef_right_time", data=np.arange(T, dtype=np.float32) / 30.0)
        _write_vlen_images(h, T)
    return str(path)
