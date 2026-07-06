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
        "source_kind": "nas_teleoperation_eef6d",
        "domain_name": "nas_real_teleop",
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
    cameras=("cam_high", "cam_left_wrist", "cam_right_wrist"),
) -> str:
    """写一个 processed XVLA HDF5。返回路径字符串。

    knobs:
      qpos          -- 直接指定 (T,20)（注入非法 rot6d/夹爪/NaN 用）
      attrs         -- 覆盖 attrs（默认取 DEFAULT_ATTRS[source]）
      drop_attr     -- 删掉某个 attr（测缺失）
      n_img_frames  -- 图像帧数（≠T 用于测模态长度不一致）
      with_timestamps -- 是否写 timestamps
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
            h.create_dataset("timestamps", data=(np.arange(T, dtype=np.float32) / 30.0))
        nfr = T if n_img_frames is None else n_img_frames
        vlen = h5py.vlen_dtype(np.uint8)
        for cam in cameras:
            ds = h.create_dataset(f"observations/images/{cam}", (nfr,), dtype=vlen)
            jpeg = np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)  # 占位最小 JPEG 标记
            for i in range(nfr):
                ds[i] = jpeg
    return str(path)
