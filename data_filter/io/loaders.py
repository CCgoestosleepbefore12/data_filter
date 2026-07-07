"""HDF5 loaders：raw pika / raw teleop / processed XVLA → 统一内部信号表示。

只做读取与字段抽取（按 io/schema.py 的声明），不做任何质量判断。
图像默认惰性读取，只取帧数不整段解码进内存。兼容两类布局：
- 逐帧 vlen dataset: observations/images/cam_high, shape=(T,)
- 分块 group + index: observations/images/cam_high + observations/images/cam_high_index

raw loader 读取 raw PIKA/UMI 与 raw teleop/NAS 的轻量信号，不解码图像。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np

from . import schema


@dataclass
class EpisodeSignals:
    """一个 episode 的统一内部表示（跨三类来源共用）。"""

    source_kind: str                 # "pika" | "teleop" | "pika_umi" | "nas_teleop" | "unknown"
    path: str
    length: int                      # T
    pose: Optional[np.ndarray] = None        # (T, D) 位姿/pose9
    action: Optional[np.ndarray] = None      # (T, D) 仅遥操/processed
    qpos: Optional[np.ndarray] = None        # (T, D)
    gripper: Optional[np.ndarray] = None     # (T,) 或 (T, n_arm)
    timestamps: Optional[np.ndarray] = None  # (T,)
    extra_timestamps: dict[str, np.ndarray] = field(default_factory=dict)  # 其他同源时钟
    attrs: dict = field(default_factory=dict)     # HDF5/dataset attrs
    image_keys: tuple[str, ...] = ()              # 图像 dataset 名
    image_lengths: dict = field(default_factory=dict)  # {image_key: T}


def _to_py(v):
    """把 h5py attr 值转成原生 python（bytes→str、numpy 标量→python）。"""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _infer_source(attrs: dict) -> str:
    """从 attrs 粗判 processed 来源，决定后续 attrs 契约。"""
    if attrs.get("source_kind") == "nas_teleoperation_eef6d":
        return "nas_teleop"
    if attrs.get("domain_name") == "nas_real_teleop":
        return "nas_teleop"
    if "tip2base_applied" in attrs or str(attrs.get("domain_name", "")).startswith("pika"):
        return "pika_umi"
    return "unknown"


def _dataset_len(obj) -> int:
    """取 dataset 第一维长度；标量 dataset 视为 1。"""
    return int(obj.shape[0]) if obj.shape else 1


def _group_chunk_len(group: h5py.Group) -> int:
    """分块图像 group 无 index 时的兜底：累加子 dataset 第一维。"""
    total = 0
    for child in group.values():
        if isinstance(child, h5py.Dataset):
            total += _dataset_len(child)
    return total


def _collect_image_lengths(h: h5py.File) -> tuple[tuple[str, ...], dict[str, int]]:
    """收集 processed 图像模态长度，不解码图像。

    真实 processed 数据可能把每路相机存成分块 group，并用同级 `*_index`
    dataset 表示帧索引；这种情况下以 index 长度作为帧数。
    """
    img_group = "observations/images"
    if img_group not in h:
        return (), {}

    image_keys: list[str] = []
    image_lengths: dict[str, int] = {}
    g = h[img_group]
    for name, obj in g.items():
        if name.endswith("_index"):
            continue
        key = f"{img_group}/{name}"
        index_key = f"{name}_index"
        if index_key in g and isinstance(g[index_key], h5py.Dataset):
            length = _dataset_len(g[index_key])
        elif isinstance(obj, h5py.Dataset):
            length = _dataset_len(obj)
        elif isinstance(obj, h5py.Group):
            if "index" in obj and isinstance(obj["index"], h5py.Dataset):
                length = _dataset_len(obj["index"])
            else:
                length = _group_chunk_len(obj)
        else:
            continue
        image_keys.append(key)
        image_lengths[key] = int(length)
    return tuple(image_keys), image_lengths


def load_raw_pika(path: str) -> EpisodeSignals:
    """读 raw PIKA/UMI HDF5。pose 合并为 [left6,right6]，gripper 为 (T,2)。"""
    with h5py.File(path, "r") as h:
        pose_l = h[schema.RAW_PIKA["pose"][0]][:]
        pose_r = h[schema.RAW_PIKA["pose"][1]][:]
        grip_l = h[schema.RAW_PIKA["gripper"][0]][:]
        grip_r = h[schema.RAW_PIKA["gripper"][1]][:]
        pose = np.concatenate([pose_l, pose_r], axis=1)
        gripper = np.stack([grip_l, grip_r], axis=1)
        timestamps = h[schema.RAW_PIKA["timestamps"]][:] if schema.RAW_PIKA["timestamps"] in h else None
        attrs = {k: _to_py(v) for k, v in h.attrs.items()}
        image_keys, image_lengths = _collect_image_lengths(h)

    return EpisodeSignals(
        source_kind="pika",
        path=str(path),
        length=int(pose.shape[0]),
        pose=pose,
        gripper=gripper,
        timestamps=timestamps,
        attrs=attrs,
        image_keys=image_keys,
        image_lengths=image_lengths,
    )


def load_raw_teleop(path: str) -> EpisodeSignals:
    """读 raw teleop/NAS HDF5。timestamp 优先用 eef_left_time。"""
    with h5py.File(path, "r") as h:
        qpos = h[schema.RAW_TELEOP["qpos"]][:]
        action = h[schema.RAW_TELEOP["action"]][:]
        timestamps = (
            h[schema.RAW_TELEOP["eef_time"][0]][:]
            if schema.RAW_TELEOP["eef_time"][0] in h
            else None
        )
        attrs = {k: _to_py(v) for k, v in h.attrs.items()}
        extra_timestamps = {}
        if schema.RAW_TELEOP["eef_time"][1] in h:
            extra_timestamps["eef_right_time"] = h[schema.RAW_TELEOP["eef_time"][1]][:]
        image_keys, image_lengths = _collect_image_lengths(h)

    return EpisodeSignals(
        source_kind="teleop",
        path=str(path),
        length=int(qpos.shape[0]),
        qpos=qpos,
        action=action,
        timestamps=timestamps,
        extra_timestamps=extra_timestamps,
        attrs=attrs,
        image_keys=image_keys,
        image_lengths=image_lengths,
    )


def load_processed_xvla(path: str) -> EpisodeSignals:
    """读 processed XVLA HDF5 → EpisodeSignals（不做质量判断）。"""
    with h5py.File(path, "r") as h:
        qpos = h[schema.PROCESSED_QPOS_KEY][:]                    # (T, 20)
        T = int(qpos.shape[0])
        attrs = {k: _to_py(v) for k, v in h[schema.PROCESSED_QPOS_KEY].attrs.items()}
        # 夹爪抽取对异常 shape 稳健：列数不足时留 None，交给 schema_shape 检查报错
        gripper = (
            qpos[:, [schema.LEFT_GRIP, schema.RIGHT_GRIP]]       # (T, 2)
            if qpos.ndim == 2 and qpos.shape[1] >= schema.QPOS_DIM
            else None
        )
        timestamps = h["timestamps"][:] if "timestamps" in h else None

        image_keys, image_lengths = _collect_image_lengths(h)

    return EpisodeSignals(
        source_kind=_infer_source(attrs),
        path=str(path),
        length=T,
        qpos=qpos,
        gripper=gripper,
        timestamps=timestamps,
        attrs=attrs,
        image_keys=image_keys,
        image_lengths=image_lengths,
    )
