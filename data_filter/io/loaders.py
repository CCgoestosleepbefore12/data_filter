"""HDF5 loaders：raw pika / raw teleop / processed XVLA → 统一内部信号表示。

只做读取与字段抽取（按 io/schema.py 的声明），不做任何质量判断。
图像默认惰性读取（vlen JPEG bytes），只取帧数不整段解码进内存。

raw loader 待 milestone 3 实现。
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
    if "tip2base_applied" in attrs or str(attrs.get("domain_name", "")).startswith("pika"):
        return "pika_umi"
    return "unknown"


def load_raw_pika(path: str) -> EpisodeSignals:
    raise NotImplementedError


def load_raw_teleop(path: str) -> EpisodeSignals:
    raise NotImplementedError


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

        image_keys: list[str] = []
        image_lengths: dict[str, int] = {}
        img_group = "observations/images"
        if img_group in h:
            for cam in h[img_group]:
                key = f"{img_group}/{cam}"
                image_keys.append(key)
                image_lengths[key] = int(h[key].shape[0])

    return EpisodeSignals(
        source_kind=_infer_source(attrs),
        path=str(path),
        length=T,
        qpos=qpos,
        gripper=gripper,
        timestamps=timestamps,
        attrs=attrs,
        image_keys=tuple(image_keys),
        image_lengths=image_lengths,
    )
