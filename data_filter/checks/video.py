"""C3 视频质量 + decode 约定一致性。

视频质量（quality 类，出 frame_mask）：黑帧、损坏帧、模糊、长静止段；
静止检测需与 state/action 联合，保留夹爪闭合等关键帧。

decode 约定一致性（hard 类）：验证 raw→processed 用同一 decode contract
（核对 decode 路径/attrs，或对固定参考帧比对通道统计）——**不做纯像素 BGR 检测**
（无 ground truth 不可靠）。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

import numpy as np

from .base import CheckResult


def check_video_quality(frames: np.ndarray, cfg: dict) -> CheckResult:
    """frames: (T, H, W, 3) 已解码图像。返回黑/糊/静止 frame_mask 与比例。"""
    raise NotImplementedError


def check_decode_contract(raw_meta: dict, processed_meta: dict, cfg: dict) -> CheckResult:
    """核对 raw/processed 的 decode 约定一致。返回一致性结论。"""
    raise NotImplementedError
