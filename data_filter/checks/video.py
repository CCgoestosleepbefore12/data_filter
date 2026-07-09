"""C3 视频质量 + decode 约定一致性。

视频质量（quality 类，出 frame_mask）：黑帧、损坏帧、模糊、长静止段；
静止检测需与 state/action 联合，保留夹爪闭合等关键帧。

decode 约定一致性（hard 类）：验证 raw→processed 用同一 decode contract
（核对 decode 路径/attrs，或对固定参考帧比对通道统计）——**不做纯像素 BGR 检测**
（无 ground truth 不可靠）。

第一版 V2 先做轻量图像质量：黑帧、模糊、长静止画面。输入可以是已解码
`(T,H,W,3)` 数组，也可以是 JPEG bytes 列表。
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from .base import CheckResult


def check_video_quality(frames, cfg: dict, name: str = "video_quality", camera: str | None = None) -> CheckResult:
    """检查黑帧、模糊、长静止画面。

    frames:
      - 已解码图像数组 `(T,H,W,3)` / `(T,H,W)`
      - JPEG bytes / uint8 arrays 的列表
    """
    decoded, decode_failures = _decode_frames(frames)
    if decoded.size == 0:
        return CheckResult(
            name=name,
            passed=True,
            severity="warn",
            metrics={"camera": camera or "", "n_frames": 0, "decode_failures": int(decode_failures)},
            flags=[_flag(camera, "decode_failed")] if decode_failures else [_flag(camera, "missing_frames")],
        )

    gray = _to_gray(decoded)
    luma = gray.mean(axis=(1, 2))
    blur = np.asarray([_laplacian_var(g) for g in gray], dtype=np.float64)

    black_luma = float(cfg.get("black_luma", 8.0))
    blur_var = float(cfg.get("blur_var", 50.0))
    max_black_ratio = float(cfg.get("max_black_ratio", 0.05))
    max_blur_ratio = float(cfg.get("max_blur_ratio", 0.5))
    static_min_frames = int(cfg.get("static_min_frames", 45))
    static_diff_eps = float(cfg.get("static_diff_eps", 1.0))
    max_decode_failure_ratio = float(cfg.get("max_decode_failure_ratio", 0.0))

    black_mask = luma < black_luma
    blur_mask = blur < blur_var
    static_run = _longest_static_run(gray, static_diff_eps)
    n_total = int(decoded.shape[0] + decode_failures)
    decode_failure_ratio = float(decode_failures / n_total) if n_total else 0.0
    black_ratio = float(np.count_nonzero(black_mask) / decoded.shape[0])
    blur_ratio = float(np.count_nonzero(blur_mask) / decoded.shape[0])

    flags: list[str] = []
    if decode_failure_ratio > max_decode_failure_ratio:
        flags.append(_flag(camera, "decode_failed"))
    if black_ratio > max_black_ratio:
        flags.append(_flag(camera, "black"))
    if blur_ratio > max_blur_ratio:
        flags.append(_flag(camera, "blur"))
    if static_run >= static_min_frames:
        flags.append(_flag(camera, "static"))

    frame_mask = black_mask | blur_mask
    return CheckResult(
        name=name,
        passed=True,
        severity="warn" if flags else "info",
        frame_mask=frame_mask,
        metrics={
            "camera": camera or "",
            "n_frames": int(decoded.shape[0]),
            "decode_failures": int(decode_failures),
            "decode_failure_ratio": decode_failure_ratio,
            "black_ratio": black_ratio,
            "blur_ratio": blur_ratio,
            "longest_static_run": int(static_run),
            "luma_min": float(luma.min()) if luma.size else 0.0,
            "luma_median": float(np.median(luma)) if luma.size else 0.0,
            "blur_var_median": float(np.median(blur)) if blur.size else 0.0,
        },
        flags=flags,
    )


def check_decode_contract(raw_meta: dict, processed_meta: dict, cfg: dict) -> CheckResult:
    """核对 raw/processed 的 decode 约定一致。返回一致性结论。"""
    raise NotImplementedError


def _decode_frames(frames) -> tuple[np.ndarray, int]:
    if isinstance(frames, np.ndarray) and frames.ndim >= 3 and frames.dtype != object:
        arr = frames
        if arr.ndim == 3:
            arr = arr[:, :, :, None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr.astype(np.uint8, copy=False), 0

    decoded: list[np.ndarray] = []
    failures = 0
    for item in frames:
        try:
            data = item.tobytes() if isinstance(item, np.ndarray) else bytes(item)
            with Image.open(BytesIO(data)) as img:
                decoded.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
        except Exception:
            failures += 1
    if not decoded:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), failures
    return np.stack(decoded, axis=0), failures


def _to_gray(frames: np.ndarray) -> np.ndarray:
    x = frames.astype(np.float64)
    if x.shape[-1] == 1:
        return x[..., 0]
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def _laplacian_var(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(np.var(lap))


def _longest_static_run(gray: np.ndarray, eps: float) -> int:
    if gray.shape[0] < 2:
        return 0
    diffs = np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2))
    longest = cur = 0
    for hit in diffs <= eps:
        if hit:
            cur += 1
            longest = max(longest, cur + 1)
        else:
            cur = 0
    return int(longest)


def _flag(camera: str | None, flag: str) -> str:
    return f"{camera}_{flag}" if camera else flag
