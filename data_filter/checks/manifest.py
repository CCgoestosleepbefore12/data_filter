"""manifest 覆盖检查（processed，hard-validity）。

核对 manifest 的文件数 / 路径 / 训练配置 dataset_path 权重与实际 HDF5 一致，
抓「配置里声明了但磁盘缺失」「磁盘有但未纳入训练」「权重与实际条目不符」等静默漏数据。

TODO(milestone 2): /tdd 实现。
"""

from __future__ import annotations

from .base import CheckResult


def check_manifest_coverage(manifest: dict, actual_files: list[str], cfg: dict) -> CheckResult:
    """manifest: 声明的文件/权重。actual_files: 实际 HDF5 路径。返回覆盖一致性。"""
    raise NotImplementedError
