"""配置加载：按源分离的 YAML（configs/raw_pika|raw_teleop|processed_xvla.yaml）。

阈值全部为 provisional，待真实分布与 review 队列校准。
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """name: "raw_pika" | "raw_teleop" | "processed_xvla"。返回该源的配置 dict。"""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
