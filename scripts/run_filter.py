#!/usr/bin/env python3
"""data_filter CLI 入口。

用法（milestone 6 后）：
    python scripts/run_filter.py --gate raw       --source pika    --root <dir> --out <dir>
    python scripts/run_filter.py --gate processed  --root <dir> --out <dir>

现为脚手架占位。
"""

from __future__ import annotations

from data_filter.pipeline import main

if __name__ == "__main__":
    main()
