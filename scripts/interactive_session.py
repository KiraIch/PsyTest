"""CLI：互動模式（測試用，骨架，Phase 5 完整實作）。

用法：
    python scripts/interactive_session.py --persona data/personas/bpd_crisis.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="互動式諮商測試")
    parser.add_argument("--persona", required=True, help="個案 YAML 路徑")
    parser.add_argument("--config", default="configs/default.yaml", help="設定檔路徑")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口（Phase 5 完整實作）。"""
    args = parse_args(argv)
    if not Path(args.persona).exists():
        print(f"錯誤：個案檔案不存在：{args.persona}", file=sys.stderr)
        return 2
    print("互動模式排定於 Phase 5 實作。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
