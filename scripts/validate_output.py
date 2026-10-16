"""驗證輸出數據完整性（骨架，Phase 5 完整實作）。

用法：
    python scripts/validate_output.py \
        --input data/output/batch_001/ --check-embeddings --check-audio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="驗證輸出數據完整性")
    parser.add_argument("--input", required=True, help="輸出目錄")
    parser.add_argument("--check-embeddings", action="store_true")
    parser.add_argument("--check-audio", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口（Phase 5 完整實作）。"""
    args = parse_args(argv)
    if not Path(args.input).exists():
        print(f"錯誤：輸入目錄不存在：{args.input}", file=sys.stderr)
        return 2
    print("OutputFormatter 驗證排定於 Phase 5 實作，目前僅驗證目錄存在。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
