"""CLI：生成單場諮商會談（骨架，Phase 5 完整實作）。

用法：
    python scripts/generate_session.py \
        --persona data/personas/mdd_severe.yaml \
        --scenario data/scenarios/initial_session.yaml \
        --output data/output/ --save-audio --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="生成單場諮商會談")
    parser.add_argument("--persona", required=True, help="個案 YAML 路徑")
    parser.add_argument("--scenario", required=True, help="場景 YAML 路徑")
    parser.add_argument("--config", default="configs/default.yaml", help="設定檔路徑")
    parser.add_argument("--output", default="data/output/", help="輸出目錄")
    parser.add_argument("--save-audio", action="store_true", help="儲存音訊")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口（Phase 5 完整實作）。"""
    args = parse_args(argv)
    for label, path in (("個案", args.persona), ("場景", args.scenario)):
        if not Path(path).exists():
            print(f"錯誤：{label}檔案不存在：{path}", file=sys.stderr)
            return 2
    print("SessionGenerator 排定於 Phase 5 實作，目前僅驗證參數與檔案存在。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
