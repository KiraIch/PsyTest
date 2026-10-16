"""CLI：批量生成諮商會談（骨架，Phase 5 完整實作）。

用法：
    python scripts/batch_generate.py \
        --config configs/default.yaml \
        --personas mdd_severe,bpd_crisis,ptsd_combat \
        --sessions-per-persona 10 \
        --output data/output/ --resume
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="批量生成諮商會談")
    parser.add_argument("--config", default="configs/default.yaml", help="設定檔路徑")
    parser.add_argument("--personas", default="all", help="個案清單，逗號分隔或 all")
    parser.add_argument("--sessions", type=int, default=5, help="每個個案的場次數")
    parser.add_argument("--sessions-per-persona", type=int, default=None)
    parser.add_argument("--output", default="data/output/batch_001/", help="輸出目錄")
    parser.add_argument("--resume", action="store_true", help="斷點續傳")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口（Phase 5 完整實作）。"""
    args = parse_args(argv)
    sessions = args.sessions_per_persona or args.sessions
    if sessions <= 0:
        print("錯誤：場次數必須為正整數", file=sys.stderr)
        return 2
    print("BatchGenerator 排定於 Phase 5 實作，目前僅驗證參數。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
