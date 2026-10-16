"""CLI：產出督導報告（Phase 6 實作）。

用法：
    python scripts/generate_report.py \
        --input data/output/sessions/session_<id>.jsonl \
        --output data/output/annotations/report_<id>.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="產出督導報告")
    parser.add_argument("--input", required=True, help="會談 JSONL 路徑")
    parser.add_argument("--output", required=True, help="報告輸出路徑（.md）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    from src.report.supervisor import Supervisor

    args = parse_args(argv)
    source = Path(args.input)
    if not source.exists():
        print(f"錯誤：輸入檔案不存在：{args.input}", file=sys.stderr)
        return 2
    try:
        with open(source, "r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        if len(lines) != 1:
            print(f"錯誤：會談檔應為單行 JSONL，實際 {len(lines)} 行", file=sys.stderr)
            return 2
        session = json.loads(lines[0])
    except (OSError, ValueError) as exc:
        print(f"錯誤：讀取失敗：{exc}", file=sys.stderr)
        return 1
    supervisor = Supervisor()
    try:
        analysis = supervisor.analyze(session)
        report = supervisor.format_report(session, analysis)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    try:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    except OSError as exc:
        print(f"錯誤：寫入失敗：{exc}", file=sys.stderr)
        return 1
    print(f"完成：{args.output}（最高危機 {analysis['max_crisis_level']} 級）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
