"""CLI：生成單場諮商會談（Phase 5 實作）。

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="生成單場諮商會談")
    parser.add_argument("--persona", required=True, help="個案 YAML 路徑")
    parser.add_argument("--scenario", required=True, help="場景 YAML 路徑")
    parser.add_argument("--config", default="configs/default.yaml", help="設定檔路徑")
    parser.add_argument("--output", default="data/output/", help="輸出目錄")
    parser.add_argument("--save-audio", action="store_true", help="儲存音訊")
    parser.add_argument("--no-audio", action="store_true", help="不合成音訊")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    from src.config import load_config
    from src.pipeline.output_formatter import OutputFormatter
    from src.pipeline.session_generator import SessionGenerator

    args = parse_args(argv)
    for label, path in (("個案", args.persona), ("場景", args.scenario)):
        if not Path(path).exists():
            print(f"錯誤：{label}檔案不存在：{path}", file=sys.stderr)
            return 2
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"錯誤：設定載入失敗：{exc}", file=sys.stderr)
        return 2
    if args.no_audio:
        config.pipeline.save_audio = False
    elif args.save_audio:
        config.pipeline.save_audio = True
    formatter = OutputFormatter(args.output, config.pipeline.embedding_save_format)

    def show_progress(turn: dict) -> None:
        if not args.verbose:
            return
        client = turn.get("client", {})
        psych = client.get("psychological", {})
        print(
            f"[第 {turn.get('turn_id')} 輪] 情緒={psych.get('emotion')} "
            f"危機={psych.get('crisis_level')} "
            f"對白={client.get('text', '')[:40]}"
        )

    try:
        generator = SessionGenerator.from_config(config, formatter)
    except RuntimeError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    try:
        session = generator.generate(
            args.persona, args.scenario, on_turn=show_progress if args.verbose else None
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"錯誤：生成失敗：{exc}", file=sys.stderr)
        return 1
    formatter.write_session(session)
    formatter.write_annotations(
        session["session_id"],
        {"summary": session["summary"], "end_reason": session["end_reason"]},
    )
    summary = session["summary"]
    print(
        f"完成：會談 {session['session_id']}，{summary['total_turns']} 輪，"
        f"最高危機 {summary['max_crisis_level']}，結束原因 {session['end_reason']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
