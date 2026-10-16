"""CLI：批量生成諮商會談（Phase 5 實作）。

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="批量生成諮商會談")
    parser.add_argument("--config", default="configs/default.yaml", help="設定檔路徑")
    parser.add_argument("--personas", default="all", help="個案清單，逗號分隔或 all")
    parser.add_argument("--sessions", type=int, default=5, help="每個個案的場次數")
    parser.add_argument("--sessions-per-persona", type=int, default=None)
    parser.add_argument("--scenario", default="data/scenarios/initial_session.yaml")
    parser.add_argument("--output", default="data/output/batch_001/", help="輸出目錄")
    parser.add_argument("--resume", action="store_true", help="斷點續傳")
    return parser.parse_args(argv)


def resolve_personas(spec: str) -> list[str]:
    """解析個案清單（名稱、路徑或 all）。"""
    if spec.strip() == "all":
        return sorted(str(p) for p in Path("data/personas").glob("*.yaml"))
    result = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        candidate = Path(item)
        if candidate.exists():
            result.append(str(candidate))
            continue
        by_name = Path("data/personas") / f"{item}.yaml"
        if by_name.exists():
            result.append(str(by_name))
            continue
        print(f"警告：找不到個案 {item}，已跳過", file=sys.stderr)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    from src.config import load_config
    from src.pipeline.batch_generator import BatchGenerator
    from src.pipeline.output_formatter import OutputFormatter
    from src.pipeline.session_generator import SessionGenerator

    args = parse_args(argv)
    sessions = args.sessions_per_persona or args.sessions
    if sessions <= 0:
        print("錯誤：場次數必須為正整數", file=sys.stderr)
        return 2
    personas = resolve_personas(args.personas)
    if not personas:
        print("錯誤：無有效的個案清單", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"錯誤：設定載入失敗：{exc}", file=sys.stderr)
        return 2
    formatter = OutputFormatter(args.output, config.pipeline.embedding_save_format)
    try:
        generator = SessionGenerator.from_config(config, formatter)
    except RuntimeError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    batch = BatchGenerator(config, generator, formatter)
    try:
        result = batch.run(
            personas, sessions, args.output, scenario=args.scenario, resume=args.resume
        )
    except ValueError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 2
    formatter.write_manifest(
        result["sessions"],
        models_used={
            "embedding": config.models.embedding.path,
            "laya": config.models.laya.path,
            "slow_think": config.models.llm.model_name,
            "tts": config.models.tts.engine,
        },
    )
    print(f"完成：成功 {result['succeeded']}/{result['total']}，失敗 {result['failed']}")
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
