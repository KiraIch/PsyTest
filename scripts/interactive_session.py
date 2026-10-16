"""CLI：互動模式（Phase 5 實作，文字互動測試用）。

用法：
    python scripts/interactive_session.py --persona data/personas/bpd_crisis.yaml

輸入 `quit` 結束。TTS 預設關閉，僅顯示對白、情緒與導演指令。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="互動式諮商測試")
    parser.add_argument("--persona", required=True, help="個案 YAML 路徑")
    parser.add_argument("--config", default="configs/default.yaml", help="設定檔路徑")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    from src.config import load_config
    from src.fast_think.emotion_router import EmotionRouter
    from src.slow_think.director import Director
    from src.slow_think.ollama_client import GenerateOptions, OllamaClient
    from src.slow_think.persona_manager import PersonaManager

    args = parse_args(argv)
    if not Path(args.persona).exists():
        print(f"錯誤：個案檔案不存在：{args.persona}", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"錯誤：設定載入失敗：{exc}", file=sys.stderr)
        return 2
    manager = PersonaManager()
    try:
        persona = manager.load_persona(args.persona)
    except (FileNotFoundError, ValueError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 2
    client = OllamaClient(host=config.models.llm.host)
    try:
        client.ensure_running()
        model = client.detect_model(config.models.llm.model_name)
    except RuntimeError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1
    system_prompt = manager.build_system_prompt(persona)
    director = Director()
    router = EmotionRouter()
    history: list[str] = []
    print(f"個案：{persona.name}（{persona.diagnosis}），輸入 quit 結束。")
    while True:
        try:
            counselor_text = input("諮商師> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if counselor_text.lower() in {"quit", "exit", "q"}:
            break
        if not counselor_text:
            continue
        prompt = (
            f"{system_prompt}\n\n【對話歷史】\n"
            + ("\n".join(history[-6:]) if history else "（會談開始）")
            + f"\n\n諮商師：{counselor_text}\n個案："
        )
        try:
            result = client.generate(
                prompt,
                model=model,
                options=GenerateOptions(
                    temperature=config.models.llm.temperature,
                    max_tokens=config.models.llm.max_tokens,
                ),
            )
        except RuntimeError as exc:
            print(f"生成失敗：{exc}")
            continue
        dialog, notes = director.parse(str(result.get("text", "")))
        route = router.route(dialog or counselor_text, "")
        print(f"個案> {dialog or '（沉默）'}")
        print(f"  [情緒 {route.emotion} {route.intensity:.2f} | 語氣 {notes.tone}]")
        history.append(f"諮商師：{counselor_text}\n個案：{dialog}")
    print("互動結束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
