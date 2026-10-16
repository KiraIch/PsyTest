"""驗證環境：檢查模型、依賴與 Ollama 是否就緒。

用法：
    python scripts/validate_setup.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REQUIRED_PACKAGES = (
    "yaml",
    "numpy",
    "requests",
    "pydantic",
)


def check_packages() -> list[str]:
    """回傳缺失的套件清單。"""
    missing = []
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def main(argv: list[str] | None = None) -> int:
    """執行檢查並回傳結束碼（0 表基本就緒）。"""
    parser = argparse.ArgumentParser(description="驗證 PsyTest 環境")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    ok = True
    missing = check_packages()
    if missing:
        print(f"[缺失套件] {', '.join(missing)}，請執行 pip install -r requirements.txt")
        ok = False
    else:
        print("[套件] 基本依賴就緒")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.backend.detector import detect_backend, format_report
        from src.config import find_existing_model_path

        print(format_report(detect_backend()))
        for label, rel in (
            ("Embedding", "models/embeddinggemma2-q8.gguf"),
            ("Laya", "models/laya-421m-bf16.gguf"),
        ):
            found = find_existing_model_path(rel)
            print(f"[模型] {label}: {found if found else '未找到（功能將受限）'}")
        print(f"[設定] {args.config}: {'存在' if Path(args.config).exists() else '不存在'}")
        try:
            from src.voice.tts_engine import list_available_engines

            status = list_available_engines()
            ready = [name for name, available in status.items() if available]
            print(f"[TTS] 可用引擎：{ready if ready else '無（至少安裝 pyttsx3）'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[TTS] 檢查失敗：{exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[錯誤] 環境檢查失敗：{exc}")
        return 1

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
