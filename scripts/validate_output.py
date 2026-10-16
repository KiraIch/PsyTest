"""驗證輸出數據完整性（Phase 5 實作）。

用法：
    python scripts/validate_output.py \
        --input data/output/batch_001/ --check-embeddings --check-audio
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="驗證輸出數據完整性")
    parser.add_argument("--input", required=True, help="輸出目錄")
    parser.add_argument("--check-embeddings", action="store_true")
    parser.add_argument("--check-audio", action="store_true")
    return parser.parse_args(argv)


def _check_session(path: Path, check_embeddings: bool) -> list[str]:
    """驗證單一會談檔，回傳錯誤清單（支援新舊兩種格式）。"""
    errors: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
    except OSError as exc:
        return [f"{path.name}：讀取失敗：{exc}"]
    if not lines:
        return [f"{path.name}：檔案為空"]
    try:
        records = [json.loads(line) for line in lines]
    except ValueError as exc:
        return [f"{path.name}：JSON 解析失敗：{exc}"]
    if len(records) == 1 and "turns" in records[0]:
        meta, turns = records[0], records[0].get("turns", [])
    else:
        meta = next((r for r in records if r.get("record") == "meta"), None)
        turns = [r for r in records if r.get("record") == "turn"]
        if meta is None:
            errors.append(f"{path.name}：缺少 meta 行")
            return errors
    for key in ("session_id", "summary"):
        if key not in meta:
            errors.append(f"{path.name}：meta 缺少 {key}")
    if not isinstance(turns, list) or not turns:
        errors.append(f"{path.name}：turns 非法或為空")
        return errors
    session_id = meta.get("session_id")
    for turn in turns:
        if "error" in turn:
            continue
        if turn.get("session_id") != session_id:
            errors.append(f"{path.name}：輪次 session_id 與 meta 不一致")
        for key in ("session_id", "turn_id", "timestamp", "counselor", "client"):
            if key not in turn:
                errors.append(f"{path.name} 第 {turn.get('turn_id')} 輪：缺少 {key}")
        if check_embeddings:
            vectors = turn.get("embedding_vectors", {})
            text_vec = vectors.get("text_embedding")
            if text_vec is not None:
                data = text_vec.get("data") if isinstance(text_vec, dict) else text_vec
                if not isinstance(data, list) or len(data) != 768:
                    errors.append(
                        f"{path.name} 第 {turn.get('turn_id')} 輪："
                        f"文字嵌入維度異常（{len(data) if isinstance(data, list) else type(data).__name__}）"
                    )
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    args = parse_args(argv)
    root = Path(args.input)
    if not root.exists():
        print(f"錯誤：輸入目錄不存在：{args.input}", file=sys.stderr)
        return 2
    errors: list[str] = []
    sessions = sorted((root / "sessions").glob("*.jsonl")) if (root / "sessions").is_dir() else []
    if not sessions:
        print(f"警告：{args.input} 下無會談檔案")
    for path in sessions:
        errors.extend(_check_session(path, args.check_embeddings))
    if args.check_audio:
        for path in sessions:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    session = json.loads(fh.readline())
            except (OSError, ValueError):
                continue
            for turn in session.get("turns", []):
                audio_path = turn.get("client", {}).get("audio", {}).get("file_path")
                if audio_path and not Path(audio_path).exists():
                    errors.append(f"音檔缺失：{audio_path}")
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError as exc:
            errors.append(f"manifest 解析失敗：{exc}")
    else:
        print("提醒：無 manifest.json（批量生成完成後會產生）")
    if errors:
        print(f"驗證失敗，共 {len(errors)} 項：")
        for message in errors[:20]:
            print(f"  - {message}")
        return 1
    print(f"驗證通過：{len(sessions)} 場會談")
    return 0


if __name__ == "__main__":
    sys.exit(main())
