"""數據管線測試（Phase 5 已實作：單輪、完整會談、輸出格式）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import yaml

from src.config import load_config


class _FakeLLM:
    """固定回應的假 LLM（回傳導演格式文本）。"""

    def __init__(self, text: str) -> None:
        """初始化固定回應文本。"""
        self.text = text

    def generate(
        self, prompt: str, model: Optional[str] = None, options: object = None
    ) -> Dict[str, Any]:
        """回傳固定文本（忽略輸入）。"""
        assert prompt
        return {"text": self.text, "model": "fake"}


def _write_scenario(tmp_path: Path, max_turns: int) -> str:
    """建立臨時場景檔並回傳路徑。"""
    path = tmp_path / "scenario.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "description": "測試場景",
                "max_turns": max_turns,
                "counselor_opening": "你好，最近怎麼樣？",
                "techniques": ["open_question"],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return str(path)


def test_config_loads() -> None:
    """確認預設設定可載入且通過驗證。"""
    config = load_config("configs/default.yaml")
    assert config.project_name == "PsyTest"
    assert config.pipeline.max_turns_per_session > 0


def test_single_turn(tmp_path: Path) -> None:
    """單輪對話生成（假 LLM，不需服務）。"""
    from src.pipeline.session_generator import SessionGenerator

    config = load_config("configs/default.yaml")
    config.pipeline.save_audio = False
    generator = SessionGenerator(
        config,
        llm_client=_FakeLLM("我最近睡不好。<director></director>"),
    )
    session = generator.generate(
        "data/personas/mdd_severe.yaml", _write_scenario(tmp_path, 1)
    )
    assert len(session["turns"]) == 1
    turn = session["turns"][0]
    assert turn["turn_id"] == 1
    assert turn["counselor"]["text"]
    assert turn["client"]["psychological"]["crisis_level"] == 0


def test_full_session() -> None:
    """完整會談生成（真實 Ollama，2 輪，無音訊）。"""
    import requests

    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:  # noqa: BLE001
        pytest.skip("Ollama 未運行，跳過完整會談測試")
    import tempfile

    from src.pipeline.session_generator import SessionGenerator

    config = load_config("configs/default.yaml")
    config.pipeline.save_audio = False
    config.models.llm.max_tokens = 64
    with tempfile.TemporaryDirectory() as tmpdir:
        scenario = Path(tmpdir) / "s.yaml"
        scenario.write_text(
            yaml.safe_dump(
                {
                    "name": "test",
                    "description": "測試",
                    "max_turns": 2,
                    "counselor_opening": "你好。",
                    "techniques": ["open_question"],
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        generator = SessionGenerator.from_config(config)
        session = generator.generate("data/personas/mdd_severe.yaml", str(scenario))
    assert len(session["turns"]) >= 1
    assert session["end_reason"]
    assert session["summary"]["total_turns"] == len(session["turns"])


def test_output_format(tmp_path: Path) -> None:
    """輸出 JSONL 往返驗證（首行 meta，其後每輪一行）。"""
    import json

    from src.pipeline.output_formatter import OutputFormatter

    formatter = OutputFormatter(str(tmp_path))
    session = {
        "session_id": "abc123",
        "persona": "mdd_severe",
        "summary": {"total_turns": 1},
        "turns": [
            {
                "session_id": "abc123",
                "turn_id": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    path = formatter.write_session(session)
    assert path.exists()
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]
    assert len(lines) == 2
    meta = json.loads(lines[0])
    turn = json.loads(lines[1])
    assert meta["record"] == "meta" and meta["session_id"] == "abc123"
    assert meta["turn_count"] == 1
    assert turn["record"] == "turn" and turn["turn_id"] == 1
