"""Phase 5 數據管線測試：收集器、格式化、分析器、批量、腳本驗證。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import yaml

from src.config import load_config
from src.perception.audio_analyzer import AudioAnalyzer
from src.perception.visual_analyzer import VisualAnalyzer
from src.pipeline.data_collector import DataCollector
from src.pipeline.output_formatter import OutputFormatter, quantize_embedding


class _FakeLLM:
    """固定回應的假 LLM。"""

    def __init__(self, text: str) -> None:
        """初始化固定回應文本。"""
        self.text = text

    def generate(
        self, prompt: str, model: Optional[str] = None, options: object = None
    ) -> Dict[str, Any]:
        """回傳固定文本（忽略輸入）。"""
        assert prompt
        return {"text": self.text, "model": "fake"}


class _FakeGenerator:
    """固定產出的假單場生成器（批量測試用）。"""

    def __init__(self) -> None:
        """初始化計數器。"""
        self.calls = 0

    def generate(self, persona_path: str, scenario: str) -> Dict[str, Any]:
        """回傳最小會談並計數。"""
        self.calls += 1
        session_id = f"fake{self.calls}"
        return {
            "session_id": session_id,
            "persona": Path(persona_path).stem,
            "turns": [],
            "summary": {"total_turns": 0},
            "end_reason": "max_turns",
        }


def _write_scenario(tmp_path: Path, max_turns: int) -> str:
    """建立臨時場景檔並回傳路徑。"""
    path = tmp_path / f"scenario_{max_turns}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "description": "測試",
                "max_turns": max_turns,
                "counselor_opening": "你好。",
                "techniques": ["open_question"],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# 數據收集器
# ---------------------------------------------------------------------------


def test_collector_normalizes_turn() -> None:
    """完整輪次應正規化為四大區塊。"""
    turn = DataCollector().collect_turn(
        {
            "session_id": "s1",
            "turn_id": 1,
            "timestamp": "t",
            "counselor": {"text": "你好"},
            "client": {"text": "嗨", "psychological": {"crisis_level": 2}},
        }
    )
    assert turn["counselor"]["technique"] == "open_question"
    assert turn["client"]["psychological"]["crisis_level"] == 2
    assert "embedding_vectors" in turn


def test_collector_clamps_crisis() -> None:
    """危機等級應鉗制在 0 至 5。"""
    turn = DataCollector().collect_turn(
        {
            "session_id": "s1",
            "turn_id": 1,
            "timestamp": "t",
            "client": {"psychological": {"crisis_level": 9}},
        }
    )
    assert turn["client"]["psychological"]["crisis_level"] == 5


def test_collector_rejects_missing_fields() -> None:
    """缺必要欄位應拋出 ValueError。"""
    with pytest.raises(ValueError):
        DataCollector().collect_turn({"session_id": "s1"})


def test_collector_session_tolerates_errors() -> None:
    """整場收集應容忍單輪錯誤。"""
    turns = DataCollector().collect_session([{"broken": True}])
    assert len(turns) == 1
    assert "error" in turns[0]


# ---------------------------------------------------------------------------
# 輸出格式化
# ---------------------------------------------------------------------------


def test_quantize_formats() -> None:
    """三種量化格式應正確。"""
    vec = [0.5, -0.25, 0.0]
    assert quantize_embedding(vec, "float32") == [0.5, -0.25, 0.0]
    assert len(quantize_embedding(vec, "float16")) == 3
    packed = quantize_embedding(vec, "int8")
    assert packed["quant"] == "int8" and len(packed["data"]) == 3
    with pytest.raises(ValueError):
        quantize_embedding(vec, "fp8")


def test_manifest_stats(tmp_path: Path) -> None:
    """manifest 統計應正確。"""
    formatter = OutputFormatter(str(tmp_path))
    path = formatter.write_manifest(
        [
            {
                "persona": "mdd_severe",
                "turns": [
                    {"client": {"audio": {"duration": 3.0}}},
                    {"client": {"audio": {"duration": 5.0}}},
                ],
            }
        ],
        models_used={"tts": "kokoro"},
        hardware="test",
    )
    assert path.exists()
    import json

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["total_sessions"] == 1
    assert manifest["total_turns"] == 2
    assert manifest["personas"] == ["mdd_severe"]


def test_write_audio_sine(tmp_path: Path) -> None:
    """合成正弦波應可寫入 WAV。"""
    formatter = OutputFormatter(str(tmp_path))
    wave = (0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    path = formatter.write_audio("s1", 1, wave, 16000)
    assert path is not None and path.exists()
    assert formatter.write_audio("s1", 2, np.zeros(0), 16000) is None


# ---------------------------------------------------------------------------
# 特徵分析器
# ---------------------------------------------------------------------------


def test_audio_analyzer_sine() -> None:
    """440Hz 正弦波特徵應在合理範圍。"""
    sr = 16000
    wave = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(float)
    features = AudioAnalyzer(sample_rate=sr).analyze(wave)
    assert features.duration == pytest.approx(1.0)
    assert features.energy > 0.5
    assert 400 <= features.pitch_mean <= 480
    assert features.pause_segments == []
    assert features.spectral_features["centroid"] > 0


def test_audio_analyzer_silence() -> None:
    """靜音應偵測為停頓且能量為零。"""
    features = AudioAnalyzer().analyze(np.zeros(16000))
    assert features.energy == pytest.approx(0.0)
    assert len(features.pause_segments) == 1
    with pytest.raises(ValueError):
        AudioAnalyzer().analyze(np.zeros(0))


def test_visual_analyzer_mapping() -> None:
    """參數映射與情緒補齊應正確。"""
    analyzer = VisualAnalyzer()
    features = analyzer.analyze({"gaze_direction": "down"}, emotion="shame")
    assert features.gaze_direction == "down"
    assert features.body_posture == "closed"
    fallback = analyzer.analyze({"gaze_direction": "亂看"}, emotion="anger")
    assert fallback.gaze_direction == "direct"
    with pytest.raises(ValueError):
        analyzer.analyze("不是字典")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 批量生成
# ---------------------------------------------------------------------------


def test_batch_run_and_resume(tmp_path: Path) -> None:
    """批量運行與斷點續傳應正確。"""
    from src.pipeline.batch_generator import BatchGenerator

    config = load_config("configs/default.yaml")
    formatter = OutputFormatter(str(tmp_path))
    fake = _FakeGenerator()
    batch = BatchGenerator(config, fake, formatter)
    result = batch.run(
        ["data/personas/mdd_severe.yaml", "data/personas/bpd_crisis.yaml"],
        sessions_per_persona=2,
        output_dir=str(tmp_path),
        scenario="data/scenarios/initial_session.yaml",
    )
    assert result["succeeded"] == 4 and result["failed"] == 0
    assert (Path(str(tmp_path)) / "checkpoints" / "progress.json").exists()
    resumed = batch.run(
        ["data/personas/mdd_severe.yaml", "data/personas/bpd_crisis.yaml"],
        sessions_per_persona=2,
        output_dir=str(tmp_path),
        scenario="data/scenarios/initial_session.yaml",
        resume=True,
    )
    assert resumed["succeeded"] == 0
    assert fake.calls == 4


def test_batch_rejects_empty() -> None:
    """空個案清單應拋出 ValueError。"""
    from src.pipeline.batch_generator import BatchGenerator

    config = load_config("configs/default.yaml")
    with pytest.raises(ValueError):
        BatchGenerator(config, _FakeGenerator()).run([], 1, "out")


# ---------------------------------------------------------------------------
# 危機中斷與腳本驗證
# ---------------------------------------------------------------------------


def test_crisis_level_5_ends_session(tmp_path: Path) -> None:
    """危機 5 級應立即結束會談。"""
    from src.pipeline.session_generator import SessionGenerator

    config = load_config("configs/default.yaml")
    config.pipeline.save_audio = False
    generator = SessionGenerator(
        config, llm_client=_FakeLLM("我想自殺，活不下去了。<director></director>")
    )
    session = generator.generate(
        "data/personas/bpd_crisis.yaml", _write_scenario(tmp_path, 10)
    )
    assert session["end_reason"] == "crisis_level_5"
    assert session["summary"]["max_crisis_level"] == 5


def test_validate_output_script(tmp_path: Path) -> None:
    """validate_output 腳本應通過合法輸出。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_output", "scripts/validate_output.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    formatter = OutputFormatter(str(tmp_path))
    formatter.write_session(
        {
            "session_id": "v1",
            "persona": "mdd_severe",
            "summary": {"total_turns": 1},
            "turns": [
                {
                    "session_id": "v1",
                    "turn_id": 1,
                    "timestamp": "t",
                    "counselor": {"text": "你好"},
                    "client": {"text": "嗨"},
                    "embedding_vectors": {"text_embedding": [0.0] * 768},
                }
            ],
        }
    )
    assert module.main(["--input", str(tmp_path), "--check-embeddings"]) == 0


def test_end_to_end_with_audio() -> None:
    """端到端：真實 Ollama 加 Kokoro，2 輪含音訊（較慢）。"""
    import requests

    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
    except Exception:  # noqa: BLE001
        pytest.skip("Ollama 未運行，跳過端到端測試")
    import tempfile

    from src.pipeline.session_generator import SessionGenerator

    config = load_config("configs/default.yaml")
    config.models.llm.max_tokens = 64
    with tempfile.TemporaryDirectory() as tmpdir:
        formatter = OutputFormatter(tmpdir)
        generator = SessionGenerator.from_config(config, formatter)
        if generator.tts_engine is None or getattr(generator.tts_engine, "name", "") != "kokoro":
            pytest.skip("Kokoro 未就緒，跳過含音訊端到端測試")
        session = generator.generate(
            "data/personas/mdd_severe.yaml", _write_scenario(Path(tmpdir), 2)
        )
        formatter.write_session(session)
        assert len(session["turns"]) >= 1
        audio_paths = [
            t["client"]["audio"]["file_path"]
            for t in session["turns"]
            if t.get("client", {}).get("audio", {}).get("file_path")
        ]
        assert audio_paths
        assert all(Path(p).exists() for p in audio_paths)
