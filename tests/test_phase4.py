"""Phase 4 聲音層測試：導演轉換、標籤解析、引擎選擇與實際合成。"""

from __future__ import annotations

import pytest

from src.slow_think.director import DirectorNotes
from src.voice.tts_engine import (
    list_available_engines,
    parse_inline_tags,
    select_engine,
    strip_inline_tags,
)
from src.voice.voice_director import VoiceDirector


# ---------------------------------------------------------------------------
# 語音導演
# ---------------------------------------------------------------------------


def test_director_covers_all_emotions() -> None:
    """六種情緒皆可轉換且數值合法。"""
    director = VoiceDirector()
    for emotion in ("shame", "anger", "fear", "dissociation", "manic", "flat_affect"):
        for engine in ("chattts", "kokoro", "pyttsx3", "orpheus"):
            params = director.to_engine_params(
                {"volume": 0.8, "speed": 1.0}, engine, emotion=emotion
            )
            assert isinstance(params, dict) and params


def test_director_explicit_values_win() -> None:
    """明確指定的數值應覆寫情緒基底。"""
    params = VoiceDirector().to_engine_params(
        {"volume": 0.3, "speed": 1.8}, "pyttsx3", emotion="anger"
    )
    assert params["volume"] == pytest.approx(0.3)
    assert params["rate"] == int(200 * 1.8)


def test_director_accepts_notes_object() -> None:
    """應接受 DirectorNotes 物件。"""
    notes = DirectorNotes(tone="sorrowful", volume=0.6, speed=0.75)
    params = VoiceDirector().to_engine_params(notes, "chattts")
    assert params["volume"] == pytest.approx(0.6)


def test_director_rejects_unknown_engine() -> None:
    """未知引擎應拋出 ValueError。"""
    with pytest.raises(ValueError):
        VoiceDirector().to_engine_params({}, "不存在的引擎")


def test_director_chattts_break_capped() -> None:
    """break 參數應鉗制在 0 至 7。"""
    params = VoiceDirector().to_engine_params(
        {"pauses": [[0.0, 20.0]]}, "chattts", emotion="neutral"
    )
    assert 0 <= params["break"] <= 7


# ---------------------------------------------------------------------------
# 標籤解析
# ---------------------------------------------------------------------------


def test_parse_both_tag_styles() -> None:
    """兩種標籤寫法皆應解析並移除。"""
    clean, tags = parse_inline_tags("你好<hesitate>，最近[breath]好嗎？")
    assert clean == "你好，最近好嗎？"
    assert set(tags) == {"hesitate", "breath"}


def test_strip_stage_directions() -> None:
    """舞台指示應移除，對白保留。"""
    from src.voice.tts_engine import ENGINE_ORDER, strip_stage_directions

    assert strip_stage_directions("（低頭）你好嗎？") == "你好嗎？"
    assert strip_stage_directions("Hello (smiling) world") == "Hello world"
    assert strip_stage_directions("純對白。") == "純對白。"
    assert ENGINE_ORDER[0] == "chattts"


def test_parse_preserves_chattts_native_tags() -> None:
    """ChatTTS 原生標籤應保留。"""
    clean, _ = parse_inline_tags("你好[laugh]，最近好嗎？")
    assert "[laugh]" in clean
    assert strip_inline_tags("純文本") == "純文本"


# ---------------------------------------------------------------------------
# 引擎選擇
# ---------------------------------------------------------------------------


def test_select_pyttsx3_when_available() -> None:
    """pyttsx3 已安裝時應可明確選用。"""
    pytest.importorskip("pyttsx3", reason="缺少 pyttsx3")
    engine = select_engine("pyttsx3")
    assert engine.is_available()
    assert engine.name == "pyttsx3"


def test_select_orpheus_unavailable_by_default() -> None:
    """Orpheus 預設應不可用並給出原因。"""
    with pytest.raises(RuntimeError):
        select_engine("orpheus")


def test_select_unknown_engine_raises() -> None:
    """未知引擎名稱應拋出 ValueError。"""
    with pytest.raises(ValueError):
        select_engine("不存在的引擎")


def test_list_available_reports_all() -> None:
    """可用性探測應涵蓋四個引擎。"""
    status = list_available_engines()
    assert set(status) == {"kokoro", "chattts", "pyttsx3", "orpheus"}
    assert status["orpheus"] is False


# ---------------------------------------------------------------------------
# 實際合成（需依賴與模型，缺失時跳過）
# ---------------------------------------------------------------------------


def test_kokoro_synthesize_zh() -> None:
    """Kokoro 中文合成 smoke test。"""
    pytest.importorskip("sherpa_onnx", reason="缺少 sherpa-onnx")
    from src.voice.kokoro_backend import KokoroBackend

    engine = KokoroBackend()
    if not engine.is_available():
        pytest.skip(f"Kokoro 模型未就緒：{engine.unavailable_reason()}")
    output = engine.synthesize("你好，很高興見到你。", {})
    assert output.sample_rate == 24000
    assert output.duration > 0.5
    assert float(abs(output.audio_array).max()) > 0.01


def test_chattts_synthesize_emotion() -> None:
    """ChatTTS 情感合成 smoke test（CPU 較慢）。"""
    pytest.importorskip("ChatTTS", reason="缺少 ChatTTS")
    from src.voice.chatts_backend import ChatTTSBackend

    engine = ChatTTSBackend()
    output = engine.synthesize("你好[uv_break]，最近好嗎？", {"oral": 2})
    assert output.sample_rate == 24000
    assert output.duration > 0.5


def test_pyttsx3_synthesize() -> None:
    """pyttsx3 保底合成 smoke test。"""
    pytest.importorskip("pyttsx3", reason="缺少 pyttsx3")
    from src.voice.pyttsx3_backend import Pyttsx3Backend

    engine = Pyttsx3Backend()
    output = engine.synthesize("測試語音。", {})
    assert output.duration > 0
    assert output.audio_array.size > 0


def test_pyttsx3_consecutive_synthesis() -> None:
    """同一實例連續合成皆須有音訊（驅動重用缺陷回歸測試）。"""
    pytest.importorskip("pyttsx3", reason="缺少 pyttsx3")
    from src.voice.pyttsx3_backend import Pyttsx3Backend

    engine = Pyttsx3Backend()
    for i in range(3):
        output = engine.synthesize(f"第{i + 1}次測試語音。", {})
        assert output.audio_array.size > 0, f"第 {i + 1} 次合成為空"


def test_synthesize_rejects_empty() -> None:
    """空文本應拋出 ValueError（不需模型）。"""
    from src.voice.kokoro_backend import KokoroBackend

    with pytest.raises(ValueError):
        KokoroBackend().synthesize("   ", {})
