"""Laya 決策引擎測試（Phase 2 啟用模型相關案例）。"""

from __future__ import annotations

import pytest

from src.config import find_existing_model_path

MODEL_PATH = "models/laya-421m-bf16.gguf"


def _require_model() -> str:
    found = find_existing_model_path(MODEL_PATH)
    if found is None:
        pytest.skip("找不到 Laya 模型檔，跳過需權重的測試")
    return str(found)


def test_load_laya() -> None:
    """確認能定位 Laya 權重（Phase 2 完整載入）。"""
    _require_model()


def test_decision_speed() -> None:
    """確認穩態決策延遲達標（暖機後應小於 2 秒，目標 50ms）。"""
    pytest.importorskip("laya", reason="缺少官方 laya 套件，跳過效能測試")
    from src.fast_think.laya_engine import LayaEngine

    with LayaEngine() as engine:
        assert engine.active_backend == "laya"
        engine.should_respond("暖機。")  # 首次建模約需數秒
        import time

        started = time.perf_counter()
        decision = engine.should_respond("諮商師：最近睡得好嗎？")
        elapsed = time.perf_counter() - started
        assert decision.emotion_tag in (
            "shame", "anger", "fear", "dissociation",
            "manic", "flat_affect", "sadness", "neutral",
        )
        assert elapsed < 2.0, f"決策延遲 {elapsed:.3f}s 超標"


def test_official_backend_decisions() -> None:
    """官方後端四種決策皆應回傳合法結構。"""
    pytest.importorskip("laya", reason="缺少官方 laya 套件")
    from src.fast_think.laya_engine import LayaEngine

    with LayaEngine() as engine:
        emotion = engine.determine_emotion("個案低頭沉默", "低落")
        assert emotion.emotion in (
            "shame", "anger", "fear", "dissociation",
            "manic", "flat_affect", "sadness", "neutral",
        )
        pacing = engine.determine_pacing("諮商師：多說一點好嗎？", 3)
        assert 0.0 <= pacing.pause_before_response <= 6.0
        crisis = engine.assess_crisis_level("我覺得活著沒有意義")
        assert 0 <= crisis.level <= 5


def test_emotion_routing() -> None:
    """確認情緒路由輸出格式正確（Phase 3 已實作，見 test_phase3.py）。"""
    from src.fast_think.emotion_router import EmotionRouter

    route = EmotionRouter().route("我很害怕", laya_emotion="fear")
    assert route.emotion == "fear"
    assert route.voice_params and route.visual_params


def test_parse_decision_output() -> None:
    """確認決策解析正確（不需模型）。"""
    from src.fast_think.laya_engine import parse_decision_output

    result = parse_decision_output("YES urgency 0.8 emotion fear")
    assert result.should_speak is True
    assert result.urgency == pytest.approx(0.8)
    assert result.emotion_tag == "fear"


def test_parse_decision_output_safe_default() -> None:
    """確認無法解析時退回安全預設（回應）。"""
    from src.fast_think.laya_engine import parse_decision_output

    result = parse_decision_output("完全無法理解的輸出")
    assert result.should_speak is True
    assert 0.0 <= result.urgency <= 1.0


def test_parse_emotion_output() -> None:
    """確認情緒解析正確（不需模型）。"""
    from src.fast_think.laya_engine import parse_emotion_output

    result = parse_emotion_output("emotion shame intensity 0.9 defense withdrawal")
    assert result.emotion == "shame"
    assert result.intensity == pytest.approx(0.9)


def test_parse_pacing_output_bounds() -> None:
    """確認節奏數值被鉗制在合法範圍（不需模型）。"""
    from src.fast_think.laya_engine import parse_pacing_output

    result = parse_pacing_output("pause 99 speed 9 length detailed")
    assert result.pause_before_response <= 30.0
    assert result.speech_speed <= 2.0
    assert result.response_length_hint == "detailed"


def test_keyword_crisis_scan_conservative() -> None:
    """確認關鍵字掃描採保守原則（不需模型）。"""
    from src.fast_think.laya_engine import keyword_crisis_scan

    assert keyword_crisis_scan("今天天氣很好").level == 0
    assert keyword_crisis_scan("我最近撐不下去，好痛苦").level >= 2
    assert keyword_crisis_scan("我想自殺").level == 5


def test_engine_validates_input_without_model() -> None:
    """確認輸入驗證在模型載入前即生效（不需模型）。"""
    from src.fast_think.laya_engine import LayaEngine

    engine = LayaEngine.__new__(LayaEngine)
    with pytest.raises(ValueError):
        engine.should_respond("   ")
    with pytest.raises(ValueError):
        engine.assess_crisis_level("")
