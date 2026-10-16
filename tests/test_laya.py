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
    """確認決策延遲小於 100ms（需 llama-cpp，否則跳過）。"""
    pytest.importorskip("llama_cpp", reason="缺少 llama-cpp-python，跳過效能測試")
    _require_model()
    pytest.skip("效能基準待 Phase 2 收尾量測")


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
