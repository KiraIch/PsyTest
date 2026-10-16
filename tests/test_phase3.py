"""Phase 3 智能層測試：情緒路由、節奏控制、人格管理、導演系統。"""

from __future__ import annotations

import pytest

from src.fast_think.emotion_router import EMOTION_MAP, EmotionRouter
from src.fast_think.laya_engine import PacingDecision
from src.fast_think.rhythm_controller import RhythmController, RhythmState
from src.slow_think.director import Director
from src.slow_think.persona_manager import (
    FRAMEWORK_STATEMENT,
    SAFETY_STATEMENT,
    Persona,
    PersonaManager,
)


# ---------------------------------------------------------------------------
# 情緒路由
# ---------------------------------------------------------------------------


def test_router_prefers_laya_emotion() -> None:
    """Laya 已判定情緒應優先採用。"""
    route = EmotionRouter().route("任何上下文", laya_emotion="anger", intensity=0.9)
    assert route.emotion == "anger"
    assert route.intensity == pytest.approx(0.9)
    assert route.voice_params["volume"] == 1.0


def test_router_keyword_classification_zh() -> None:
    """中文關鍵字分類應正確。"""
    router = EmotionRouter()
    assert router.route("我覺得好害怕，呼吸很急促").emotion == "fear"
    assert router.route("我什麼都做不好，算了吧").emotion == "shame"


def test_router_invalid_falls_back_neutral() -> None:
    """非法情緒與空狀態應退回 neutral。"""
    route = EmotionRouter().route("今天天氣很好", laya_emotion="不存在的情緒")
    assert route.emotion == "neutral"


def test_router_rejects_empty() -> None:
    """空上下文應拋出 ValueError。"""
    with pytest.raises(ValueError):
        EmotionRouter().route("   ")


def test_router_covers_all_mapped_emotions() -> None:
    """映射表八種情緒皆可路由且參數完整。"""
    router = EmotionRouter()
    for emotion in EMOTION_MAP:
        route = router.route("上下文", laya_emotion=emotion)
        assert route.emotion == emotion
        assert route.director_hint and route.voice_params and route.visual_params


def test_router_latency() -> None:
    """路由延遲應遠小於 10ms 目標（寬鬆斷言避免 CI 誤殺）。"""
    route = EmotionRouter().route("我很憤怒，這不公平")
    assert route.latency_ms < 500.0


# ---------------------------------------------------------------------------
# 節奏控制
# ---------------------------------------------------------------------------


def test_rhythm_full_cycle() -> None:
    """完整狀態機循環應正確推進並回到 IDLE。"""
    controller = RhythmController()
    assert controller.state == RhythmState.IDLE
    controller.advance("start_listening")
    controller.advance("utterance_done")
    controller.advance("decision_ready")
    controller.advance("pause_done")
    assert controller.advance("response_done") == RhythmState.IDLE


def test_rhythm_interrupt_path() -> None:
    """打斷路徑應可由 LISTENING 直達 RESPONDING。"""
    controller = RhythmController()
    controller.advance("start_listening")
    assert controller.advance("interrupt") == RhythmState.RESPONDING


def test_rhythm_invalid_transition_raises() -> None:
    """非法轉換應拋出 ValueError 並說明允許事件。"""
    controller = RhythmController()
    with pytest.raises(ValueError):
        controller.advance("pause_done")


def test_rhythm_plan_from_pacing() -> None:
    """節奏計畫應融合 Laya 決策與情緒乘數。"""
    controller = RhythmController()
    pacing = PacingDecision(
        pause_before_response=6.0,
        speech_speed=1.0,
        should_interrupt=False,
        response_length_hint="moderate",
    )
    plan = controller.plan_from_pacing(pacing, "sadness")
    assert plan.pause_before_response == pytest.approx(6.0)
    assert plan.speech_speed < 1.0  # 憂鬱乘數 0.9
    assert plan.ambient_for_silence is True  # 超過 5 秒
    assert plan.filler_sound == "sigh"


def test_rhythm_plan_clamps() -> None:
    """極端數值應被鉗制。"""
    controller = RhythmController()
    plan = controller.plan(pause=99.0, should_interrupt=False, speed=9.0)
    assert plan.pause_before_response <= 30.0
    assert plan.speech_speed <= 2.0


# ---------------------------------------------------------------------------
# 人格管理
# ---------------------------------------------------------------------------


def test_load_all_personas() -> None:
    """三份個案 YAML 皆應載入並通過驗證。"""
    manager = PersonaManager()
    for path in (
        "data/personas/mdd_severe.yaml",
        "data/personas/bpd_crisis.yaml",
        "data/personas/ptsd_combat.yaml",
    ):
        persona = manager.load_persona(path)
        assert manager.validate_persona(persona) is True
        assert persona.triggers


def test_load_missing_persona_raises() -> None:
    """缺失檔案應拋出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        PersonaManager().load_persona("data/personas/not_exist.yaml")


def test_validate_rejects_incomplete() -> None:
    """不完整個案應驗證失敗（不拋出例外）。"""
    assert PersonaManager().validate_persona(Persona(name="", age=-1)) is False


def test_system_prompt_contains_framework_and_safety() -> None:
    """系統提示必須包含框架聲明、安全聲明與個案資訊。"""
    manager = PersonaManager()
    persona = manager.load_persona("data/personas/mdd_severe.yaml")
    prompt = manager.build_system_prompt(persona, "初次會談")
    assert FRAMEWORK_STATEMENT in prompt
    assert SAFETY_STATEMENT in prompt
    assert persona.name in prompt
    assert "初次會談" in prompt


def test_response_style_ranges() -> None:
    """風格參數應在合法範圍。"""
    manager = PersonaManager()
    persona = manager.load_persona("data/personas/bpd_crisis.yaml")
    style = manager.get_response_style(persona, "anger")
    assert 0.5 <= float(style["speech_speed"]) <= 2.0
    assert style["filler_words"]


# ---------------------------------------------------------------------------
# 導演系統
# ---------------------------------------------------------------------------

_VALID_OUTPUT = (
    "我不知道該怎麼辦。"
    '<director>{"tone": "sorrowful", "volume": 0.6, "speed": 0.75, '
    '"pitch_shift": -0.05, "pauses": [[0.5, 1.2]], "breathing": "deep_sigh", '
    '"nonverbal": ["sniffle"], "body_language": ["cross_arms"]}</director>'
)


def test_director_parse_valid() -> None:
    """合法輸出應正確分離對白與指令。"""
    dialog, notes = Director().parse(_VALID_OUTPUT)
    assert dialog == "我不知道該怎麼辦。"
    assert notes.tone == "sorrowful"
    assert notes.volume == pytest.approx(0.6)
    assert notes.nonverbal == ["sniffle"]


def test_director_missing_uses_defaults() -> None:
    """缺失指令應套用預設值。"""
    dialog, notes = Director().parse("純對白，沒有指令。")
    assert dialog == "純對白，沒有指令。"
    assert notes.tone == "neutral"


def test_director_broken_json_uses_defaults() -> None:
    """損壞 JSON 應套用預設值而非中斷。"""
    dialog, notes = Director().parse("對白<director>{不是 json</director>")
    assert dialog == "對白"
    assert notes.tone == "neutral"


def test_director_clamps_ranges() -> None:
    """超範圍數值應被鉗制。"""
    _, notes = Director().parse('對白<director>{"volume": 9, "speed": 0.1}</director>')
    assert notes.volume <= 1.0
    assert notes.speed >= 0.5


def test_director_rejects_empty() -> None:
    """空輸入應拋出 ValueError。"""
    with pytest.raises(ValueError):
        Director().parse("   ")


def test_director_to_tts_tags() -> None:
    """標籤轉換應符合引擎格式。"""
    _, notes = Director().parse(_VALID_OUTPUT)
    orpheus = Director().to_tts_tags(notes, engine="orpheus")
    chattts = Director().to_tts_tags(notes, engine="chattts")
    assert "<cry>" in orpheus or "<breath>" in orpheus
    assert all(tag.startswith("[") for tag in chattts)
