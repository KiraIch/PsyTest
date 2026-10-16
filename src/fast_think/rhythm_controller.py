"""對話節奏控制器（Phase 3 實作）。

使用 Laya 的節奏決策控制對話的時間維度：停頓、打斷、沉默與語速變化。
狀態機：IDLE → LISTENING → PROCESSING → PAUSING → RESPONDING → IDLE。

規則：
    停頓期間產生填充音（呼吸聲、吞嚥聲）並記錄至標註。
    打斷（should_interrupt）允許在諮商師說話中途插入回應。
    長時間沉默（>5 秒）產生環境音並記錄持續時間。
    語速依情緒調整（焦慮快、憂鬱慢、解離平板）。

使用範例：
    from src.fast_think.laya_engine import PacingDecision
    from src.fast_think.rhythm_controller import RhythmController

    controller = RhythmController()
    controller.advance("start_listening")
    plan = controller.plan_from_pacing(
        PacingDecision(2.5, 0.7, False, "moderate"), "sadness"
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

LONG_SILENCE_SECONDS = 5.0


class RhythmState(str, Enum):
    """節奏狀態。"""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    PAUSING = "PAUSING"
    RESPONDING = "RESPONDING"


# （當前狀態，事件）→ 下一狀態。非法轉換拋出 ValueError。
_TRANSITIONS: Dict[Tuple[RhythmState, str], RhythmState] = {
    (RhythmState.IDLE, "start_listening"): RhythmState.LISTENING,
    (RhythmState.LISTENING, "utterance_done"): RhythmState.PROCESSING,
    (RhythmState.LISTENING, "interrupt"): RhythmState.RESPONDING,
    (RhythmState.PROCESSING, "decision_ready"): RhythmState.PAUSING,
    (RhythmState.PROCESSING, "no_response"): RhythmState.IDLE,
    (RhythmState.PAUSING, "pause_done"): RhythmState.RESPONDING,
    (RhythmState.RESPONDING, "response_done"): RhythmState.IDLE,
    (RhythmState.IDLE, "reset"): RhythmState.IDLE,
    (RhythmState.LISTENING, "reset"): RhythmState.IDLE,
    (RhythmState.PROCESSING, "reset"): RhythmState.IDLE,
    (RhythmState.PAUSING, "reset"): RhythmState.IDLE,
    (RhythmState.RESPONDING, "reset"): RhythmState.IDLE,
}

# 情緒到語速乘數（與 Laya 建議相乘後再鉗制）。
_EMOTION_SPEED_FACTOR: Dict[str, float] = {
    "anger": 1.1,
    "fear": 1.1,
    "manic": 1.2,
    "shame": 0.9,
    "sadness": 0.9,
    "dissociation": 0.95,
    "flat_affect": 1.0,
    "neutral": 1.0,
}

# 情緒到填充音。
_EMOTION_FILLER: Dict[str, str] = {
    "shame": "sigh",
    "sadness": "sigh",
    "fear": "breath_shallow",
    "anger": "breath_heavy",
    "dissociation": "silence",
    "manic": "none",
    "flat_affect": "none",
    "neutral": "breath",
}


@dataclass
class RhythmPlan:
    """單輪節奏計畫。"""

    pause_before_response: float = 0.0
    should_interrupt: bool = False
    speech_speed: float = 1.0
    filler_sound: str = "none"
    ambient_for_silence: bool = False
    events: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """正規化欄位。"""
        self.pause_before_response = max(0.0, min(30.0, self.pause_before_response))
        self.speech_speed = max(0.5, min(2.0, self.speech_speed))


class RhythmController:
    """對話節奏控制器。"""

    def __init__(self) -> None:
        """初始化為 IDLE 狀態。"""
        self.state: RhythmState = RhythmState.IDLE
        self.history: List[str] = []

    def reset(self) -> RhythmState:
        """回到 IDLE 狀態並清空歷史。"""
        self.state = RhythmState.IDLE
        self.history = []
        return self.state

    def advance(self, event: str) -> RhythmState:
        """依事件推進狀態機。

        Args:
            event: 事件名稱（例如 start_listening、pause_done）。

        Returns:
            推進後的狀態。

        Raises:
            ValueError: 非法狀態轉換時。
        """
        key = (self.state, event)
        if key not in _TRANSITIONS:
            raise ValueError(
                f"非法節奏轉換：{self.state.value} + {event}。"
                f"當前允許的事件：{sorted(e for s, e in _TRANSITIONS if s == self.state)}"
            )
        self.state = _TRANSITIONS[key]
        self.history.append(f"{key[0].value}+{event}->{self.state.value}")
        return self.state

    def plan(self, pause: float, should_interrupt: bool, speed: float) -> RhythmPlan:
        """由原始數值產生節奏計畫（相容舊介面）。

        Args:
            pause: 停頓秒數。
            should_interrupt: 是否打斷。
            speed: 語速。

        Returns:
            :class:`RhythmPlan`。
        """
        return RhythmPlan(
            pause_before_response=pause,
            should_interrupt=bool(should_interrupt),
            speech_speed=speed,
            filler_sound="breath" if pause >= 1.0 else "none",
            ambient_for_silence=pause > LONG_SILENCE_SECONDS,
        )

    def plan_from_pacing(
        self, pacing, emotion: str = "neutral"
    ) -> RhythmPlan:
        """由 Laya 節奏決策與情緒產生執行計畫。

        Args:
            pacing: :class:`PacingDecision`（需含 pause_before_response、
                speech_speed、should_interrupt 屬性）。
            emotion: 情緒標籤，影響語速乘數與填充音。

        Returns:
            :class:`RhythmPlan`。
        """
        factor = _EMOTION_SPEED_FACTOR.get(emotion, 1.0)
        try:
            speed = float(pacing.speech_speed) * factor
        except (AttributeError, TypeError, ValueError):
            speed = 1.0
        try:
            pause = float(pacing.pause_before_response)
        except (AttributeError, TypeError, ValueError):
            pause = 0.0
        try:
            interrupt = bool(pacing.should_interrupt)
        except AttributeError:
            interrupt = False
        filler = _EMOTION_FILLER.get(emotion, "breath")
        if pause < 0.5 and filler not in {"none", "silence"}:
            filler = "none"
        return RhythmPlan(
            pause_before_response=pause,
            should_interrupt=interrupt,
            speech_speed=speed,
            filler_sound=filler,
            ambient_for_silence=pause > LONG_SILENCE_SECONDS,
            events=[f"emotion={emotion}", f"factor={factor}"],
        )
