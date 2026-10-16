"""情緒路由器（Phase 3 實作）。

根據對話上下文決定個案的情緒反應路徑，並映射為三組參數：
導演指令、語音參數、肢體參數。決策為純查表與關鍵字匹配，
目標 <10ms，適合每輪呼叫。

情緒來源有二（優先順序由高至低）：
    1. 快思層 Laya 已判定的情緒（``laya_emotion`` 參數）。
    2. 本模組的關鍵字快速分類（中英對照）。

使用範例：
    from src.fast_think.emotion_router import EmotionRouter

    router = EmotionRouter()
    route = router.route("個案低頭沉默，聲音顫抖", "fear")
    print(route.voice_params)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

VALID_EMOTIONS = (
    "shame",
    "anger",
    "fear",
    "dissociation",
    "manic",
    "flat_affect",
    "sadness",
    "neutral",
)

# 情緒到三組參數的映射（規格書定義，數值含義見各消費者模組）。
EMOTION_MAP: Dict[str, Dict[str, Dict[str, object]]] = {
    "shame": {
        "director_hint": {"tone": "subdued", "pauses": "frequent"},
        "voice_params": {"volume": 0.6, "speed": 0.7, "pitch_shift": -0.10},
        "visual_params": {
            "gaze_direction": "down",
            "body_posture": "closed",
            "arousal_level": 0.4,
        },
    },
    "anger": {
        "director_hint": {"tone": "harsh", "pauses": "few"},
        "voice_params": {"volume": 1.0, "speed": 1.3, "pitch_shift": 0.05},
        "visual_params": {
            "gaze_direction": "direct",
            "body_posture": "defensive",
            "arousal_level": 0.9,
        },
    },
    "fear": {
        "director_hint": {"tone": "trembling", "breathing": "shallow_rapid"},
        "voice_params": {
            "volume": 0.7,
            "speed": 1.1,
            "pitch_shift": 0.15,
            "tremor": 0.3,
        },
        "visual_params": {
            "gaze_direction": "aversion_left",
            "body_posture": "defensive",
            "arousal_level": 0.85,
        },
    },
    "dissociation": {
        "director_hint": {"tone": "hollow", "pauses": "long"},
        "voice_params": {
            "volume": 0.5,
            "speed": 0.8,
            "pitch_shift": 0.0,
            "monotone": True,
        },
        "visual_params": {
            "gaze_direction": "down",
            "body_posture": "closed",
            "arousal_level": 0.2,
        },
    },
    "manic": {
        "director_hint": {"tone": "pressured", "pauses": "none"},
        "voice_params": {"volume": 0.9, "speed": 1.5, "pitch_variance": "high"},
        "visual_params": {
            "gaze_direction": "up",
            "body_posture": "agitated",
            "arousal_level": 1.0,
        },
    },
    "flat_affect": {
        "director_hint": {"tone": "mechanical", "pauses": "even"},
        "voice_params": {
            "volume": 0.5,
            "speed": 0.9,
            "pitch_shift": 0.0,
            "monotone": True,
        },
        "visual_params": {
            "gaze_direction": "direct",
            "body_posture": "closed",
            "arousal_level": 0.1,
        },
    },
    "sadness": {
        "director_hint": {"tone": "sorrowful", "breathing": "deep_sigh"},
        "voice_params": {"volume": 0.6, "speed": 0.75, "pitch_shift": -0.05},
        "visual_params": {
            "gaze_direction": "down",
            "body_posture": "closed",
            "arousal_level": 0.35,
        },
    },
    "neutral": {
        "director_hint": {"tone": "neutral", "pauses": "normal"},
        "voice_params": {"volume": 0.8, "speed": 1.0, "pitch_shift": 0.0},
        "visual_params": {
            "gaze_direction": "direct",
            "body_posture": "open",
            "arousal_level": 0.5,
        },
    },
}

# 關鍵字快速分類（中英對照），用於 Laya 未提供情緒時。
_KEYWORD_EMOTIONS: List[tuple[str, tuple[str, ...]]] = [
    ("shame", ("羞愧", "丟臉", "羞恥", "沒用", "做不好", "對不起", "shame", "ashamed")),
    ("anger", ("憤怒", "生氣", "火大", "不公平", "anger", "angry", "furious")),
    ("fear", ("害怕", "恐懼", "擔心", "緊張", "fear", "afraid", "anxious")),
    ("dissociation", ("解離", "空洞", "麻木", "不真實", "dissoci", "numb", "empty")),
    ("manic", ("興奮", "停不下來", "躁", "manic", "euphoric", "racing")),
    ("flat_affect", ("平板", "面無表情", "flat", "monotone", "mechanical")),
    ("sadness", ("難過", "悲傷", "哭", "沮喪", "sad", "sorrow", "grief", "cry")),
]


@dataclass(frozen=True)
class EmotionRoute:
    """情緒路由結果。"""

    emotion: str
    intensity: float
    director_hint: Dict[str, object] = field(default_factory=dict)
    voice_params: Dict[str, object] = field(default_factory=dict)
    visual_params: Dict[str, object] = field(default_factory=dict)
    latency_ms: float = 0.0


def classify_by_keywords(text: str) -> str:
    """以關鍵字快速分類情緒，命中多個時取第一個，皆無時回傳 neutral。

    Args:
        text: 對話上下文（中英皆可）。

    Returns:
        情緒標籤。
    """
    lowered = text.lower()
    for emotion, keywords in _KEYWORD_EMOTIONS:
        for keyword in keywords:
            if keyword in lowered or keyword in text:
                return emotion
    return "neutral"


class EmotionRouter:
    """情緒路由器。"""

    def route(
        self,
        context: str,
        client_state: str = "",
        laya_emotion: Optional[str] = None,
        intensity: float = 0.5,
    ) -> EmotionRoute:
        """執行緒路由決策。

        Args:
            context: 對話上下文，不可為空。
            client_state: 個案狀態描述（輔助分類，可空）。
            laya_emotion: Laya 已判定情緒，優先採用；非法值視為未提供。
            intensity: 情緒強度，會被鉗制至 [0, 1]。

        Returns:
            :class:`EmotionRoute`，必為合法情緒，絕不拋出例外。

        Raises:
            ValueError: context 為空時。
        """
        if not context or not context.strip():
            raise ValueError("context 不可為空")
        started = time.perf_counter()
        try:
            if laya_emotion in EMOTION_MAP:
                emotion = str(laya_emotion)
            else:
                emotion = classify_by_keywords(f"{context} {client_state}")
                if emotion not in EMOTION_MAP:
                    emotion = "neutral"
            try:
                clamped = float(intensity)
            except (TypeError, ValueError):
                clamped = 0.5
            clamped = max(0.0, min(1.0, clamped))
            mapped = EMOTION_MAP[emotion]
            return EmotionRoute(
                emotion=emotion,
                intensity=clamped,
                director_hint=dict(mapped["director_hint"]),
                voice_params=dict(mapped["voice_params"]),
                visual_params=dict(mapped["visual_params"]),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception:  # noqa: BLE001
            mapped = EMOTION_MAP["neutral"]
            return EmotionRoute(
                emotion="neutral",
                intensity=0.5,
                director_hint=dict(mapped["director_hint"]),
                voice_params=dict(mapped["voice_params"]),
                visual_params=dict(mapped["visual_params"]),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
