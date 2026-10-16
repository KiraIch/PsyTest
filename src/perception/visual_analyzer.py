"""視覺/表情特徵分析（Phase 5 實作，無頭模式）。

無頭模式下接收表情參數而非實際圖片，負責驗證並正規化為
結構化視覺特徵。若提供情緒標籤，可據此補齊缺漏欄位。
實際圖片輸入待 EmbeddingGemma 2 視覺編碼器（Phase 7）。

使用範例：
    from src.perception.visual_analyzer import VisualAnalyzer

    features = VisualAnalyzer().analyze(
        {"gaze_direction": "down"}, emotion="shame"
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

VALID_GAZE = ("direct", "aversion_left", "aversion_right", "down", "up")
VALID_POSTURE = ("open", "closed", "defensive", "agitated")

# 情緒到視覺預設的映射（與 EmotionRouter 視覺參數一致）。
_EMOTION_VISUAL: Dict[str, Dict[str, object]] = {
    "shame": {"gaze_direction": "down", "body_posture": "closed", "arousal_level": 0.4},
    "anger": {"gaze_direction": "direct", "body_posture": "defensive", "arousal_level": 0.9},
    "fear": {"gaze_direction": "aversion_left", "body_posture": "defensive", "arousal_level": 0.85},
    "dissociation": {"gaze_direction": "down", "body_posture": "closed", "arousal_level": 0.2},
    "manic": {"gaze_direction": "up", "body_posture": "agitated", "arousal_level": 1.0},
    "flat_affect": {"gaze_direction": "direct", "body_posture": "closed", "arousal_level": 0.1},
    "sadness": {"gaze_direction": "down", "body_posture": "closed", "arousal_level": 0.35},
    "neutral": {"gaze_direction": "direct", "body_posture": "open", "arousal_level": 0.5},
}


@dataclass
class VisualFeatures:
    """視覺特徵結構。"""

    gaze_direction: str = "direct"
    facial_expression: str = ""
    body_posture: str = "open"
    micro_expressions: List[str] = field(default_factory=list)
    arousal_level: float = 0.0


class VisualAnalyzer:
    """視覺特徵分析器（無頭模式）。"""

    def analyze(
        self, params: Dict[str, object], emotion: str = ""
    ) -> VisualFeatures:
        """由表情參數推斷視覺特徵。

        參數缺漏時以情緒預設補齊（無情緒時用中性預設），
        非法枚舉值視為缺漏。

        Args:
            params: 表情參數字典（blendshapes 或路由器輸出皆可）。
            emotion: 情緒標籤（輔助補齊，可空）。

        Returns:
            :class:`VisualFeatures`。

        Raises:
            ValueError: params 非字典時。
        """
        if not isinstance(params, dict):
            raise ValueError("params 必須為字典")
        fallback = _EMOTION_VISUAL.get(emotion, _EMOTION_VISUAL["neutral"])
        gaze = params.get("gaze_direction", fallback["gaze_direction"])
        if gaze not in VALID_GAZE:
            gaze = fallback["gaze_direction"]
        posture = params.get("body_posture", params.get("body_language", fallback["body_posture"]))
        if posture not in VALID_POSTURE:
            posture = fallback["body_posture"]
        try:
            arousal = max(0.0, min(1.0, float(params.get("arousal_level", fallback["arousal_level"]))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            arousal = float(fallback["arousal_level"])
        expression = params.get("facial_expression", "")
        micro = params.get("micro_expressions", [])
        micro_list = list(micro) if isinstance(micro, list) else []
        return VisualFeatures(
            gaze_direction=str(gaze),
            facial_expression=str(expression or ""),
            body_posture=str(posture),
            micro_expressions=[str(m) for m in micro_list],
            arousal_level=arousal,
        )
