"""視覺/表情特徵分析（骨架，Phase 3 實作）。

無頭模式下接收表情參數而非實際圖片；若提供實際圖片，
則使用 EmbeddingGemma 2 視覺編碼器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VisualFeatures:
    """視覺特徵結構。"""

    gaze_direction: str = "direct"
    facial_expression: str = ""
    body_posture: str = "open"
    micro_expressions: List[str] = field(default_factory=list)
    arousal_level: float = 0.0


class VisualAnalyzer:
    """視覺特徵分析器（Phase 3 實作）。"""

    def analyze(self, params: Dict[str, object]) -> VisualFeatures:
        """由表情參數推斷視覺特徵（Phase 3 實作）。"""
        raise NotImplementedError("VisualAnalyzer.analyze 排定於 Phase 3 實作")
