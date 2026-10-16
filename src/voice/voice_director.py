"""語音導演（骨架，Phase 4 實作）。

將導演指令轉換為各 TTS 引擎參數：
    shame → volume=0.6, speed=0.7, pitch_shift=-10%
    anger → volume=1.0, speed=1.3, pitch_shift=+5%
    fear → volume=0.7, speed=1.1, pitch_shift=+15%, tremor=0.3
    dissociation → volume=0.5, speed=0.8, monotone=True
    manic → volume=0.9, speed=1.5, pitch_variance=high
"""

from __future__ import annotations

from typing import Dict

EMOTION_TO_VOICE: Dict[str, Dict[str, object]] = {
    "shame": {"volume": 0.6, "speed": 0.7, "pitch_shift": -0.10},
    "anger": {"volume": 1.0, "speed": 1.3, "pitch_shift": 0.05},
    "fear": {"volume": 0.7, "speed": 1.1, "pitch_shift": 0.15, "tremor": 0.3},
    "dissociation": {"volume": 0.5, "speed": 0.8, "pitch_shift": 0.0, "monotone": True},
    "manic": {"volume": 0.9, "speed": 1.5, "pitch_variance": "high"},
    "flat_affect": {"volume": 0.5, "speed": 0.9, "pitch_shift": 0.0, "monotone": True},
}


class VoiceDirector:
    """語音導演（Phase 4 實作）。"""

    def to_engine_params(
        self, director_notes: Dict[str, object], engine: str = "orpheus"
    ) -> Dict[str, object]:
        """轉換導演指令為引擎參數（Phase 4 實作）。"""
        raise NotImplementedError("VoiceDirector.to_engine_params 排定於 Phase 4 實作")
