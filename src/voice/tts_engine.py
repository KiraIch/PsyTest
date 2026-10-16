"""TTS 抽象層（骨架，Phase 4 實作）。

自動選擇邏輯：Orpheus（macOS MLX 優先）→ ChatTTS（ONNX 跨平台）
→ pyttsx3（系統內建保底）。所有引擎均須解析導演標籤：
hesitate、cry、whisper、angry、flat、breath。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class AudioOutput:
    """語音合成輸出。"""

    audio_array: np.ndarray
    sample_rate: int
    duration: float
    metadata: Dict[str, object] = field(default_factory=dict)


class TTSEngine(ABC):
    """TTS 引擎抽象基類。"""

    @abstractmethod
    def synthesize(self, text: str, director_notes: Dict[str, object]) -> AudioOutput:
        """合成語音（Phase 4 實作）。"""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """引擎是否可用（Phase 4 實作）。"""
        raise NotImplementedError

    @abstractmethod
    def get_voice_options(self) -> List[str]:
        """可用音色清單（Phase 4 實作）。"""
        raise NotImplementedError


def select_engine(preference: str = "auto") -> TTSEngine:
    """依偏好與平台自動選擇引擎（Phase 4 實作）。"""
    raise NotImplementedError("select_engine 排定於 Phase 4 實作")
