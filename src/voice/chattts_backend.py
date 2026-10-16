"""ChatTTS 後端（骨架，Phase 4 實作）。

預定為跨平台 ONNX 選項，接收 [tag] 格式與語氣詞插入。
"""

from __future__ import annotations

from typing import Dict, List

from src.voice.tts_engine import AudioOutput, TTSEngine


class ChatTTSBackend(TTSEngine):
    """ChatTTS 實作（Phase 4 實作）。"""

    def synthesize(self, text: str, director_notes: Dict[str, object]) -> AudioOutput:
        """合成語音（Phase 4 實作）。"""
        raise NotImplementedError("ChatTTSBackend.synthesize 排定於 Phase 4 實作")

    def is_available(self) -> bool:
        """檢查可用性（Phase 4 實作）。"""
        raise NotImplementedError("ChatTTSBackend.is_available 排定於 Phase 4 實作")

    def get_voice_options(self) -> List[str]:
        """音色清單（Phase 4 實作）。"""
        raise NotImplementedError("ChatTTSBackend.get_voice_options 排定於 Phase 4 實作")
