"""Orpheus TTS 後端（受限實作，預設停用）。

重要發現（2026-10-16 實測）：PyPI 的 ``orpheus-tts`` 0.1.6 是雲端
WebSocket 客戶端，需連外網並可能需要 API key，並非原規格假設的
本地 MLX 模型。這與本專案本地端原則衝突，因此：

    1. 本後端預設不可用，不參與自動選擇。
    2. 僅在明確指定且已配置端點（環境變數 ``ORPHEUS_WS_URL`` 或
       ``ORPHEUS_API_KEY``）時啟用。
    3. 本地輕量需求改由 Kokoro 後端承擔（見 ``kokoro_backend.py``
       與 HANDOVER.md ADR）。

標籤轉換（``Director.to_tts_tags`` 的 ``<tag>`` 格式）保留完整，
啟用後可直接使用。
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from src.voice.tts_engine import AudioOutput, TTSEngine


class OrpheusBackend(TTSEngine):
    """Orpheus TTS 實作（需雲端端點配置）。"""

    name = "orpheus"

    def __init__(self, voice: str = "default", sample_rate: int = 24000) -> None:
        """初始化後端參數。

        Args:
            voice: 音色名稱。
            sample_rate: 目標取樣率。

        Raises:
            ValueError: 參數非法時。
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate 必須為正整數")
        self.voice = voice
        self.sample_rate = sample_rate

    def unavailable_reason(self) -> str:
        """回傳不可用原因。"""
        try:
            import orpheus_tts  # noqa: F401
        except ImportError:
            return "缺少 orpheus-tts，請執行 pip install orpheus-tts"
        if not os.getenv("ORPHEUS_WS_URL") and not os.getenv("ORPHEUS_API_KEY"):
            return (
                "Orpheus 為雲端服務，需設定 ORPHEUS_WS_URL 或 ORPHEUS_API_KEY。"
                "本地使用請改用 kokoro 或 chattts 引擎"
            )
        return ""

    def is_available(self) -> bool:
        """僅在套件存在且已配置端點時可用。"""
        return self.unavailable_reason() == ""

    def get_voice_options(self) -> List[str]:
        """回傳音色清單（未配置時為空）。"""
        if not self.is_available():
            return []
        try:
            from orpheus_tts import VOICES

            return list(VOICES)
        except Exception:  # noqa: BLE001
            return [self.voice]

    def synthesize(self, text: str, director_notes: Dict[str, object]) -> AudioOutput:
        """經雲端合成語音（需先配置端點）。

        Raises:
            ValueError: 文本為空時。
            RuntimeError: 未配置或合成失敗時。
        """
        if not text or not text.strip():
            raise ValueError("輸入文本不可為空")
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"Orpheus 不可用：{reason}")
        raise RuntimeError(
            "Orpheus 雲端串流尚未接線：orpheus-tts 客戶端 API 需依官方文件"
            "確認串流參數後方可啟用。目前請改用 kokoro 或 chattts 引擎"
        )

    def _empty(self, text: str) -> AudioOutput:
        """回傳空佔位（保留供未來接線使用）。"""
        return AudioOutput(
            audio_array=np.zeros(0, dtype=np.float32),
            sample_rate=self.sample_rate,
            duration=0.0,
            metadata={"engine": "orpheus", "text": text},
        )
