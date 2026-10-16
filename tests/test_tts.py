"""TTS 引擎測試（Phase 4 啟用）。"""

from __future__ import annotations

import pytest

from src.voice.voice_director import EMOTION_TO_VOICE


def test_emotion_mapping_covers_required() -> None:
    """確認情緒映射表涵蓋規格要求的六種情緒。"""
    for emotion in ("shame", "anger", "fear", "dissociation", "manic", "flat_affect"):
        assert emotion in EMOTION_TO_VOICE


def test_tts_synthesis() -> None:
    """語音合成（Phase 4 實作後啟用）。"""
    pytest.skip("TTS 引擎排定於 Phase 4 實作")
