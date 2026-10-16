"""TTS 引擎測試（Phase 4 啟用）。"""

from __future__ import annotations

import pytest

from src.voice.voice_director import EMOTION_TO_VOICE


def test_emotion_mapping_covers_required() -> None:
    """確認情緒映射表涵蓋規格要求的六種情緒。"""
    for emotion in ("shame", "anger", "fear", "dissociation", "manic", "flat_affect"):
        assert emotion in EMOTION_TO_VOICE


def test_tts_synthesis() -> None:
    """保底引擎合成 smoke test（Phase 4 已實作，見 test_phase4.py）。"""
    pytest.importorskip("pyttsx3", reason="缺少 pyttsx3")
    from src.voice.pyttsx3_backend import Pyttsx3Backend

    output = Pyttsx3Backend().synthesize("測試語音。", {})
    assert output.duration > 0
