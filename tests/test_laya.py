"""Laya 決策引擎測試（Phase 2 啟用模型相關案例）。"""

from __future__ import annotations

import pytest

from src.config import find_existing_model_path

MODEL_PATH = "models/laya-421m-bf16.gguf"


def _require_model() -> str:
    found = find_existing_model_path(MODEL_PATH)
    if found is None:
        pytest.skip("找不到 Laya 模型檔，跳過需權重的測試")
    return str(found)


def test_load_laya() -> None:
    """確認能定位 Laya 權重（Phase 2 完整載入）。"""
    _require_model()


def test_decision_speed() -> None:
    """確認決策延遲小於 100ms（Phase 2 實作後啟用）。"""
    _require_model()
    pytest.skip("LayaEngine 排定於 Phase 2 實作")


def test_emotion_routing() -> None:
    """確認情緒路由輸出格式正確（Phase 3 實作後啟用）。"""
    pytest.skip("EmotionRouter 排定於 Phase 3 實作")
