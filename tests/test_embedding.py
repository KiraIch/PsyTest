"""Embedding 引擎測試（Phase 2 啟用模型相關案例）。

模型尚未載入時，僅驗證不依賴權重的純函式；
需權重的案例在缺檔時自動跳過。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import find_existing_model_path
from src.perception.embedding_engine import EMBEDDING_DIM, EmbeddingEngine

MODEL_PATH = "models/embeddinggemma2-q8.gguf"


def _require_model() -> str:
    found = find_existing_model_path(MODEL_PATH)
    if found is None:
        pytest.skip("找不到 Embedding 模型檔，跳過需權重的測試")
    return str(found)


def test_embedding_dim_constant() -> None:
    """確認嵌入維度常數為 768。"""
    assert EMBEDDING_DIM == 768


def test_compute_similarity_identical() -> None:
    """相同向量的餘弦相似度應為 1。"""
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    vec = np.array([1.0, 2.0, 3.0])
    assert engine.compute_similarity(vec, vec) == pytest.approx(1.0)


def test_compute_similarity_orthogonal() -> None:
    """正交向量的餘弦相似度應為 0。"""
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    assert engine.compute_similarity(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(0.0)


def test_load_embedding_model() -> None:
    """確認能定位 embeddinggemma2 權重（Phase 2 完整載入）。"""
    _require_model()


def test_embed_text() -> None:
    """確認文字嵌入輸出維度正確（Phase 2 實作後啟用）。"""
    _require_model()
    pytest.skip("EmbeddingEngine.embed_text 排定於 Phase 2 實作")


def test_embed_audio() -> None:
    """確認音訊嵌入功能正常（Phase 2 實作後啟用）。"""
    _require_model()
    pytest.skip("EmbeddingEngine.embed_audio 排定於 Phase 2 實作")
