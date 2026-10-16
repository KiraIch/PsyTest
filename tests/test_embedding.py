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
    """確認文字嵌入輸出維度正確（架構不支援時明確跳過）。"""
    pytest.importorskip("llama_cpp", reason="缺少 llama-cpp-python，跳過需權重的測試")
    from src.perception.embedding_engine import EmbeddingEngine

    try:
        with EmbeddingEngine(_require_model()) as engine:
            vec = engine.embed_text("個案表達強烈的無助感")
    except RuntimeError as exc:
        if "尚不支援" in str(exc):
            pytest.skip(f"llama.cpp 尚不支援該 GGUF 架構：{exc}")
        raise
    assert vec.shape[0] == EMBEDDING_DIM


def test_st_backend_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """ST 套件缺失時應給出安裝指引。"""
    import sys

    from src.perception.embedding_engine import EmbeddingEngine

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError):
        EmbeddingEngine("models/embeddinggemma2-q8.gguf", backend="st").load()


def test_auto_falls_back_to_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto 模式無 Ollama 且 ST 缺失時應回退 GGUF（此處架構不支援故報錯）。"""
    import sys

    from src.perception.embedding_engine import EmbeddingEngine
    from src.slow_think import ollama_client

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(ollama_client.OllamaClient, "list_models", lambda self: [])
    engine = EmbeddingEngine(_require_model())
    assert engine.backend == "auto"
    with pytest.raises(RuntimeError, match="尚不支援"):
        engine.load()


def test_st_path_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """ST 路徑應送查詢前綴並回傳 768 維向量（假模組，不需下載）。"""
    import sys
    import types

    import numpy as np

    from src.perception.embedding_engine import EmbeddingEngine

    seen: dict = {}

    class _FakeST:
        def __init__(self, model_id: str) -> None:
            seen["model"] = model_id

        def encode_query(self, texts: list) -> np.ndarray:
            seen["texts"] = list(texts)
            return np.zeros((len(texts), 768), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    engine = EmbeddingEngine("models/embeddinggemma2-q8.gguf", backend="st").load()
    assert engine.active_backend == "st"
    vec = engine.embed_text("測試")
    assert vec.shape[0] == 768
    assert seen["texts"][0].startswith("task: search result | query: ")


def test_invalid_backend_rejected() -> None:
    """非法後端應拋出 ValueError。"""
    from src.perception.embedding_engine import EmbeddingEngine

    with pytest.raises(ValueError):
        EmbeddingEngine("models/embeddinggemma2-q8.gguf", backend="不存在")


def test_ollama_backend_real() -> None:
    """Ollama 後端真實嵌入（無模型則跳過）。"""
    from src.perception.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine("models/embeddinggemma2-q8.gguf", backend="ollama")
    try:
        engine.load()
    except RuntimeError as exc:
        pytest.skip(f"Ollama 嵌入未就緒：{str(exc)[:100]}")
    assert engine.active_backend == "ollama"
    vec = engine.embed_text("個案表達強烈的無助感")
    assert vec.shape[0] == 768
    engine.unload()


def test_auto_prefers_ollama() -> None:
    """auto 模式有 Ollama 模型時應優先使用（無則跳過）。"""
    from src.perception.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine(_require_model())
    assert engine.backend == "auto"
    try:
        engine.load()
    except RuntimeError:
        pytest.skip("無可用嵌入後端")
    assert engine.active_backend == "ollama"
    engine.unload()


def test_embed_text_rejects_empty() -> None:
    """確認空輸入在模型載入前即被拒絕（不需模型）。"""
    from src.perception.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    with pytest.raises(ValueError):
        engine.embed_text("   ")


def test_embed_audio() -> None:
    """確認音訊嵌入明確拋出例外而非回傳偽造向量（不需模型）。"""
    from src.perception.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    with pytest.raises(RuntimeError):
        engine.embed_audio(np.zeros(16000, dtype=np.float32), 16000)
