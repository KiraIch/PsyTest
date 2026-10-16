"""Ollama 客戶端測試（需 Ollama 運行，否則跳過）。"""

from __future__ import annotations

import pytest

requests = pytest.importorskip("requests", reason="缺少 requests 套件")

from src.slow_think.ollama_client import DEFAULT_HOST, MODEL_KEYWORDS


def _ollama_up(host: str = DEFAULT_HOST) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def test_ollama_connection() -> None:
    """確認能連接 Ollama，未運行則跳過。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過連線測試")


def test_model_detection() -> None:
    """確認能找到 Gemma 4 Uncensored 模型（Phase 2 實作後啟用）。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過模型偵測測試")
    pytest.skip("OllamaClient.detect_model 排定於 Phase 2 實作")


def test_generate_response() -> None:
    """確認能生成回應（Phase 2 實作後啟用）。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過生成測試")
    pytest.skip("OllamaClient.generate 排定於 Phase 2 實作")


def test_model_keywords_cover_variants() -> None:
    """確認模糊匹配關鍵字涵蓋 e2b/e4b 命名變體。"""
    joined = " ".join(MODEL_KEYWORDS)
    for keyword in ("gemma", "uncensored", "hauhau", "aggressive"):
        assert keyword in joined
