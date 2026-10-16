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
    """確認能找到本專案鎖定的 e4b 變體。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過模型偵測測試")
    from src.slow_think.ollama_client import PINNED_MODEL, OllamaClient

    detected = OllamaClient().detect_model()
    assert detected == PINNED_MODEL


def test_model_detection_invalid_preference() -> None:
    """確認指定不存在的模型會拋出例外。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過模型偵測測試")
    from src.slow_think.ollama_client import OllamaClient

    with pytest.raises(RuntimeError):
        OllamaClient().detect_model(preference="不存在的模型")


def test_generate_response() -> None:
    """確認能生成回應（短 token smoke test）。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過生成測試")
    from src.slow_think.ollama_client import GenerateOptions, OllamaClient

    result = OllamaClient(timeout=180).generate(
        "用一句話描述下雨的感覺。",
        options=GenerateOptions(max_tokens=128, temperature=0.7),
    )
    assert result["text"].strip()
    assert result["model"]


def test_generate_empty_prompt_raises() -> None:
    """確認空提示會拋出 ValueError（不需連線）。"""
    from src.slow_think.ollama_client import OllamaClient

    with pytest.raises(ValueError):
        OllamaClient().generate("   ")


def test_detect_embedding_model() -> None:
    """確認能找到官方 embeddinggemma-2（缺失則跳過）。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過嵌入模型偵測測試")
    from src.slow_think.ollama_client import OllamaClient

    try:
        model = OllamaClient().detect_embedding_model()
    except RuntimeError as exc:
        pytest.skip(f"無嵌入模型：{exc}")
    assert "embeddinggemma" in model


def test_embed_text() -> None:
    """確認嵌入端點回傳 768 維向量（缺失則跳過）。"""
    if not _ollama_up():
        pytest.skip("Ollama 未運行，跳過嵌入測試")
    from src.slow_think.ollama_client import OllamaClient

    client = OllamaClient()
    try:
        model = client.detect_embedding_model()
    except RuntimeError as exc:
        pytest.skip(f"無嵌入模型：{exc}")
    vectors = client.embed("個案表達強烈的無助感", model=model)
    assert len(vectors) == 1
    assert len(vectors[0]) == 768


def test_model_keywords_cover_variants() -> None:
    """確認模糊匹配關鍵字涵蓋 e2b/e4b 命名變體。"""
    joined = " ".join(MODEL_KEYWORDS)
    for keyword in ("gemma", "uncensored", "hauhau", "aggressive"):
        assert keyword in joined
