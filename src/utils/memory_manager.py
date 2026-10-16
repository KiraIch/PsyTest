"""記憶體管理（骨架，Phase 6 實作）。

24GB 記憶體需同時調度多個模型，採按需載入與卸載：
    生成對白時僅載入 LLM（Ollama 管理）
    標註時載入 EmbeddingGemma 2
    決策時載入 Laya 421M
以引用計數管理模型生命週期，GPU 留給模型推理，
音訊前處理一律在 CPU 完成。
"""

from __future__ import annotations


class MemoryManager:
    """記憶體管理器（Phase 6 實作）。"""

    def __init__(self, max_memory_gb: float | None = None) -> None:
        """初始化管理器。

        Args:
            max_memory_gb: 上限（GB），None 表自動偵測。
        """
        if max_memory_gb is not None and max_memory_gb <= 0:
            raise ValueError("max_memory_gb 必須為正數")
        self.max_memory_gb = max_memory_gb

    def acquire(self, name: str, size_gb: float) -> bool:
        """申請載入模型（Phase 6 實作）。"""
        raise NotImplementedError("MemoryManager.acquire 排定於 Phase 6 實作")

    def release(self, name: str) -> None:
        """釋放模型（Phase 6 實作）。"""
        raise NotImplementedError("MemoryManager.release 排定於 Phase 6 實作")
