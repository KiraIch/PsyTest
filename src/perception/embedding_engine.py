"""EmbeddingGemma 2 嵌入引擎（骨架，Phase 2 實作）。

預定行為：經由 llama-cpp-python 以嵌入模式載入 GGUF，
n_ctx=2048，文字前綴 ``task: search result | query: ``，
mean pooling，輸出 768 維向量。

方法契約：
    embed_text(text) -> 768 維向量
    embed_audio(audio_array, sample_rate) -> 768 維向量
    embed_image(image_array) -> 768 維向量
    compute_similarity(vec_a, vec_b) -> 餘弦相似度
    batch_embed(texts) -> 向量清單
"""

from __future__ import annotations

from typing import List

import numpy as np

TEXT_PREFIX = "task: search result | query: "
EMBEDDING_DIM = 768


class EmbeddingEngine:
    """EmbeddingGemma 2 載入與推理（Phase 2 實作）。"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_batch: int = 32,
    ) -> None:
        """初始化引擎參數（實際載入延至 Phase 2）。

        Args:
            model_path: GGUF 權重路徑。
            n_ctx: 上下文長度。
            n_gpu_layers: GPU offload 層數，-1 表全部。
            n_batch: 批次大小。
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch

    def embed_text(self, text: str) -> np.ndarray:
        """文字嵌入（Phase 2 實作）。"""
        raise NotImplementedError("EmbeddingEngine.embed_text 排定於 Phase 2 實作")

    def embed_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """音訊嵌入（Phase 2 實作）。"""
        raise NotImplementedError("EmbeddingEngine.embed_audio 排定於 Phase 2 實作")

    def embed_image(self, image_array: np.ndarray) -> np.ndarray:
        """影像嵌入（Phase 2 實作）。"""
        raise NotImplementedError("EmbeddingEngine.embed_image 排定於 Phase 2 實作")

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """餘弦相似度（純數學實作，不依賴模型）。

        Args:
            vec_a: 第一個向量。
            vec_b: 第二個向量。

        Returns:
            餘弦相似度，範圍 [-1, 1]。

        Raises:
            ValueError: 維度不一致或零向量時。
        """
        a = np.asarray(vec_a, dtype=np.float64).ravel()
        b = np.asarray(vec_b, dtype=np.float64).ravel()
        if a.shape != b.shape:
            raise ValueError(f"向量維度不一致：{a.shape} vs {b.shape}")
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            raise ValueError("零向量無法計算餘弦相似度")
        return float(np.dot(a, b) / denom)

    def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """批次文字嵌入（Phase 2 實作）。"""
        raise NotImplementedError("EmbeddingEngine.batch_embed 排定於 Phase 2 實作")
