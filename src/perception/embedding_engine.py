"""EmbeddingGemma 2 嵌入引擎（Phase 2 實作）。

經由 llama-cpp-python 以嵌入模式載入 GGUF 權重，提供文字嵌入、
批次嵌入與餘弦相似度計算。模型採按需載入與卸載，避免與其他
模型同時佔用記憶體。

載入設定：
    n_ctx=2048，embedding=True，文字前綴
    ``task: search result | query: ``，mean pooling（由模型端處理），
    輸出 768 維向量。n_gpu_layers 依後端自動設定
   （Metal、CUDA、ROCm 全量 offload，CPU 為 0）。

限制說明：
    GGUF 經 llama-cpp 僅暴露文字嵌入端點，音訊與影像的語意嵌入
    需待 Phase 3 的特徵分析器與 Phase 5 的數據收集器組合完成。
    因此 ``embed_audio`` 與 ``embed_image`` 會明確拋出例外，
    而非回傳偽造向量，避免污染數據集。

使用範例：
    from src.perception.embedding_engine import EmbeddingEngine

    engine = EmbeddingEngine("models/embeddinggemma2-q8.gguf")
    engine.load()
    try:
        vec = engine.embed_text("個案表達強烈的無助感")
    finally:
        engine.unload()
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.config import resolve_model_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

TEXT_PREFIX = "task: search result | query: "
EMBEDDING_DIM = 768

_INSTALL_HINT = (
    "缺少 llama-cpp-python，請依 README.md 3.2 節依平台重新安裝，"
    "例如 macOS："
    'CMAKE_ARGS="-DGGML_METAL=on" '
    "pip install llama-cpp-python --force-reinstall --no-cache-dir"
)


def _resolve_gpu_layers(n_gpu_layers: Union[str, int, None]) -> int:
    """解析 GPU offload 層數設定。

    Args:
        n_gpu_layers: ``auto``、``max``、``none``、數字或 None。

    Returns:
        llama-cpp 接受的層數，-1 表全部 offload，0 表純 CPU。
    """
    if n_gpu_layers is None:
        return -1
    if isinstance(n_gpu_layers, int):
        return n_gpu_layers
    normalized = str(n_gpu_layers).strip().lower()
    if normalized in {"auto", "max", "-1"}:
        try:
            from src.backend.detector import detect_backend

            return detect_backend().recommended_offload_layers
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"後端偵測失敗，退回全量 offload：{exc}")
            return -1
    if normalized in {"none", "cpu", "0"}:
        return 0
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"無效的 gpu_layers：{n_gpu_layers!r}，"
            '允許值為 auto、max、none 或整數'
        ) from exc


class EmbeddingEngine:
    """EmbeddingGemma 2 載入與推理。"""

    def __init__(
        self,
        model_path: Union[str, Path],
        n_ctx: int = 2048,
        n_gpu_layers: Union[str, int, None] = "auto",
        n_batch: int = 32,
    ) -> None:
        """初始化引擎參數（實際載入請呼叫 :meth:`load`）。

        Args:
            model_path: GGUF 權重路徑（支援規格別名）。
            n_ctx: 上下文長度。
            n_gpu_layers: GPU offload 層數設定，預設 auto。
            n_batch: 批次大小。

        Raises:
            ValueError: 參數非法時。
        """
        if n_ctx <= 0:
            raise ValueError("n_ctx 必須為正整數")
        if n_batch <= 0:
            raise ValueError("n_batch 必須為正整數")
        self.model_path = resolve_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers_setting = n_gpu_layers
        self.n_batch = n_batch
        self._model: Optional[object] = None
        self._resolved_layers: Optional[int] = None

    @property
    def is_loaded(self) -> bool:
        """模型是否已載入。"""
        return self._model is not None

    def load(self) -> "EmbeddingEngine":
        """載入模型（冪等，重複呼叫不會重複載入）。

        Returns:
            自身，支援鏈式呼叫。

        Raises:
            ImportError: 未安裝 llama-cpp-python 時。
            FileNotFoundError: 權重檔不存在時。
            RuntimeError: 載入失敗時（含 BF16 相容性指引）。
        """
        if self._model is not None:
            return self
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到 Embedding 模型檔：{self.model_path}，"
                "請依 models/README.md 取得檔案"
            )
        layers = _resolve_gpu_layers(self.n_gpu_layers_setting)
        logger.info(f"載入 Embedding 模型：{self.model_path}（GPU 層數={layers}）")
        try:
            self._model = Llama(
                model_path=str(self.model_path),
                embedding=True,
                n_ctx=self.n_ctx,
                n_gpu_layers=layers,
                n_batch=self.n_batch,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Embedding 模型載入失敗：{exc}。"
                "若為 BF16 相容性問題，請確認 llama-cpp 為最新版，"
                "或改用 FP16 量化版本"
            ) from exc
        self._resolved_layers = layers
        logger.info("Embedding 模型載入完成")
        return self

    def unload(self) -> None:
        """卸載模型並釋放記憶體（冪等）。"""
        if self._model is None:
            return
        try:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"模型卸載時發生非致命錯誤：{exc}")
        finally:
            self._model = None
            self._resolved_layers = None
        logger.info("Embedding 模型已卸載")

    def __enter__(self) -> "EmbeddingEngine":
        """支援 with 語句自動載入。"""
        return self.load()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """離開 with 區塊時自動卸載。"""
        self.unload()

    def _require_model(self) -> object:
        """確認模型已載入，未載入時拋出指引明確的例外。"""
        if self._model is None:
            raise RuntimeError("模型尚未載入，請先呼叫 load()")
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """文字嵌入，回傳 768 維 float32 向量。

        Args:
            text: 輸入文字，不可為空。

        Returns:
            768 維嵌入向量。

        Raises:
            ValueError: 輸入為空時。
            RuntimeError: 模型未載入或推理失敗時。
        """
        if not text or not text.strip():
            raise ValueError("輸入文字不可為空")
        model = self._require_model()
        try:
            result = model.create_embedding(TEXT_PREFIX + text.strip())  # type: ignore[union-attr]
            vector = np.asarray(result["data"][0]["embedding"], dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"文字嵌入失敗：{exc}") from exc
        if vector.shape[0] != EMBEDDING_DIM:
            logger.warning(
                f"嵌入維度為 {vector.shape[0]}，與預期 {EMBEDDING_DIM} 不一致"
            )
        return vector

    def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """批次文字嵌入。

        Args:
            texts: 輸入文字清單，不可為空。

        Returns:
            與輸入等長的向量清單。

        Raises:
            ValueError: 清單為空或含空字串時。
            RuntimeError: 模型未載入或推理失敗時。
        """
        if not texts:
            raise ValueError("輸入清單不可為空")
        if any(not t or not t.strip() for t in texts):
            raise ValueError("輸入清單不可包含空字串")
        model = self._require_model()
        prefixed = [TEXT_PREFIX + t.strip() for t in texts]
        try:
            result = model.create_embedding(prefixed)  # type: ignore[union-attr]
            return [
                np.asarray(item["embedding"], dtype=np.float32)
                for item in result["data"]
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"批次嵌入失敗，退回逐筆嵌入：{exc}")
        vectors: List[np.ndarray] = []
        for text in texts:
            vectors.append(self.embed_text(text))
        return vectors

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

    def embed_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """音訊嵌入（尚未支援，刻意拋出例外）。

        GGUF 經 llama-cpp 僅暴露文字嵌入端點，無法由原始波形直接
        產出語意向量。音訊語意標註需待 Phase 3 的 ``audio_analyzer``
        萃取特徵後，於 Phase 5 的數據收集器組合完成。本方法刻意
        拋出例外，避免回傳偽造向量污染數據集。

        Raises:
            RuntimeError: 說明替代方案的例外。
        """
        raise RuntimeError(
            "音訊嵌入尚未支援：GGUF 僅提供文字嵌入端點。"
            "請改用 src/perception/audio_analyzer.py 萃取聲學特徵"
            "（Phase 3），語意組合於 Phase 5 完成"
        )

    def embed_image(self, image_array: np.ndarray) -> np.ndarray:
        """影像嵌入（尚未支援，刻意拋出例外）。

        理由同 :meth:`embed_audio`。視覺特徵請改用
        ``src/perception/visual_analyzer.py``（Phase 3）。

        Raises:
            RuntimeError: 說明替代方案的例外。
        """
        raise RuntimeError(
            "影像嵌入尚未支援：GGUF 僅提供文字嵌入端點。"
            "請改用 src/perception/visual_analyzer.py（Phase 3）"
        )


def load_default_engine(n_batch: Optional[int] = None) -> EmbeddingEngine:
    """依預設設定建立並載入引擎的便捷函式。

    Args:
        n_batch: 覆寫批次大小，未指定時使用設定值。

    Returns:
        已載入的 :class:`EmbeddingEngine`。
    """
    from src.config import load_config

    config = load_config()
    return EmbeddingEngine(
        model_path=config.models.embedding.path,
        n_ctx=config.models.embedding.context_length,
        n_gpu_layers=config.models.embedding.gpu_layers,
        n_batch=n_batch or config.models.embedding.batch_size,
    ).load()
