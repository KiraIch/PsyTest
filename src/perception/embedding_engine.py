"""EmbeddingGemma 2 嵌入引擎（Phase 2 實作，Phase 7 加入替代後端）。

三後端架構（``backend`` 參數，預設 ``auto`` 依序嘗試）：
    ollama：經 Ollama /api/embed 使用官方 embeddinggemma-2，
        免授權，本機有模型即用（首選）。
    st：經 sentence-transformers 載入 HF 模型（需接受授權）。
    llama_cpp：經 llama-cpp 載入 GGUF（待上游支援 gemma-embedding2）。

輸出 768 維向量，文字前綴 ``task: search result | query: ``。
模型採按需載入與卸載，避免與其他模型同時佔用記憶體。

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

# sentence-transformers 替代模型（與本機 Q8 GGUF 同源，768 維）。
ST_MODEL = "google/embeddinggemma-300m"

_ST_INSTALL_HINT = "缺少 sentence-transformers，請執行 pip install sentence-transformers"

_ST_GATED_HINT = (
    "google/embeddinggemma-300m 為授權管制模型，請先以瀏覽器登入 HuggingFace "
    "並於 https://huggingface.co/google/embeddinggemma-300m 接受授權，"
    "再執行 hf download google/embeddinggemma-300m（需本機 HF 登入）"
)

# Ollama 嵌入模型（官方 embeddinggemma-2，免授權，auto 首選）。
OLLAMA_EMBED_MODEL = "embeddinggemma-2"

_INSTALL_HINT = (
    "缺少 llama-cpp-python，請依 README.md 3.2 節依平台重新安裝，"
    "例如 macOS："
    'CMAKE_ARGS="-DGGML_METAL=on" '
    "pip install llama-cpp-python --force-reinstall --no-cache-dir"
)

# llama.cpp 尚不支援的嵌入架構（讀取 GGUF 中繼資料判定）。
_UNSUPPORTED_ARCHS = {"gemma-embedding2"}


def read_gguf_arch(path: str | Path) -> str:
    """讀取 GGUF 檔案的 general.architecture 欄位。

    Args:
        path: GGUF 檔案路徑。

    Returns:
        架構字串，讀取失敗時回傳空字串（不拋出例外）。
    """
    import struct

    try:
        with open(path, "rb") as fh:
            header = fh.read(24)
            if len(header) < 24 or header[:4] != b"GGUF":
                return ""
            n_tensors, n_kv = struct.unpack("<qq", header[8:24])
            for _ in range(min(n_kv, 64)):
                raw_len = fh.read(8)
                if len(raw_len) < 8:
                    return ""
                (key_len,) = struct.unpack("<q", raw_len)
                if key_len <= 0 or key_len > 256:
                    return ""
                key = fh.read(key_len).decode("utf-8", errors="replace")
                raw_type = fh.read(4)
                if len(raw_type) < 4:
                    return ""
                (value_type,) = struct.unpack("<i", raw_type)
                if key == "general.architecture" and value_type == 8:
                    raw_vlen = fh.read(8)
                    (value_len,) = struct.unpack("<q", raw_vlen)
                    return fh.read(value_len).decode("utf-8", errors="replace")
                _skip_gguf_value(fh, value_type)
        return ""
    except (OSError, struct.error, ValueError):
        return ""


def _skip_gguf_value(fh: object, value_type: int) -> None:
    """跳過 GGUF kv 值（供 read_gguf_arch 使用）。"""
    import struct

    read = fh.read  # type: ignore[union-attr]
    if value_type in {0, 1}:  # uint8, int8
        read(1)
    elif value_type in {2, 3, 6}:  # uint16, int16, float16
        read(2)
    elif value_type in {4, 5, 7, 11}:  # uint32, int32, float32, bool
        read(4)
    elif value_type in {8, 10}:  # string, array of ...: string→len+bytes
        if value_type == 8:
            (length,) = struct.unpack("<q", read(8))
            read(length)
        else:
            (elem_type,) = struct.unpack("<i", read(4))
            (count,) = struct.unpack("<q", read(8))
            for _ in range(min(count, 1024)):
                _skip_gguf_value(fh, elem_type)
    elif value_type in {9, 12}:  # uint64, int64, float64
        read(8)
    else:
        raise ValueError(f"未知的 GGUF 型別：{value_type}")


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
        backend: str = "auto",
        st_model: str = ST_MODEL,
        ollama_host: Optional[str] = None,
        ollama_model: str = OLLAMA_EMBED_MODEL,
    ) -> None:
        """初始化引擎參數（實際載入請呼叫 :meth:`load`）。

        Args:
            model_path: GGUF 權重路徑（llama_cpp 後端用，支援規格別名）。
            n_ctx: 上下文長度。
            n_gpu_layers: GPU offload 層數設定，預設 auto。
            n_batch: 批次大小。
            backend: ``auto``（Ollama 優先，有模型即用；否則 ST；
                再失敗回退 GGUF）、``ollama``、``st``、``llama_cpp``。
            st_model: ST 後端的 HF 模型 ID。
            ollama_host: Ollama 位址，未指定時讀取 OLLAMA_HOST。
            ollama_model: Ollama 嵌入模型名稱。

        Raises:
            ValueError: 參數非法時。
        """
        if n_ctx <= 0:
            raise ValueError("n_ctx 必須為正整數")
        if n_batch <= 0:
            raise ValueError("n_batch 必須為正整數")
        if backend not in {"auto", "llama_cpp", "st", "ollama"}:
            raise ValueError("backend 必須為 auto、llama_cpp、st 或 ollama")
        self.model_path = resolve_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers_setting = n_gpu_layers
        self.n_batch = n_batch
        self.backend = backend
        self.st_model = st_model
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self._model: Optional[object] = None
        self._st: Optional[object] = None
        self._ollama: Optional[object] = None
        self._resolved_layers: Optional[int] = None

    @property
    def is_loaded(self) -> bool:
        """任一後端就緒即為已載入。"""
        return (
            self._model is not None
            or self._st is not None
            or self._ollama is not None
        )

    @property
    def active_backend(self) -> str:
        """回傳作用中後端（ollama、st、llama_cpp 或空字串）。"""
        if self._ollama is not None:
            return "ollama"
        if self._st is not None:
            return "st"
        if self._model is not None:
            return "llama_cpp"
        return ""

    def load(self) -> "EmbeddingEngine":
        """載入模型（冪等，重複呼叫不會重複載入）。

        Returns:
            自身，支援鏈式呼叫。

        Raises:
            ImportError: 所選後端依賴缺失時。
            FileNotFoundError: GGUF 權重檔不存在時（llama_cpp 後端）。
            RuntimeError: 載入失敗時（含架構與授權指引）。
        """
        if self.is_loaded:
            return self
        if self.backend in {"auto", "ollama"}:
            try:
                return self._load_ollama()
            except (ImportError, RuntimeError) as exc:
                if self.backend == "ollama":
                    raise
                logger.warning(f"Ollama 嵌入不可用，改試 ST：{str(exc)[:100]}")
        if self.backend in {"auto", "st"}:
            try:
                return self._load_st()
            except ImportError as exc:
                if self.backend == "st":
                    raise
                logger.warning(f"ST 後端不可用，回退 GGUF：{exc}")
            except RuntimeError as exc:
                if self.backend == "st":
                    raise
                logger.warning(f"ST 載入失敗，回退 GGUF：{str(exc)[:100]}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到 Embedding 模型檔：{self.model_path}，"
                "請依 models/README.md 取得檔案"
            )
        arch = read_gguf_arch(self.model_path)
        if arch in _UNSUPPORTED_ARCHS:
            raise RuntimeError(
                f"Embedding 模型架構尚不支援：{arch}（llama.cpp 未收錄）。"
                f"可改用 sentence-transformers 後端（backend='st'），{_ST_GATED_HINT}"
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
            message = str(exc)
            if "unknown model architecture" in message:
                raise RuntimeError(
                    f"Embedding 模型載入失敗：llama.cpp 尚不支援此架構：{exc}。"
                    f"可改用 sentence-transformers 後端（backend='st'），{_ST_GATED_HINT}"
                ) from exc
            raise RuntimeError(
                f"Embedding 模型載入失敗：{exc}。"
                "若為 BF16 相容性問題，請確認 llama-cpp 為最新版，"
                "或改用 FP16 量化版本"
            ) from exc
        self._resolved_layers = layers
        logger.info("Embedding 模型載入完成")
        return self

    def _load_st(self) -> "EmbeddingEngine":
        """經 sentence-transformers 載入 HF 模型。

        Returns:
            自身。

        Raises:
            ImportError: 套件缺失時。
            RuntimeError: 模型取得失敗時（含授權管制指引）。
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(_ST_INSTALL_HINT) from exc
        logger.info(f"載入 ST 嵌入模型：{self.st_model}")
        try:
            self._st = SentenceTransformer(self.st_model)
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "restricted" in message.lower() or "401" in message or "403" in message:
                raise RuntimeError(f"ST 模型取得失敗（授權管制）：{_ST_GATED_HINT}") from exc
            raise RuntimeError(f"ST 模型載入失敗：{exc}") from exc
        logger.info("ST 嵌入模型載入完成")
        return self

    def _load_ollama(self) -> "EmbeddingEngine":
        """經 Ollama /api/embed 就緒（官方 embeddinggemma-2，免授權）。

        Returns:
            自身。

        Raises:
            ImportError: requests 缺失時（極少見）。
            RuntimeError: Ollama 未運行或模型未拉取時（含拉取指引）。
        """
        from src.slow_think.ollama_client import OllamaClient

        try:
            client = OllamaClient(host=self.ollama_host)
        except ValueError as exc:
            raise RuntimeError(f"Ollama 客戶端建立失敗：{exc}") from exc
        if not client.is_running():
            raise RuntimeError(
                "Ollama 未運行，無法使用嵌入後端。請先啟動 Ollama"
            )
        try:
            model = client.detect_embedding_model(self.ollama_model)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}") from exc
        self._ollama = (client, model)
        logger.info(f"Ollama 嵌入就緒：{model}")
        return self

    def unload(self) -> None:
        """卸載模型並釋放記憶體（冪等）。"""
        if self._model is None and self._st is None and self._ollama is None:
            return
        try:
            close = getattr(self._model, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"模型卸載時發生非致命錯誤：{exc}")
        finally:
            self._model = None
            self._st = None
            self._ollama = None
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
        if self._ollama is not None:
            return self._ollama
        if self._st is not None:
            return self._st
        if self._model is None:
            raise RuntimeError("模型尚未載入，請先呼叫 load()")
        return self._model

    def _embed_via_ollama(self, texts: List[str]) -> List[np.ndarray]:
        """經 Ollama /api/embed 批次編碼（含查詢前綴）。"""
        if self._ollama is None:
            raise RuntimeError("模型尚未載入，請先呼叫 load()")
        client, model = self._ollama
        prefixed = [TEXT_PREFIX + t.strip() for t in texts]
        try:
            vectors = client.embed(prefixed, model=model)  # type: ignore[union-attr]
            return [np.asarray(v, dtype=np.float32).ravel() for v in vectors]
        except Exception as exc:
            raise RuntimeError(f"Ollama 嵌入失敗：{exc}") from exc

    def _embed_via_st(self, texts: List[str]) -> List[np.ndarray]:
        """經 sentence-transformers 批次編碼（含查詢前綴）。"""
        model = self._st
        if model is None:
            raise RuntimeError("模型尚未載入，請先呼叫 load()")
        prefixed = [TEXT_PREFIX + t.strip() for t in texts]
        try:
            encode_query = getattr(model, "encode_query", None)
            if callable(encode_query):
                vectors = encode_query(prefixed)
            else:
                vectors = model.encode(prefixed, convert_to_numpy=True)  # type: ignore[union-attr]
            return [np.asarray(v, dtype=np.float32).ravel() for v in vectors]
        except Exception as exc:
            raise RuntimeError(f"ST 文字嵌入失敗：{exc}") from exc

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
        if self._ollama is not None:
            vectors = self._embed_via_ollama([text])
            vector = vectors[0]
            if vector.shape[0] != EMBEDDING_DIM:
                logger.warning(
                    f"嵌入維度為 {vector.shape[0]}，與預期 {EMBEDDING_DIM} 不一致"
                )
            return vector
        if self._st is not None:
            vectors = self._embed_via_st([text])
            vector = vectors[0]
            if vector.shape[0] != EMBEDDING_DIM:
                logger.warning(
                    f"嵌入維度為 {vector.shape[0]}，與預期 {EMBEDDING_DIM} 不一致"
                )
            return vector
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
        if self._ollama is not None:
            return self._embed_via_ollama(texts)
        if self._st is not None:
            return self._embed_via_st(texts)
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
