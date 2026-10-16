"""輸出格式化（Phase 5 實作）。

支援 JSONL（主要格式，每行一場會談）、WAV 音訊個別儲存、
中繼數據摘要 manifest.json。Parquet 為未來選項（未實作）。

輸出結構：
    data/output/manifest.json
    data/output/sessions/session_<id>.jsonl
    data/output/audio/session_<id>_turn_<n>.wav
    data/output/annotations/session_<id>_annotations.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 numpy，請先執行 pip install -r requirements.txt"
    ) from exc

from src.utils.logger import get_logger

logger = get_logger(__name__)


def quantize_embedding(vector: List[float], fmt: str = "float32") -> Any:
    """量化嵌入向量以節省儲存。

    Args:
        vector: 浮點向量。
        fmt: ``float32``（原樣）、``float16``（半精度）、
            ``int8``（線性量化為字典，含 scale）。

    Returns:
        量化後的表示法。

    Raises:
        ValueError: 格式非法時。
    """
    if fmt == "float32":
        return [float(v) for v in vector]
    if fmt == "float16":
        return np.asarray(vector, dtype=np.float16).astype(np.float64).tolist()
    if fmt == "int8":
        arr = np.asarray(vector, dtype=np.float64)
        scale = float(np.abs(arr).max()) / 127.0 if arr.size else 1.0
        scale = scale if scale > 0 else 1.0
        quantized = np.round(arr / scale).astype(np.int8).tolist()
        return {"quant": "int8", "scale": scale, "data": quantized}
    raise ValueError(f"不支援的嵌入格式：{fmt!r}")


class OutputFormatter:
    """輸出格式化器。"""

    def __init__(
        self,
        output_dir: str | Path,
        embedding_format: str = "float32",
    ) -> None:
        """初始化輸出目錄結構。

        Args:
            output_dir: 輸出根目錄（自動建立子目錄）。
            embedding_format: 嵌入儲存格式。

        Raises:
            ValueError: 嵌入格式非法時。
            OSError: 目錄無法建立時。
        """
        if embedding_format not in {"float32", "float16", "int8"}:
            raise ValueError(f"不支援的嵌入格式：{embedding_format!r}")
        self.output_dir = Path(output_dir)
        self.embedding_format = embedding_format
        try:
            for sub in ("sessions", "audio", "annotations"):
                (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"無法建立輸出目錄：{self.output_dir}：{exc}") from exc

    def _encode_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """量化單輪中的嵌入向量，回傳可 JSON 序列化字典。"""
        import copy

        encoded = copy.deepcopy(turn)
        for key, value in encoded.get("embedding_vectors", {}).items():
            if isinstance(value, list) and value:
                encoded["embedding_vectors"][key] = quantize_embedding(
                    value, self.embedding_format
                )
        encoded["record"] = "turn"
        return encoded

    def write_session(self, session: Dict[str, Any]) -> Path:
        """寫入單場會談（JSONL：首行為 meta，其後每輪一行）。

        逐輪分行存放，可串流讀取、逐行檢查，不再是單一巨行。
        歷史單行格式仍可由驗證與報告腳本讀取（向下相容）。

        Args:
            session: 會談字典（須含 session_id 與 turns）。

        Returns:
            寫入的檔案路徑。

        Raises:
            ValueError: 缺少必要欄位時。
            OSError: 寫入失敗時。
        """
        import json

        session_id = session.get("session_id")
        if not session_id:
            raise ValueError("session 缺少 session_id")
        turns = session.get("turns")
        if not isinstance(turns, list):
            raise ValueError("session 缺少 turns 清單")
        meta = {key: value for key, value in session.items() if key != "turns"}
        meta["record"] = "meta"
        meta["turn_count"] = len(turns)
        path = self.output_dir / "sessions" / f"session_{session_id}.jsonl"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                for turn in turns:
                    fh.write(json.dumps(self._encode_turn(turn), ensure_ascii=False) + "\n")
        except OSError as exc:
            raise OSError(f"會談寫入失敗：{path}：{exc}") from exc
        logger.info(f"已寫入會談：{path}（{len(turns)} 輪）")
        return path

    def write_audio(
        self, session_id: str, turn_id: int, waveform: np.ndarray, sample_rate: int
    ) -> Optional[Path]:
        """寫入單輪音訊 WAV。

        Args:
            session_id: 會談 ID。
            turn_id: 輪次。
            waveform: 波形（任意範圍，自動正規化）。
            sample_rate: 取樣率。

        Returns:
            寫入路徑；waveform 為空時回傳 None。
        """
        arr = np.asarray(waveform, dtype=np.float64).ravel()
        if arr.size == 0:
            return None
        try:
            import soundfile as sf

            path = self.output_dir / "audio" / f"session_{session_id}_turn_{turn_id:02d}.wav"
            peak = float(np.abs(arr).max())
            normalized = (arr / peak * 0.95).astype(np.float32) if peak > 1.0 else arr.astype(np.float32)
            sf.write(str(path), normalized, sample_rate)
            return path
        except ImportError:
            logger.warning("缺少 soundfile，跳過音訊寫入")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"音訊寫入失敗（跳過）：{exc}")
            return None

    def write_annotations(self, session_id: str, annotations: Dict[str, Any]) -> Path:
        """寫入會談標註檔。"""
        import json

        path = self.output_dir / "annotations" / f"session_{session_id}_annotations.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(annotations, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise OSError(f"標註寫入失敗：{path}：{exc}") from exc
        return path

    def write_manifest(
        self,
        sessions: List[Dict[str, Any]],
        models_used: Optional[Dict[str, str]] = None,
        hardware: str = "",
    ) -> Path:
        """寫入數據集索引 manifest.json。

        Args:
            sessions: 已寫入的會談字典清單。
            models_used: 使用的模型版本對照。
            hardware: 硬體描述。

        Returns:
            manifest 路徑。
        """
        import json

        total_turns = sum(len(s.get("turns", [])) for s in sessions)
        total_seconds = sum(
            float(t.get("client", {}).get("audio", {}).get("duration") or 0.0)
            for s in sessions
            for t in s.get("turns", [])
        )
        manifest = {
            "dataset_name": "PsyTest-MM-v1",
            "total_sessions": len(sessions),
            "total_turns": total_turns,
            "total_audio_duration_hours": round(total_seconds / 3600.0, 3),
            "personas": sorted({str(s.get("persona", "")) for s in sessions if s.get("persona")}),
            "modalities": ["text", "audio", "visual", "psychological"],
            "embedding_dimensions": 768,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware,
            "models_used": models_used or {},
        }
        path = self.output_dir / "manifest.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise OSError(f"manifest 寫入失敗：{path}：{exc}") from exc
        logger.info(f"已寫入 manifest：{path}")
        return path
