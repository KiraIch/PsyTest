"""pyttsx3 保底 TTS 後端（Phase 4 實作，已於本機驗證）。

使用系統內建語音（macOS NSSpeech、Windows SAPI5、Linux espeak），
無情感但保證可用。僅支援 rate 與 volume 調整；
停頓標籤在 macOS 經 ``[[slnc 毫秒]]`` 實現，其餘平台移除標籤。

注意：pyttsx3 非執行緒安全，請勿跨執行緒共用實例。

驗證紀錄：macOS 合成成功（184 音色，AIFF 存檔正常）。
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logger import get_logger
from src.voice.tts_engine import AudioOutput, TTSEngine, parse_inline_tags

logger = get_logger(__name__)

_SLNC_RE = re.compile(r"\[\[slnc\s+\d+\s*\]\]")


class Pyttsx3Backend(TTSEngine):
    """pyttsx3 系統語音實作。"""

    name = "pyttsx3"

    def __init__(
        self,
        voice: str | None = None,
        rate: int = 200,
        volume: float = 0.8,
        sample_rate: int = 24000,
    ) -> None:
        """初始化後端參數（驅動延至首次合成時建立）。

        Args:
            voice: 音色 ID，未指定時使用系統預設。
            rate: 語速（每分鐘字數），僅為預設，可被導演參數覆寫。
            volume: 音量 0 至 1。
            sample_rate: 目標取樣率（輸出重取樣至此值）。

        Raises:
            ValueError: 參數非法時。
        """
        if rate <= 0:
            raise ValueError("rate 必須為正整數")
        if sample_rate <= 0:
            raise ValueError("sample_rate 必須為正整數")
        self.voice = voice
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.sample_rate = sample_rate
        self._engine: object | None = None

    def unavailable_reason(self) -> str:
        """回傳不可用原因。"""
        try:
            import pyttsx3  # noqa: F401
        except ImportError:
            return "缺少 pyttsx3，請執行 pip install pyttsx3（Linux 另需 espeak-ng）"
        return ""

    def is_available(self) -> bool:
        """檢查 pyttsx3 是否已安裝。"""
        return self.unavailable_reason() == ""

    def get_voice_options(self) -> List[str]:
        """回傳系統音色 ID 清單。"""
        engine = self._require_engine()
        try:
            voices = engine.getProperty("voices")  # type: ignore[union-attr]
            return [str(v.id) for v in (voices or [])]
        except Exception as exc:
            raise RuntimeError(f"取得音色清單失敗：{exc}") from exc

    def _require_engine(self) -> object:
        """建立（或回傳已快取的）pyttsx3 驅動。"""
        if self._engine is not None:
            return self._engine
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"pyttsx3 不可用：{reason}")
        import pyttsx3

        try:
            engine = pyttsx3.init()
        except Exception as exc:
            raise RuntimeError(f"pyttsx3 驅動初始化失敗：{exc}") from exc
        if self.voice:
            try:
                engine.setProperty("voice", self.voice)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"音色設定失敗，使用系統預設：{exc}")
        self._engine = engine
        return engine

    def _apply_tags(self, text: str, pause_ms: List[int]) -> str:
        """轉換停頓標籤：macOS 用 [[slnc]]，其餘平台移除。"""
        clean, tags = parse_inline_tags(text)
        if sys.platform != "darwin":
            return clean
        extra = 0
        if "hesitate" in tags:
            extra = max(extra, 1000)
        if "breath" in tags:
            extra = max(extra, 800)
        total = sum(pause_ms) + extra
        if total > 0:
            return f"{clean} [[slnc {int(total)}]]"
        return clean

    @staticmethod
    def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        """線性內插重取樣（無 scipy 依賴）。"""
        if source_rate == target_rate or waveform.size == 0:
            return waveform
        duration = len(waveform) / source_rate
        target_len = max(1, int(duration * target_rate))
        source_pos = np.linspace(0, len(waveform) - 1, num=target_len)
        return np.interp(source_pos, np.arange(len(waveform)), waveform).astype(np.float32)

    def synthesize(self, text: str, director_notes: Dict[str, object]) -> AudioOutput:
        """以系統語音合成。

        Args:
            text: 輸入文本（可含行內標籤）。
            director_notes: 可含 rate、volume、pause_ms 覆寫。

        Returns:
            :class:`AudioOutput`（float32 單聲道，重取樣至目標率）。

        Raises:
            ValueError: 文本為空或清理後為空時。
            RuntimeError: 引擎不可用或合成失敗時。
        """
        if not text or not text.strip():
            raise ValueError("輸入文本不可為空")
        engine = self._require_engine()
        try:
            rate = int(director_notes.get("rate", self.rate))  # type: ignore[arg-type]
            rate = max(50, min(400, rate))
            volume = max(0.0, min(1.0, float(director_notes.get("volume", self.volume))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rate, volume = self.rate, self.volume
        pause_ms = director_notes.get("pause_ms", [])
        pause_list = [int(p) for p in pause_ms if isinstance(p, (int, float))] if isinstance(pause_ms, list) else []
        spoken = self._apply_tags(text.strip(), pause_list)
        spoken = _SLNC_RE.sub("", spoken) if sys.platform != "darwin" else spoken
        if not spoken.strip():
            raise ValueError("文本經清理後為空，無法合成")
        try:
            engine.setProperty("rate", rate)  # type: ignore[union-attr]
            engine.setProperty("volume", volume)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"語速或音量設定失敗：{exc}")
        with tempfile.TemporaryDirectory(prefix="psytest_tts_") as tmpdir:
            out_path = str(Path(tmpdir) / "output.wav")
            try:
                engine.save_to_file(spoken, out_path)  # type: ignore[union-attr]
                engine.runAndWait()  # type: ignore[union-attr]
            except Exception as exc:
                raise RuntimeError(f"pyttsx3 合成失敗：{exc}") from exc
            try:
                import soundfile as sf

                waveform, source_rate = sf.read(out_path, dtype="float32", always_2d=False)
            except Exception as exc:
                raise RuntimeError(f"合成音檔讀取失敗：{exc}") from exc
        mono = np.asarray(waveform, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1).astype(np.float32)
        resampled = self._resample(mono.ravel(), int(source_rate), self.sample_rate)
        peak = float(np.abs(resampled).max()) if resampled.size else 0.0
        return AudioOutput(
            audio_array=resampled,
            sample_rate=self.sample_rate,
            duration=float(len(resampled) / self.sample_rate),
            metadata={
                "engine": "pyttsx3",
                "rate": rate,
                "volume": volume,
                "peak": round(peak, 3),
            },
        )
