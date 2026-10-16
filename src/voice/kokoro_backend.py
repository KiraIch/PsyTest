"""Kokoro 輕量本地 TTS 後端（Phase 4 實作，經 sherpa-onnx）。

採用 kokoro-multi-lang-v1_1（中英雙語，103 音色，24kHz），
Apache-2.0 授權，CPU 可即時運行。本專案以其取代原規格的
Orpheus 雲端方案，決策見 HANDOVER.md ADR。

模型檔置於版本控制之外，預設路徑為
``~/.cache/psytest/kokoro-multi-lang-v1_1``，
可經環境變數 ``PSYTEST_KOKORO_DIR`` 覆寫。缺檔時拋出內含
下載指令的例外，不會靜默失敗。

使用範例：
    from src.voice.kokoro_backend import KokoroBackend

    engine = KokoroBackend(voice=46)  # 中文女聲
    output = engine.synthesize("你好，很高興見到你。", {})
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logger import get_logger
from src.voice.tts_engine import AudioOutput, TTSEngine, parse_inline_tags

logger = get_logger(__name__)

DEFAULT_MODEL_DIR = Path(
    os.getenv(
        "PSYTEST_KOKORO_DIR",
        str(Path.home() / ".cache" / "psytest" / "kokoro-multi-lang-v1_1"),
    )
)

REQUIRED_FILES = (
    "model.onnx",
    "voices.bin",
    "tokens.txt",
    "lexicon-us-en.txt",
    "lexicon-zh.txt",
)

SID_COUNT = 103
DEFAULT_SID = 46  # 中文女聲（v1_0 對照，v1_1 中文區段相容）

# 全形標點正規化（避免❓等未知 token 警告）。
_PUNCT_MAP = str.maketrans({"？": "。", "！": "。", "；": "，", "：": "，"})  # noqa: RUF001

_DOWNLOAD_HINT = (
    "Kokoro 模型檔缺失，請執行：\n"
    "mkdir -p ~/.cache/psytest && "
    "curl -sL -o ~/.cache/psytest/kokoro-multi-lang-v1_1.tar.bz2 "
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "kokoro-multi-lang-v1_1.tar.bz2 && "
    "tar -xjf ~/.cache/psytest/kokoro-multi-lang-v1_1.tar.bz2 "
    "-C ~/.cache/psytest/"
)


class KokoroBackend(TTSEngine):
    """Kokoro TTS 實作。"""

    name = "kokoro"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        voice: int | str = DEFAULT_SID,
        speed: float = 1.0,
        num_threads: int = 4,
        sample_rate: int = 24000,
    ) -> None:
        """初始化後端參數（模型延至首次合成時載入）。

        Args:
            model_dir: 模型目錄，未指定時使用預設快取路徑。
            voice: 音色 sid 整數（0 至 102），中文建議 45 至 52。
            speed: 語速乘數。
            num_threads: 推理線程數。
            sample_rate: 目標取樣率（Kokoro 原生 24000）。

        Raises:
            ValueError: 參數非法時。
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        try:
            self.voice = int(str(voice).split(":")[-1]) if isinstance(voice, str) else int(voice)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"voice 必須為 0 至 {SID_COUNT - 1} 的整數：{voice!r}") from exc
        if not 0 <= self.voice < SID_COUNT:
            raise ValueError(f"voice 超出範圍：{voice!r}")
        if num_threads <= 0:
            raise ValueError("num_threads 必須為正整數")
        if sample_rate <= 0:
            raise ValueError("sample_rate 必須為正整數")
        self.speed = max(0.5, min(2.0, float(speed)))
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self._tts: object | None = None

    def unavailable_reason(self) -> str:
        """回傳不可用原因。"""
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return "缺少 sherpa-onnx，請執行 pip install sherpa-onnx"
        missing = [f for f in REQUIRED_FILES if not (self.model_dir / f).exists()]
        if missing:
            return f"模型目錄缺檔：{missing}。{_DOWNLOAD_HINT}"
        return ""

    def is_available(self) -> bool:
        """檢查 sherpa-onnx 與模型檔是否就緒。"""
        return self.unavailable_reason() == ""

    def get_voice_options(self) -> List[str]:
        """回傳 sid 音色清單。"""
        return [f"sid:{i}" for i in range(SID_COUNT)]

    def _require_tts(self) -> object:
        """載入（或回傳已快取的）sherpa-onnx TTS 實例。"""
        if self._tts is not None:
            return self._tts
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"Kokoro 不可用：{reason}")
        import sherpa_onnx

        base = str(self.model_dir)
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=f"{base}/model.onnx",
                    voices=f"{base}/voices.bin",
                    tokens=f"{base}/tokens.txt",
                    data_dir=f"{base}/espeak-ng-data",
                    lexicon=f"{base}/lexicon-us-en.txt,{base}/lexicon-zh.txt",
                ),
                num_threads=self.num_threads,
                debug=False,
            ),
            rule_fsts=(
                f"{base}/phone-zh.fst,{base}/date-zh.fst,{base}/number-zh.fst"
            ),
            max_num_sentences=2,
        )
        logger.info(f"載入 Kokoro 模型：{base}")
        self._tts = sherpa_onnx.OfflineTts(config)
        return self._tts

    def _split_chunks(self, text: str) -> List[tuple[str, float]]:
        """依行內停頓標籤切分文本，回傳（片段，後續靜音秒數）。"""
        clean, tags = parse_inline_tags(text)
        pause = 0.0
        if "hesitate" in tags:
            pause = max(pause, 1.0)
        if "breath" in tags:
            pause = max(pause, 0.8)
        parts = [p.strip() for p in re.split(r"[。！？!?]", clean) if p.strip()]
        if not parts:
            return []
        chunks = [(part, 0.0) for part in parts[:-1]]
        chunks.append((parts[-1], pause))
        return chunks

    def synthesize(self, text: str, director_notes: Dict[str, object]) -> AudioOutput:
        """合成中文或英文語音。

        Args:
            text: 輸入文本（可含行內標籤，標籤轉為停頓）。
            director_notes: 可含 speed、voice 覆寫。

        Returns:
            :class:`AudioOutput`（float32 單聲道，24kHz）。

        Raises:
            ValueError: 文本為空時。
            RuntimeError: 引擎不可用或合成失敗時。
        """
        if not text or not text.strip():
            raise ValueError("輸入文本不可為空")
        tts = self._require_tts()
        try:
            speed = max(0.5, min(2.0, float(director_notes.get("speed", self.speed))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            speed = self.speed
        try:
            voice_raw = director_notes.get("voice", self.voice)
            voice = int(str(voice_raw).split(":")[-1]) if isinstance(voice_raw, str) else int(voice_raw)  # type: ignore[arg-type]
            if not 0 <= voice < SID_COUNT:
                voice = self.voice
        except (TypeError, ValueError):
            voice = self.voice
        chunks = self._split_chunks(text.strip().translate(_PUNCT_MAP))
        if not chunks:
            raise ValueError("文本經清理後為空，無法合成")
        pieces: List[np.ndarray] = []
        try:
            for part, pause in chunks:
                audio = tts.generate(part, sid=voice, speed=speed)  # type: ignore[union-attr]
                samples = np.asarray(audio.samples, dtype=np.float32).ravel()
                pieces.append(samples)
                if pause > 0:
                    pieces.append(np.zeros(int(pause * 24000), dtype=np.float32))
        except Exception as exc:
            raise RuntimeError(f"Kokoro 合成失敗：{exc}") from exc
        waveform = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        peak = float(np.abs(waveform).max()) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        return AudioOutput(
            audio_array=waveform.astype(np.float32),
            sample_rate=24000,
            duration=float(len(waveform) / 24000),
            metadata={
                "engine": "kokoro",
                "voice": f"sid:{voice}",
                "speed": speed,
                "peak": round(peak, 3),
            },
        )

    def unload(self) -> None:
        """釋放模型（冪等，供管線記憶體管理使用）。"""
        self._tts = None
        logger.info("Kokoro 模型已卸載")
