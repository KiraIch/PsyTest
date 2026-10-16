"""TTS 抽象層（Phase 4 實作）。

引擎選擇順序（情感優先、輕量次之）：
    chattts → kokoro → pyttsx3。
ChatTTS 情感韻律最佳但 CPU 較慢；Kokoro 輕量即時但語調平穩；
pyttsx3 為系統語音保底。Orpheus 僅在明確指定且已配置雲端端點時使用，
不參與自動選擇（詳見 ``orpheus_backend.py``）。

導演行內標籤（兩種寫法皆接受，合成前由各後端轉換或移除）：
    hesitate、cry、whisper、angry、flat、breath。

使用範例：
    from src.voice.tts_engine import select_engine

    engine = select_engine("auto")
    output = engine.synthesize("你好，很高興見到你。", {"volume": 0.8})
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

INLINE_TAGS = ("hesitate", "cry", "whisper", "angry", "flat", "breath")

_INLINE_TAG_RE = re.compile(r"<(hesitate|cry|whisper|angry|flat|breath)>", re.IGNORECASE)
_BRACKET_TAG_RE = re.compile(r"\[(hesitate|cry|whisper|angry|flat|breath)\]", re.IGNORECASE)

# 舞台指示（動作與表情描述）不可念出，合成前移除。
# 注意：僅移除整段括號，括號內的導演標籤已先由標籤解析處理。
_STAGE_FULLWIDTH_RE = re.compile(r"（[^（）]*）")
_STAGE_ASCII_RE = re.compile(r"\([^()]*\)")

# 自動選擇順序（Orpheus 排除在外，見模組 docstring）。
ENGINE_ORDER = ("chattts", "kokoro", "pyttsx3")


@dataclass
class AudioOutput:
    """語音合成輸出。"""

    audio_array: np.ndarray
    sample_rate: int
    duration: float
    metadata: Dict[str, object] = field(default_factory=dict)


class TTSEngine(ABC):
    """TTS 引擎抽象基類。"""

    name: str = "base"

    @abstractmethod
    def synthesize(self, text: str, director_notes: Dict[str, object]) -> AudioOutput:
        """合成語音。

        Args:
            text: 輸入文本（可含行內導演標籤）。
            director_notes: 導演參數（音量、語速等）。

        Returns:
            :class:`AudioOutput`，音訊為 float32 單聲道。

        Raises:
            ValueError: 文本為空時。
            RuntimeError: 引擎不可用或合成失敗時。
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """引擎是否可用（依賴與模型是否就緒）。"""
        raise NotImplementedError

    @abstractmethod
    def get_voice_options(self) -> List[str]:
        """可用音色清單。"""
        raise NotImplementedError

    def unavailable_reason(self) -> str:
        """不可用原因說明（供錯誤訊息使用）。"""
        return "依賴或模型尚未就緒"


def parse_inline_tags(text: str) -> Tuple[str, List[str]]:
    """解析並移除行內導演標籤。

    接受 ``<tag>``（Orpheus 風格）與 ``[tag]``（ChatTTS 風格）兩種寫法，
    僅識別 :data:`INLINE_TAGS` 六種標籤，其餘方括號內容（如 ChatTTS
    原生標籤 ``[laugh]``）予以保留。

    Args:
        text: 輸入文本。

    Returns:
        （乾淨文本，標籤清單去重保序）。
    """
    tags: List[str] = []
    for match in _INLINE_TAG_RE.findall(text):
        tag = match.lower()
        if tag not in tags:
            tags.append(tag)
    for match in _BRACKET_TAG_RE.findall(text):
        tag = match.lower()
        if tag not in tags:
            tags.append(tag)
    clean = _INLINE_TAG_RE.sub("", text)
    clean = _BRACKET_TAG_RE.sub("", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, tags


def strip_inline_tags(text: str) -> str:
    """移除行內導演標籤，僅回傳乾淨文本。"""
    clean, _ = parse_inline_tags(text)
    return clean


def strip_stage_directions(text: str) -> str:
    """移除舞台指示（中英文括號段落），保留對白。

    LLM 常在對白中夾帶動作與表情描述（例如「（低頭）你好」），
    這些不可送入 TTS。JSONL 原文保留完整，僅合成用清理版。

    Args:
        text: 含舞台指示的對白。

    Returns:
        移除括號段落並整理空白後的文本。
    """
    clean, _ = parse_inline_tags(text)
    clean = _STAGE_FULLWIDTH_RE.sub("", clean)
    clean = _STAGE_ASCII_RE.sub("", clean)
    return re.sub(r"[ \t]{2,}", " ", clean).strip()


def _build_engine(name: str, sample_rate: int, **kwargs: object) -> TTSEngine:
    """依名稱建構引擎（不檢查可用性）。"""
    if name == "kokoro":
        from src.voice.kokoro_backend import KokoroBackend

        return KokoroBackend(sample_rate=sample_rate, **kwargs)  # type: ignore[arg-type]
    if name == "chattts":
        from src.voice.chatts_backend import ChatTTSBackend

        return ChatTTSBackend(sample_rate=sample_rate, **kwargs)  # type: ignore[arg-type]
    if name == "pyttsx3":
        from src.voice.pyttsx3_backend import Pyttsx3Backend

        return Pyttsx3Backend(sample_rate=sample_rate, **kwargs)  # type: ignore[arg-type]
    if name == "orpheus":
        from src.voice.orpheus_backend import OrpheusBackend

        return OrpheusBackend(sample_rate=sample_rate, **kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"未知的 TTS 引擎：{name!r}，"
        "允許值為 auto、kokoro、chattts、pyttsx3、orpheus"
    )


def list_available_engines(sample_rate: int = 24000) -> Dict[str, bool]:
    """探測各引擎可用性（不載入模型，只檢查依賴與檔案）。

    Returns:
        引擎名稱到可用性的映射。
    """
    status: Dict[str, bool] = {}
    for name in ("kokoro", "chattts", "pyttsx3", "orpheus"):
        try:
            status[name] = _build_engine(name, sample_rate).is_available()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"引擎 {name} 可用性檢查失敗：{exc}")
            status[name] = False
    return status


def select_engine(
    preference: str = "auto", sample_rate: int = 24000, **kwargs: object
) -> TTSEngine:
    """自動選擇 TTS 引擎。

    Args:
        preference: ``auto`` 表依 :data:`ENGINE_ORDER` 選第一個可用者；
            指定名稱時若該引擎不可用則拋出例外。
        sample_rate: 目標取樣率。
        kwargs: 轉交後端建構子的額外參數。

    Returns:
        可用的 :class:`TTSEngine`。

    Raises:
        ValueError: 引擎名稱非法時。
        RuntimeError: 指定引擎不可用，或 auto 下無可用引擎時。
    """
    name = (preference or "auto").strip().lower()
    if name != "auto":
        engine = _build_engine(name, sample_rate, **kwargs)
        if not engine.is_available():
            raise RuntimeError(
                f"TTS 引擎 {name} 不可用：{engine.unavailable_reason()}"  # type: ignore[attr-defined]
            )
        logger.info(f"選用 TTS 引擎：{name}")
        return engine
    for candidate in ENGINE_ORDER:
        try:
            engine = _build_engine(candidate, sample_rate, **kwargs)
        except Exception:  # noqa: BLE001
            continue
        try:
            if engine.is_available():
                logger.info(f"自動選用 TTS 引擎：{candidate}")
                return engine
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"引擎 {candidate} 可用性檢查失敗：{exc}")
    raise RuntimeError(
        "無可用的 TTS 引擎。保底方案為 pyttsx3，"
        "請執行 pip install pyttsx3（Linux 另需 espeak-ng）"
    )
