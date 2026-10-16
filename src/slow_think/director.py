"""導演系統（Phase 3 實作）。

LLM 輸出格式為對白文字加 ``<director>JSON</director>`` 區段，
描述語氣、音量、語速、音高、停頓、呼吸、非語言聲音與肢體動作。
解析器負責分離對白與指令、驗證並正規化欄位；
LLM 遺漏或寫壞指令時套用預設值（平淡語氣），絕不中斷管線。

一致性保證（文件層約束，強制執行於 Phase 5 管線）：
    導演指令須與對白情緒一致，病理特徵須正確反映
   （憂鬱慢、躁症快），無觸發事件不得出現劇烈情緒轉折。

使用範例：
    from src.slow_think.director import Director

    dialog, notes = Director().parse(output_text)
    tags = Director().to_tts_tags(notes, engine="orpheus")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DIRECTOR_RE = re.compile(
    r"<director>(.*?)</director>", re.IGNORECASE | re.DOTALL
)

VALID_BREATHING = (
    "normal",
    "shallow_rapid",
    "deep_sigh",
    "held",
)

# 非語言聲音到 TTS 標籤的映射（Orpheus 用 <tag>，ChatTTS 用 [tag]）。
_NONVERBAL_TAGS: Dict[str, str] = {
    "sniffle": "cry",
    "sob": "cry",
    "cry": "cry",
    "throat_clear": "hesitate",
    "lip_tremble": "hesitate",
    "sigh": "breath",
    "gasp": "breath",
}

_TONE_TAGS: Dict[str, str] = {
    "whisper": "whisper",
    "angry": "angry",
    "harsh": "angry",
    "flat": "flat",
    "mechanical": "flat",
    "hollow": "flat",
    "sorrowful": "cry",
    "trembling": "cry",
}


@dataclass
class DirectorNotes:
    """導演指令結構。"""

    tone: str = "neutral"
    volume: float = 0.8
    speed: float = 1.0
    pitch_shift: float = 0.0
    pauses: list = field(default_factory=list)
    breathing: str = "normal"
    nonverbal: list = field(default_factory=list)
    body_language: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """轉為字典。"""
        return {
            "tone": self.tone,
            "volume": self.volume,
            "speed": self.speed,
            "pitch_shift": self.pitch_shift,
            "pauses": list(self.pauses),
            "breathing": self.breathing,
            "nonverbal": list(self.nonverbal),
            "body_language": list(self.body_language),
        }


def _clamp(value: object, low: float, high: float, default: float) -> float:
    """將值轉為浮點並鉗制，失敗時回傳預設。"""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _as_list(value: object) -> list:
    """正規化為清單（非清單輸入視為空清單）。"""
    return list(value) if isinstance(value, list) else []


def coerce_notes(raw: object) -> DirectorNotes:
    """將任意解析結果正規化為合法導演指令。

    Args:
        raw: ``json.loads`` 的結果（期望為字典）。

    Returns:
        正規化後的 :class:`DirectorNotes`，非法輸入時回傳預設值。
    """
    defaults = DirectorNotes()
    if not isinstance(raw, dict):
        return defaults
    breathing = raw.get("breathing", defaults.breathing)
    if breathing not in VALID_BREATHING:
        breathing = defaults.breathing
    return DirectorNotes(
        tone=str(raw.get("tone", defaults.tone) or defaults.tone),
        volume=_clamp(raw.get("volume"), 0.0, 1.0, defaults.volume),
        speed=_clamp(raw.get("speed"), 0.5, 2.0, defaults.speed),
        pitch_shift=_clamp(raw.get("pitch_shift"), -0.5, 0.5, defaults.pitch_shift),
        pauses=_as_list(raw.get("pauses")),
        breathing=str(breathing),
        nonverbal=_as_list(raw.get("nonverbal")),
        body_language=_as_list(raw.get("body_language")),
    )


class Director:
    """導演指令解析器。"""

    def parse(self, llm_output: str) -> Tuple[str, DirectorNotes]:
        """分離對白文字與導演指令。

        Args:
            llm_output: LLM 原始輸出。

        Returns:
            （對白文本，導演指令）。缺失或損壞時指令為預設值，
            對白為去除標籤後的原文。

        Raises:
            ValueError: 輸入為空時。
        """
        if not llm_output or not llm_output.strip():
            raise ValueError("llm_output 不可為空")
        match = _DIRECTOR_RE.search(llm_output)
        if match is None:
            # 無完整指令：可能是 max_tokens 截斷，移除尾部殘缺標籤後取用對白。
            cleaned = re.sub(r"<director.*$", "", llm_output, flags=re.IGNORECASE | re.DOTALL).strip()
            if cleaned:
                logger.warning("導演指令缺失或遭截斷，套用預設值")
                return cleaned, DirectorNotes()
            logger.warning("LLM 未生成導演指令，套用預設值")
            return llm_output.strip(), DirectorNotes()
        dialog = _DIRECTOR_RE.sub("", llm_output).strip()
        try:
            raw = json.loads(match.group(1).strip() or "{}")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"導演指令 JSON 解析失敗，套用預設值：{exc}")
            return dialog, DirectorNotes()
        return dialog, coerce_notes(raw)

    def with_defaults(self) -> DirectorNotes:
        """回傳預設（平淡語氣）導演指令。"""
        return DirectorNotes()

    def to_tts_tags(
        self, notes: DirectorNotes, engine: str = "orpheus"
    ) -> List[str]:
        """將導演指令轉為 TTS 引擎標籤清單。

        Args:
            notes: 導演指令。
            engine: ``orpheus``（``<tag>``）或 ``chattts``（``[tag]``），
                其他值視為 orpheus。

        Returns:
            標籤字串清單（去重並保序）。
        """
        collected: List[str] = []
        tone_key = notes.tone.strip().lower()
        for keyword, tag in _TONE_TAGS.items():
            if keyword in tone_key:
                collected.append(tag)
        for item in notes.nonverbal:
            tag = _NONVERBAL_TAGS.get(str(item).strip().lower())
            if tag:
                collected.append(tag)
        if notes.breathing in {"deep_sigh", "shallow_rapid", "held"}:
            collected.append("breath")
        if notes.speed >= 1.4:
            collected.append("angry" if "angry" not in collected else "hesitate")
        if not collected:
            return []
        seen: List[str] = []
        for tag in collected:
            if tag not in seen:
                seen.append(tag)
        if engine.strip().lower() == "chattts":
            return [f"[{tag}]" for tag in seen]
        return [f"<{tag}>" for tag in seen]
