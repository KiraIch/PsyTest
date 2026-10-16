"""導演系統（骨架，Phase 3 實作）。

LLM 輸出格式為對白文字加 <director>JSON</director> 區段，
描述語氣、音量、語速、音高、停頓、呼吸、非語言聲音與肢體動作。
解析器需分離對白與指令、驗證格式、缺失時套用預設值，
並保證指令與對白情緒一致、符合病理特徵。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


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


class Director:
    """導演指令解析器（Phase 3 實作）。"""

    def parse(self, llm_output: str) -> Tuple[str, DirectorNotes]:
        """分離對白與導演指令（Phase 3 實作）。"""
        raise NotImplementedError("Director.parse 排定於 Phase 3 實作")

    def with_defaults(self) -> DirectorNotes:
        """回傳預設（平淡語氣）導演指令。"""
        return DirectorNotes()
