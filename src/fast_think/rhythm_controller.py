"""對話節奏控制器（骨架，Phase 3 實作）。

管理停頓、打斷、沉默與語速變化。
狀態機：IDLE → LISTENING → PROCESSING → PAUSING → RESPONDING → IDLE。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RhythmState(str, Enum):
    """節奏狀態。"""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    PAUSING = "PAUSING"
    RESPONDING = "RESPONDING"


@dataclass
class RhythmPlan:
    """單輪節奏計畫。"""

    pause_before_response: float = 0.0
    should_interrupt: bool = False
    speech_speed: float = 1.0
    filler_sound: str = "none"


class RhythmController:
    """對話節奏控制器（Phase 3 實作）。"""

    def __init__(self) -> None:
        """初始化為 IDLE 狀態。"""
        self.state: RhythmState = RhythmState.IDLE

    def plan(self, pause: float, should_interrupt: bool, speed: float) -> RhythmPlan:
        """依 Laya 節奏決策產生執行計畫（Phase 3 實作）。"""
        raise NotImplementedError("RhythmController.plan 排定於 Phase 3 實作")
