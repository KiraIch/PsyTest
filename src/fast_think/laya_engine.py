"""Laya 421M 決策引擎（骨架，Phase 2 實作）。

Laya 為非自回歸決策模型，輸入為問題加選項，輸出為決策結果。
載入設定：n_ctx=512，temperature=0.0（greedy decoding）。
BF16 在不支援的硬體上需轉換為 FP16（Phase 2 處理）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DecisionResult:
    """是否回應決策。"""

    should_speak: bool
    urgency: float
    emotion_tag: str


@dataclass(frozen=True)
class EmotionDecision:
    """情緒反應決策。"""

    emotion: str
    intensity: float
    defense_mechanism: str


@dataclass(frozen=True)
class PacingDecision:
    """對話節奏決策。"""

    pause_before_response: float
    speech_speed: float
    should_interrupt: bool
    response_length_hint: str  # "brief" | "moderate" | "detailed"


@dataclass(frozen=True)
class CrisisAssessment:
    """危機等級評估。"""

    level: int  # 0-5
    triggers: List[str] = field(default_factory=list)


class LayaEngine:
    """Laya 421M 載入與推理（Phase 2 實作）。"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_gpu_layers: int = -1,
    ) -> None:
        """初始化引擎參數。

        Args:
            model_path: GGUF 權重路徑。
            n_ctx: 上下文長度。
            n_gpu_layers: GPU offload 層數。
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

    def should_respond(self, context: str) -> DecisionResult:
        """判斷是否需要虛擬個案回應（Phase 2 實作）。"""
        raise NotImplementedError("LayaEngine.should_respond 排定於 Phase 2 實作")

    def determine_emotion(self, context: str, client_state: str) -> EmotionDecision:
        """判斷個案情緒反應（Phase 2 實作）。"""
        raise NotImplementedError("LayaEngine.determine_emotion 排定於 Phase 2 實作")

    def determine_pacing(self, context: str, conversation_turn: int) -> PacingDecision:
        """決定對話節奏（Phase 2 實作）。"""
        raise NotImplementedError("LayaEngine.determine_pacing 排定於 Phase 2 實作")

    def assess_crisis_level(self, response_text: str) -> CrisisAssessment:
        """評估危機等級 0-5（Phase 2 實作）。"""
        raise NotImplementedError(
            "LayaEngine.assess_crisis_level 排定於 Phase 2 實作"
        )
