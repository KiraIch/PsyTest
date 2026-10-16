"""情緒路由器（骨架，Phase 3 實作）。

依對話上下文快速判定情緒類別，並映射為導演指令、
語音參數與肢體參數。決策目標 <10ms。

情緒映射（摘要）：
    shame → 視線下垂、音量降低、語速減慢
    anger → 音量增大、語速加快、肢體僵硬
    fear → 呼吸急促、聲音顫抖、視線游移
    dissociation → 語調平板、眼神空洞、反應延遲
    manic → 語速極快、音量波動大
    flat_affect → 無語調變化、機械式回應
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class EmotionRoute:
    """情緒路由結果。"""

    emotion: str
    intensity: float
    director_hint: Dict[str, object] = field(default_factory=dict)
    voice_params: Dict[str, object] = field(default_factory=dict)
    visual_params: Dict[str, object] = field(default_factory=dict)


class EmotionRouter:
    """情緒路由器（Phase 3 實作）。"""

    def route(self, context: str, client_state: str) -> EmotionRoute:
        """執行緒路由決策（Phase 3 實作）。"""
        raise NotImplementedError("EmotionRouter.route 排定於 Phase 3 實作")
