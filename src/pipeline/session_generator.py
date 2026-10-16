"""單場諮商會談生成器（骨架，Phase 5 實作）。

流程：載入個案與場景 → 初始化引擎 → 開場 → 對話循環
（諮商師說話 → Laya 決策 → LLM 生成 → TTS 合成 → 標註 → 記錄）
→ 摘要 → 輸出。

結束條件：達最大輪數（預設 30）、達最大時間（預設 50 分鐘模擬時間）、
個案情緒達解決狀態、危機介入完成。
"""

from __future__ import annotations

from typing import Any, Dict


class SessionGenerator:
    """單場諮商生成器（Phase 5 實作）。"""

    def __init__(self, config: Dict[str, Any] | object) -> None:
        """初始化生成器。

        Args:
            config: AppConfig 或等效字典。
        """
        self.config = config

    def generate(self, persona_path: str, scenario_path: str) -> Dict[str, Any]:
        """生成單場會談並回傳會談數據（Phase 5 實作）。"""
        raise NotImplementedError("SessionGenerator.generate 排定於 Phase 5 實作")
