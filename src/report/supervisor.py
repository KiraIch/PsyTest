"""AI 督導分析（骨架，Phase 6 實作）。

預定行為：讀取會談數據，分析諮商技巧、同理心、危機處理，
並依模板產出督導報告。
"""

from __future__ import annotations

from typing import Any, Dict


class Supervisor:
    """AI 督導（Phase 6 實作）。"""

    def analyze(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """分析會談並回傳督導結果（Phase 6 實作）。"""
        raise NotImplementedError("Supervisor.analyze 排定於 Phase 6 實作")
