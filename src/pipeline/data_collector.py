"""多模態數據收集器（骨架，Phase 5 實作）。

負責彙整文字、音訊、視覺、心理標註與嵌入向量，
組成統一的輪次數據結構（見規格書 4.6 節）。
"""

from __future__ import annotations

from typing import Any, Dict


class DataCollector:
    """數據收集器（Phase 5 實作）。"""

    def collect_turn(self, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集並正規化單輪數據（Phase 5 實作）。"""
        raise NotImplementedError("DataCollector.collect_turn 排定於 Phase 5 實作")
