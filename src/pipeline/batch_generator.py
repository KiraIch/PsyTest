"""批量生成排程器（骨架，Phase 5 實作）。

功能：讀取生成計畫、監控記憶體與溫度、斷點續傳、並行控制、
進度顯示。預設串行生成（安全優先）。
"""

from __future__ import annotations

from typing import Any, Dict, List


class BatchGenerator:
    """批量生成排程器（Phase 5 實作）。"""

    def __init__(self, config: Dict[str, Any] | object) -> None:
        """初始化排程器。

        Args:
            config: AppConfig 或等效字典。
        """
        self.config = config

    def run(
        self,
        personas: List[str],
        sessions_per_persona: int,
        output_dir: str,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """執行批量生成（Phase 5 實作）。"""
        raise NotImplementedError("BatchGenerator.run 排定於 Phase 5 實作")
