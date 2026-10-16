"""輸出格式化（骨架，Phase 5 實作）。

支援 JSONL（主要）、Parquet、WAV 與 manifest.json。
輸出結構：
    data/output/manifest.json
    data/output/sessions/session_*.jsonl
    data/output/audio/session_*_turn_*.wav
    data/output/annotations/session_*_annotations.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class OutputFormatter:
    """輸出格式化器（Phase 5 實作）。"""

    def __init__(self, output_dir: str | Path) -> None:
        """初始化輸出目錄。

        Args:
            output_dir: 輸出根目錄。
        """
        self.output_dir = Path(output_dir)

    def write_session(self, session: Dict[str, Any]) -> Path:
        """寫入單場會談（Phase 5 實作）。"""
        raise NotImplementedError("OutputFormatter.write_session 排定於 Phase 5 實作")

    def write_manifest(self, sessions: List[Dict[str, Any]]) -> Path:
        """寫入數據集索引（Phase 5 實作）。"""
        raise NotImplementedError("OutputFormatter.write_manifest 排定於 Phase 5 實作")
