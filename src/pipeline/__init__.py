"""數據生成管線：單場生成、批量排程、數據收集與輸出格式化。

完整實作排定於 Phase 5。每一步失敗不得導致管線崩潰，
應記錄錯誤並跳過該輪，繼續下一輪。
"""

from __future__ import annotations

__all__: list[str] = []
