"""快思層：極速決策模型。

負責載入 Laya 421M（``models/laya-421m-bf16.gguf``），
提供對話節奏控制與情緒路由決策，單次推理目標 <50ms。
完整實作排定於 Phase 2（引擎）與 Phase 3（智能層）。
"""

from __future__ import annotations

__all__: list[str] = []
