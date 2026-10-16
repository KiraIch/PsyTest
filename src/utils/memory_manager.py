"""記憶體管理（Phase 6 實作）。

24GB 記憶體需同時調度多個模型，採按需載入與卸載：
    生成對白時僅載入 LLM（Ollama 管理）
    標註時載入 EmbeddingGemma 2
    決策時載入 Laya 421M
以引用計數管理模型生命週期，計數歸零時調用卸載回呼。
GPU 留給模型推理，音訊前處理一律在 CPU 完成。

另提供 ``check_pressure`` 讀取系統記憶體壓力，
供批量排程在每場開始前評估是否需等待或釋放。

使用範例：
    from src.utils.memory_manager import MemoryManager

    manager = MemoryManager(max_memory_gb=12.0)
    if manager.acquire("embedding", 1.5, unloader=engine.unload):
        try:
            ...
        finally:
            manager.release("embedding")
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _Holder:
    """模型持有紀錄。"""

    size_gb: float
    refs: int = 0
    unloader: Optional[Callable[[], None]] = None


class MemoryManager:
    """引用計數式記憶體管理器。"""

    def __init__(self, max_memory_gb: Optional[float] = None) -> None:
        """初始化管理器。

        Args:
            max_memory_gb: 自管模型上限（GB），None 表不設帳面上限
                （仍可查詢系統壓力）。

        Raises:
            ValueError: 上限非正數時。
        """
        if max_memory_gb is not None and max_memory_gb <= 0:
            raise ValueError("max_memory_gb 必須為正數")
        self.max_memory_gb = max_memory_gb
        self._holders: Dict[str, _Holder] = {}

    def booked_gb(self) -> float:
        """回傳帳面已登記的模型總量（GB）。"""
        return sum(holder.size_gb for holder in self._holders.values() if holder.refs > 0)

    def acquire(
        self,
        name: str,
        size_gb: float,
        unloader: Optional[Callable[[], None]] = None,
    ) -> bool:
        """申請載入模型（引用計數加一）。

        超出帳面上限時拒絕並回傳 False，不拋出例外。
        同名重複申請僅增加計數（size 取首次登記值）。

        Args:
            name: 模型名稱（如 embedding、laya）。
            size_gb: 模型大小（GB，須為正數）。
            unloader: 計數歸零時調用的卸載回呼。

        Returns:
            核准為 True，超出上限為 False。

        Raises:
            ValueError: 名稱為空或大小非正數時。
        """
        if not name or not name.strip():
            raise ValueError("name 不可為空")
        if size_gb <= 0:
            raise ValueError("size_gb 必須為正數")
        holder = self._holders.get(name)
        if holder is None:
            if (
                self.max_memory_gb is not None
                and self.booked_gb() + size_gb > self.max_memory_gb
            ):
                logger.warning(
                    f"記憶體申請被拒：{name} {size_gb}GB，"
                    f"已登記 {self.booked_gb():.1f}/{self.max_memory_gb}GB"
                )
                return False
            holder = _Holder(size_gb=size_gb, unloader=unloader)
            self._holders[name] = holder
        holder.refs += 1
        if unloader is not None:
            holder.unloader = unloader
        return True

    def release(self, name: str) -> None:
        """釋放模型（計數減一，歸零時調用卸載回呼）。

        未登記或計數已為零時僅記錄警告，不拋出例外。
        """
        holder = self._holders.get(name)
        if holder is None or holder.refs <= 0:
            logger.warning(f"釋放未持有模型：{name}")
            return
        holder.refs -= 1
        if holder.refs == 0 and holder.unloader is not None:
            try:
                holder.unloader()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"模型 {name} 卸載回呼失敗：{exc}")

    def held_models(self) -> Dict[str, int]:
        """回傳持有中模型與計數。"""
        return {name: h.refs for name, h in self._holders.items() if h.refs > 0}

    def system_pressure(self) -> Dict[str, float]:
        """讀取系統記憶體狀態（失敗回傳空字典）。

        Returns:
            含 percent、available_gb 的字典。
        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "percent": float(memory.percent),
                "available_gb": round(float(memory.available) / (1024**3), 2),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"系統記憶體讀取失敗：{exc}")
            return {}

    def check_pressure(self) -> str:
        """評估系統壓力：ok、high（>85%）、critical（>93%）。"""
        pressure = self.system_pressure()
        percent = pressure.get("percent", 0.0)
        if percent > 93.0:
            gc.collect()
            return "critical"
        if percent > 85.0:
            return "high"
        return "ok"
