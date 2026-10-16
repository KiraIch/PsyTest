"""溫度監控（骨架，Phase 6 實作）。

MacBook Air 無風扇散熱管理：
    溫度 < 70°C：全速運行
    70 至 80°C：降低 batch_size，增加生成間隔
    80 至 85°C：暫停生成，等待降溫
    溫度 > 85°C：完全停止並警告

macOS 經 powermetrics 或 istats 讀取，Linux 經 sysfs 或 psutil，
不可用時以 CPU 使用率 > 90% 作為過載指標。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ThermalState:
    """溫度狀態。"""

    temperature: Optional[float]
    action: str  # "full_speed" | "reduce" | "pause" | "stop"


class ThermalMonitor:
    """溫度監控器（Phase 6 實作）。"""

    def __init__(self, throttle_threshold: float = 85.0) -> None:
        """初始化監控器。

        Args:
            throttle_threshold: 降速閾值（攝氏）。
        """
        if throttle_threshold <= 0:
            raise ValueError("throttle_threshold 必須為正數")
        self.throttle_threshold = throttle_threshold

    def check(self) -> ThermalState:
        """檢查溫度並回傳處置建議（Phase 6 實作）。"""
        raise NotImplementedError("ThermalMonitor.check 排定於 Phase 6 實作")
