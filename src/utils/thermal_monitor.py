"""溫度監控（Phase 6 實作）。

MacBook Air 無風扇散熱管理，策略如下：
    溫度 < 70°C：全速運行（full_speed）
    70 至 80°C：降低負載（reduce：縮小批次、拉長間隔）
    80 至 85°C：暫停生成等待降溫（pause）
    溫度 > 85°C：完全停止並警告（stop）

溫度來源優先順序：psutil 感測器 → 平台指令 → None。
macOS 無 root 時 powermetrics 不可用，此時以 CPU 使用率 > 90%
作為過載替代指標（reduce）。所有判斷皆為純函式邏輯，可測試。

使用範例：
    from src.utils.thermal_monitor import ThermalMonitor

    monitor = ThermalMonitor(throttle_threshold=85.0)
    state = monitor.check()
    if state.action == "pause":
        ...
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

FULL_SPEED_LIMIT = 70.0
REDUCE_LIMIT = 80.0

CPU_OVERLOAD_PERCENT = 90.0


@dataclass(frozen=True)
class ThermalState:
    """溫度狀態。"""

    temperature: Optional[float]
    action: str  # "full_speed" | "reduce" | "pause" | "stop"
    reason: str = ""


def decide_action(
    temperature: Optional[float],
    throttle_threshold: float = 85.0,
    cpu_percent: Optional[float] = None,
) -> ThermalState:
    """依溫度與 CPU 使用率決定處置（純函式）。

    Args:
        temperature: 攝氏溫度，None 表不可讀。
        throttle_threshold: 停止閾值（須大於 80）。
        cpu_percent: CPU 使用率（溫度不可讀時的替代指標）。

    Returns:
        :class:`ThermalState`。
    """
    if temperature is not None:
        if temperature > throttle_threshold:
            return ThermalState(temperature, "stop", f"{temperature:.1f}°C 超過上限 {throttle_threshold}°C")
        if temperature >= REDUCE_LIMIT:
            return ThermalState(temperature, "pause", f"{temperature:.1f}°C 達暫停區間")
        if temperature >= FULL_SPEED_LIMIT:
            return ThermalState(temperature, "reduce", f"{temperature:.1f}°C 達降載區間")
        return ThermalState(temperature, "full_speed", "溫度正常")
    if cpu_percent is not None and cpu_percent > CPU_OVERLOAD_PERCENT:
        return ThermalState(None, "reduce", f"CPU 使用率 {cpu_percent:.0f}% 過高（溫度不可讀）")
    return ThermalState(None, "full_speed", "溫度不可讀且負載正常")


class ThermalMonitor:
    """溫度監控器。"""

    def __init__(self, throttle_threshold: float = 85.0) -> None:
        """初始化監控器。

        Args:
            throttle_threshold: 停止閾值（攝氏，須大於 80）。

        Raises:
            ValueError: 閾值非法時。
        """
        if throttle_threshold <= REDUCE_LIMIT:
            raise ValueError("throttle_threshold 必須大於 80")
        self.throttle_threshold = throttle_threshold

    def get_temperature(self) -> Optional[float]:
        """讀取溫度（攝氏），不可讀時回傳 None（不拋出例外）。"""
        try:
            import psutil

            temps = psutil.sensors_temperatures() or {}
            for entries in temps.values():
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if isinstance(current, (int, float)) and current > 0:
                        return float(current)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"psutil 溫度讀取失敗：{exc}")
        platform_value = self._platform_temperature()
        return platform_value

    def _platform_temperature(self) -> Optional[float]:
        """平台指令讀取（macOS sysctl 備援，失敗回傳 None）。"""
        if sys.platform != "darwin":
            return None
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return None  # 該值為節流等級非溫度，僅確認指令可用
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    def get_cpu_percent(self) -> Optional[float]:
        """讀取 CPU 使用率，失敗回傳 None。"""
        try:
            import psutil

            return float(psutil.cpu_percent(interval=0.5))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"CPU 使用率讀取失敗：{exc}")
            return None

    def check(self) -> ThermalState:
        """檢查溫度並回傳處置建議（永不拋出例外）。"""
        try:
            temperature = self.get_temperature()
            cpu_percent = self.get_cpu_percent() if temperature is None else None
            state = decide_action(temperature, self.throttle_threshold, cpu_percent)
            if state.action in {"pause", "stop"}:
                logger.warning(f"溫度處置：{state.action}（{state.reason}）")
            return state
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"溫度檢查失敗，視為全速：{exc}")
            return ThermalState(None, "full_speed", "檢查失敗")
