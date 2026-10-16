"""計算後端統一介面。

所有後端（Metal、CUDA、ROCm、CPU）必須實作 :class:`ComputeBackend`。
完整實作排定於 Phase 2 回補；目前提供介面定義供各模組依賴。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviceInfo:
    """裝置資訊。"""

    backend_type: str
    device_name: str
    total_memory: int = 0
    available_memory: int = 0
    compute_capability: str = ""


class ComputeBackend(ABC):
    """計算後端抽象基類。"""

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """回傳裝置資訊。"""
        raise NotImplementedError

    @abstractmethod
    def get_recommended_gpu_layers(self, model_size_gb: float) -> int:
        """依模型大小回傳建議的 GGUF GPU offload 層數。"""
        raise NotImplementedError

    @abstractmethod
    def monitor_temperature(self) -> Optional[float]:
        """回傳目前溫度（攝氏），不支援時回傳 None。"""
        raise NotImplementedError

    @abstractmethod
    def throttle_if_needed(self) -> bool:
        """溫度超標時執行降速，回傳是否已降速。"""
        raise NotImplementedError
