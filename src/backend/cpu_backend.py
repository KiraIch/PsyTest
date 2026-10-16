"""純 CPU 後端（骨架，Phase 2 實作）。

預定行為：永遠可用，offload 層數為 0，以 CPU 線程數為計算能力指標。
溫度不可讀時以 CPU 使用率 > 90% 作為過載指標。
"""

from __future__ import annotations

from typing import Optional

from src.backend.base import ComputeBackend, DeviceInfo
from src.backend.detector import _cpu_thread_count


class CpuBackend(ComputeBackend):
    """CPU 計算後端。"""

    def get_device_info(self) -> DeviceInfo:
        """回傳 CPU 資訊。"""
        threads = _cpu_thread_count()
        return DeviceInfo(
            backend_type="cpu",
            device_name=f"CPU（{threads} 線程）",
            compute_capability=f"cpu-{threads}threads",
        )

    def get_recommended_gpu_layers(self, model_size_gb: float) -> int:
        """CPU 不 offload，回傳 0。"""
        return 0

    def monitor_temperature(self) -> Optional[float]:
        """CPU 溫度（Phase 2 實作，優先 psutil，否則回傳 None）。"""
        raise NotImplementedError("CpuBackend.monitor_temperature 排定於 Phase 2 實作")

    def throttle_if_needed(self) -> bool:
        """過載時降速（Phase 2 實作）。"""
        raise NotImplementedError("CpuBackend.throttle_if_needed 排定於 Phase 2 實作")
