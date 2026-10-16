"""AMD ROCm 後端（骨架，Phase 2 實作）。

預定行為：經由 torch.version.hip 或 rocm-smi 偵測，
全量 offload（n_gpu_layers=-1）。
"""

from __future__ import annotations

from typing import Optional

from src.backend.base import ComputeBackend, DeviceInfo


class RocmBackend(ComputeBackend):
    """ROCm 計算後端。"""

    def get_device_info(self) -> DeviceInfo:
        """回傳 AMD 裝置資訊（Phase 2 實作）。"""
        raise NotImplementedError("RocmBackend 排定於 Phase 2 實作")

    def get_recommended_gpu_layers(self, model_size_gb: float) -> int:
        """ROCm 一律全量 offload，回傳 -1。"""
        return -1

    def monitor_temperature(self) -> Optional[float]:
        """讀取 GPU 溫度（Phase 2 實作）。"""
        raise NotImplementedError("RocmBackend 排定於 Phase 2 實作")

    def throttle_if_needed(self) -> bool:
        """溫度超標時降速（Phase 2 實作）。"""
        raise NotImplementedError("RocmBackend 排定於 Phase 2 實作")
