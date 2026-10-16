"""PsyTest 硬體後端抽象層。

提供跨平台的計算後端偵測與統一介面，支援 Metal、CUDA、ROCm 與 CPU。
"""

from __future__ import annotations

__all__ = ["detector"]
