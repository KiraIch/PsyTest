"""自動偵測系統可用的計算後端。

偵測順序：Metal（macOS Apple Silicon）→ CUDA → ROCm → CPU。
可經由參數或 CLI ``--force-backend`` 手動指定後端。

使用範例：
    python -m src.backend.detector
    python -m src.backend.detector --force-backend cpu
"""

from __future__ import annotations

import argparse
import multiprocessing
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

BackendType = Literal["metal", "cuda", "rocm", "cpu"]

_VALID_BACKENDS: tuple[str, ...] = ("metal", "cuda", "rocm", "cpu")


@dataclass(frozen=True)
class BackendInfo:
    """後端偵測結果。"""

    backend_type: BackendType
    device_name: str
    total_memory: int  # bytes，未知時為 0
    available_memory: int  # bytes，未知時為 0
    compute_capability: str
    recommended_offload_layers: int  # GGUF 模型的 GPU offload 層數，-1 表全部
    forced: bool = False


def get_platform() -> str:
    """回傳正規化平台名稱：``macos``、``windows`` 或 ``linux``。"""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def _is_apple_silicon() -> bool:
    """判斷是否為 Apple Silicon Mac。"""
    if platform.system() != "Darwin":
        return False
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return True
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        brand = result.stdout.strip().lower()
        return "apple" in brand
    except (OSError, subprocess.SubprocessError):
        return False


def _cuda_available() -> tuple[bool, str]:
    """檢查 CUDA 可用性，回傳（是否可用，裝置描述）。"""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                name = "NVIDIA GPU"
            return True, name
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, result.stdout.strip().splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return False, ""


def _rocm_available() -> tuple[bool, str]:
    """檢查 ROCm 可用性，回傳（是否可用，裝置描述）。"""
    try:
        import torch  # type: ignore

        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        if hip_version:
            return True, f"AMD GPU (ROCm {hip_version})"
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("rocm-smi") is not None:
        return True, "AMD GPU (ROCm)"
    return False, ""


def _cpu_thread_count() -> int:
    """取得 CPU 邏輯核心數，失敗時回傳 4。"""
    try:
        count = multiprocessing.cpu_count()
        return count if count > 0 else 4
    except NotImplementedError:
        return 4


def recommend_gpu_layers(backend: BackendType, model_size_gb: float = 1.0) -> int:
    """依後端回傳建議的 GGUF GPU offload 層數。

    Args:
        backend: 後端類型。
        model_size_gb: 模型大小（GB），CPU 後端會忽略此值。

    Returns:
        -1 表全部 offload，0 表純 CPU。
    """
    if backend in {"metal", "cuda", "rocm"}:
        return -1
    return 0


def detect_backend(
    force: Optional[str] = None,
    model_size_gb: float = 1.0,
) -> BackendInfo:
    """偵測可用計算後端。

    Args:
        force: 強制指定後端（``metal``、``cuda``、``rocm``、``cpu``），
            None 表自動偵測。
        model_size_gb: 模型大小（GB），用於估算 offload 層數。

    Returns:
        :class:`BackendInfo` 偵測結果。

    Raises:
        ValueError: ``force`` 非法時。
    """
    if force is not None and force not in _VALID_BACKENDS:
        raise ValueError(
            f"無效的後端：{force!r}，允許值為 {list(_VALID_BACKENDS)}"
        )

    if force is not None:
        backend = force  # type: ignore[assignment]
        return BackendInfo(
            backend_type=backend,
            device_name=f"{backend} (手動指定)",
            total_memory=0,
            available_memory=0,
            compute_capability="forced",
            recommended_offload_layers=recommend_gpu_layers(backend, model_size_gb),
            forced=True,
        )

    if _is_apple_silicon():
        return BackendInfo(
            backend_type="metal",
            device_name="Apple Silicon (Metal)",
            total_memory=0,
            available_memory=0,
            compute_capability="metal",
            recommended_offload_layers=-1,
        )

    cuda_ok, cuda_name = _cuda_available()
    if cuda_ok:
        return BackendInfo(
            backend_type="cuda",
            device_name=cuda_name or "NVIDIA GPU (CUDA)",
            total_memory=0,
            available_memory=0,
            compute_capability="cuda",
            recommended_offload_layers=-1,
        )

    rocm_ok, rocm_name = _rocm_available()
    if rocm_ok:
        return BackendInfo(
            backend_type="rocm",
            device_name=rocm_name,
            total_memory=0,
            available_memory=0,
            compute_capability="rocm",
            recommended_offload_layers=-1,
        )

    threads = _cpu_thread_count()
    return BackendInfo(
        backend_type="cpu",
        device_name=f"CPU（{threads} 線程）",
        total_memory=0,
        available_memory=0,
        compute_capability=f"cpu-{threads}threads",
        recommended_offload_layers=0,
    )


def format_report(info: BackendInfo) -> str:
    """將偵測結果格式化為人類可讀報告。"""
    lines = [
        f"平台：{get_platform()} ({platform.system()} {platform.machine()})",
        f"後端：{info.backend_type}{'（手動指定）' if info.forced else ''}",
        f"裝置：{info.device_name}",
        f"計算能力：{info.compute_capability}",
        f"建議 GPU offload 層數：{info.recommended_offload_layers}",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口：``python -m src.backend.detector [--force-backend cpu]``。"""
    parser = argparse.ArgumentParser(description="偵測 PsyTest 可用的計算後端")
    parser.add_argument(
        "--force-backend",
        choices=list(_VALID_BACKENDS),
        default=None,
        help="手動指定後端（略過自動偵測）",
    )
    args = parser.parse_args(argv)
    try:
        info = detect_backend(force=args.force_backend)
    except ValueError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 2
    print(format_report(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
