"""後端偵測測試：確認能正確偵測當前硬體，且無 GPU 時可回退 CPU。"""

from __future__ import annotations

from src.backend.detector import (
    detect_backend,
    format_report,
    recommend_gpu_layers,
)


def test_detect_backend() -> None:
    """確認自動偵測回傳合法後端類型。"""
    info = detect_backend()
    assert info.backend_type in {"metal", "cuda", "rocm", "cpu"}
    assert isinstance(info.device_name, str) and info.device_name
    assert isinstance(info.recommended_offload_layers, int)


def test_cpu_fallback() -> None:
    """確認手動指定 CPU 可作為最終 fallback。"""
    info = detect_backend(force="cpu")
    assert info.backend_type == "cpu"
    assert info.forced is True
    assert info.recommended_offload_layers == 0


def test_force_invalid_backend_raises() -> None:
    """確認非法後端名稱會拋出 ValueError。"""
    import pytest

    with pytest.raises(ValueError):
        detect_backend(force="tpu")


def test_recommend_gpu_layers() -> None:
    """確認各後端 offload 建議值符合規格。"""
    assert recommend_gpu_layers("metal") == -1
    assert recommend_gpu_layers("cuda") == -1
    assert recommend_gpu_layers("rocm") == -1
    assert recommend_gpu_layers("cpu") == 0


def test_format_report_contains_backend() -> None:
    """確認報告包含後端資訊。"""
    info = detect_backend()
    report = format_report(info)
    assert info.backend_type in report
