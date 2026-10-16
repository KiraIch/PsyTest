"""數據管線測試（Phase 5 啟用端到端案例）。"""

from __future__ import annotations

import pytest

from src.config import load_config


def test_config_loads() -> None:
    """確認預設設定可載入且通過驗證。"""
    config = load_config("configs/default.yaml")
    assert config.project_name == "PsyTest"
    assert config.pipeline.max_turns_per_session > 0


def test_single_turn() -> None:
    """單輪對話生成（Phase 5 實作後啟用）。"""
    pytest.skip("SessionGenerator 排定於 Phase 5 實作")


def test_full_session() -> None:
    """完整會談生成（Phase 5 實作後啟用）。"""
    pytest.skip("SessionGenerator 排定於 Phase 5 實作")


def test_output_format() -> None:
    """輸出 JSONL 格式驗證（Phase 5 實作後啟用）。"""
    pytest.skip("OutputFormatter 排定於 Phase 5 實作")
