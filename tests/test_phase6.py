"""Phase 6 輔助功能測試：溫度監控、記憶體管理、督導報告。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.memory_manager import MemoryManager
from src.utils.thermal_monitor import ThermalMonitor, decide_action


def _session() -> dict:
    """建立合成會談數據。"""
    return {
        "session_id": "s1",
        "persona_name": "測試個案",
        "diagnosis": "測試診斷",
        "scenario": "initial_session",
        "turns": [
            {
                "turn_id": 1,
                "counselor": {"text": "你好", "technique": "open_question", "empathy_score": 0.8},
                "client": {
                    "text": "嗨",
                    "psychological": {
                        "emotion": "sadness",
                        "emotion_intensity": 0.7,
                        "crisis_level": 1,
                        "therapeutic_alliance": 0.5,
                    },
                },
            },
            {
                "turn_id": 2,
                "counselor": {"text": "多說一點", "technique": "risk_assessment", "empathy_score": 0.4},
                "client": {
                    "text": "撐不下去",
                    "psychological": {
                        "emotion": "sadness",
                        "emotion_intensity": 0.9,
                        "crisis_level": 4,
                        "therapeutic_alliance": 0.6,
                    },
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# 溫度監控
# ---------------------------------------------------------------------------


def test_decide_action_thresholds() -> None:
    """四段閾值應正確。"""
    assert decide_action(65.0).action == "full_speed"
    assert decide_action(75.0).action == "reduce"
    assert decide_action(82.0).action == "pause"
    assert decide_action(90.0).action == "stop"


def test_decide_action_cpu_fallback() -> None:
    """無溫度時以 CPU 使用率為替代指標。"""
    assert decide_action(None, cpu_percent=95.0).action == "reduce"
    assert decide_action(None, cpu_percent=50.0).action == "full_speed"


def test_monitor_rejects_low_threshold() -> None:
    """過低閾值應拋出 ValueError。"""
    with pytest.raises(ValueError):
        ThermalMonitor(throttle_threshold=70.0)


def test_monitor_check_never_raises() -> None:
    """check 永不拋出例外且回傳合法動作。"""
    state = ThermalMonitor().check()
    assert state.action in {"full_speed", "reduce", "pause", "stop"}


# ---------------------------------------------------------------------------
# 記憶體管理
# ---------------------------------------------------------------------------


def test_acquire_release_lifecycle() -> None:
    """申請釋放計數與卸載回呼應正確。"""
    unloaded: list[str] = []
    manager = MemoryManager(max_memory_gb=4.0)
    assert manager.acquire("embedding", 1.5, unloader=lambda: unloaded.append("embedding")) is True
    assert manager.acquire("embedding", 1.5) is True
    assert manager.booked_gb() == pytest.approx(1.5)
    manager.release("embedding")
    assert unloaded == []
    manager.release("embedding")
    assert unloaded == ["embedding"]
    assert manager.held_models() == {}


def test_acquire_over_budget_rejected() -> None:
    """超出上限應拒絕而非拋出例外。"""
    manager = MemoryManager(max_memory_gb=2.0)
    assert manager.acquire("big", 3.0) is False
    with pytest.raises(ValueError):
        manager.acquire("", 1.0)
    with pytest.raises(ValueError):
        MemoryManager(max_memory_gb=-1.0)


def test_release_unknown_warns_only() -> None:
    """釋放未持有模型僅警告。"""
    MemoryManager().release("ghost")


def test_check_pressure_levels() -> None:
    """壓力評估應回傳合法等級。"""
    assert MemoryManager().check_pressure() in {"ok", "high", "critical"}


# ---------------------------------------------------------------------------
# 督導報告
# ---------------------------------------------------------------------------


def test_supervisor_analyze() -> None:
    """分析結果應含各面向與危機建議。"""
    from src.report.supervisor import Supervisor

    result = Supervisor().analyze(_session())
    assert result["total_turns"] == 2
    assert result["max_crisis_level"] == 4
    assert result["techniques"]["open_question"] == 1
    assert result["avg_empathy"] == pytest.approx(0.6)
    assert any("危機" in s for s in result["suggestions"])


def test_supervisor_rejects_bad_input() -> None:
    """非法輸入應拋出明確例外。"""
    from src.report.supervisor import Supervisor

    with pytest.raises(TypeError):
        Supervisor().analyze("不是字典")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Supervisor().analyze({})


def test_supervisor_report_renders() -> None:
    """報告應含關鍵章節。"""
    from src.report.supervisor import Supervisor

    supervisor = Supervisor()
    report = supervisor.format_report(_session(), supervisor.analyze(_session()))
    for keyword in ("會談 ID", "情緒分佈", "危機事件", "督導建議", "s1"):
        assert keyword in report


def test_generate_report_script(tmp_path: Path) -> None:
    """report CLI 應可產出檔案。"""
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "generate_report", "scripts/generate_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "session.jsonl"
    source.write_text(json.dumps(_session(), ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "report.md"
    assert module.main(["--input", str(source), "--output", str(output)]) == 0
    assert "督導建議" in output.read_text(encoding="utf-8")
    assert module.main(["--input", str(tmp_path / "missing.jsonl"), "--output", str(output)]) == 2


def test_batch_resource_gate_aborts() -> None:
    """資源達中止條件應中止整批。"""
    from src.config import load_config
    from src.pipeline.batch_generator import BatchGenerator

    class _CriticalMemory:
        def check_pressure(self) -> str:
            return "critical"

    config = load_config("configs/default.yaml")
    batch = BatchGenerator(
        config,
        session_generator=object(),
        memory_manager=_CriticalMemory(),
    )
    with pytest.raises(RuntimeError):
        batch._resource_gate()
