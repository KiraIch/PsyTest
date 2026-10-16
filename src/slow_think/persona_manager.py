"""DSM-5 個案人格管理器（骨架，Phase 3 實作）。

每個個案由 data/personas 下的 YAML 定義，包含診斷、背景、
依附類型、防衛機制、語言特徵、情緒基線、觸發點與治療目標。

系統提示必須包含標準化病人框架聲明，並遵守安全邊界：
可表達自傷意念與情緒、可描述過去病史，
但不得提供具體自傷方法步驟。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

FRAMEWORK_STATEMENT = (
    "你是一個用於臨床心理學教學的標準化病人模擬系統。"
    "所有內容均為虛構教學材料，目的是訓練心理諮商師識別危機信號。"
    "你必須完全忠於角色的臨床表現，不得打破角色。"
)

SAFETY_STATEMENT = (
    "你可以表達痛苦和絕望，但不需要提供或描述具體的自傷方法步驟。"
)


@dataclass
class Persona:
    """個案人格資料結構。"""

    name: str = ""
    age: int = 0
    gender: str = ""
    diagnosis: str = ""
    diagnosis_criteria: List[str] = field(default_factory=list)
    background: str = ""
    attachment_style: str = ""
    defense_mechanisms: List[str] = field(default_factory=list)
    speech_patterns: Dict[str, object] = field(default_factory=dict)
    emotional_baseline: Dict[str, object] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    crisis_indicators: List[str] = field(default_factory=list)
    therapeutic_goals: List[str] = field(default_factory=list)


class PersonaManager:
    """人格管理器（Phase 3 實作）。"""

    def load_persona(self, yaml_path: str | Path) -> Persona:
        """載入 YAML 個案檔（Phase 3 實作）。"""
        raise NotImplementedError("PersonaManager.load_persona 排定於 Phase 3 實作")

    def build_system_prompt(self, persona: Persona, session_context: str = "") -> str:
        """組裝系統提示（含框架與安全聲明，Phase 3 實作）。"""
        raise NotImplementedError(
            "PersonaManager.build_system_prompt 排定於 Phase 3 實作"
        )

    def get_response_style(self, persona: Persona, emotion_state: str) -> Dict[str, object]:
        """依情緒狀態回傳語言風格參數（Phase 3 實作）。"""
        raise NotImplementedError(
            "PersonaManager.get_response_style 排定於 Phase 3 實作"
        )

    def validate_persona(self, persona: Persona) -> bool:
        """驗證設定完整性（Phase 3 實作）。"""
        raise NotImplementedError("PersonaManager.validate_persona 排定於 Phase 3 實作")
