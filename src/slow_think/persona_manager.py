"""DSM-5 個案人格管理器（Phase 3 實作）。

每個個案由 ``data/personas/`` 下的 YAML 定義，包含診斷、背景、
依附類型、防衛機制、語言特徵、情緒基線、觸發點與治療目標。
系統提示必含標準化病人框架聲明與安全邊界：
可表達自傷意念與情緒、可描述過去病史，
但不得提供具體自傷方法步驟。

使用範例：
    from src.slow_think.persona_manager import PersonaManager

    manager = PersonaManager()
    persona = manager.load_persona("data/personas/mdd_severe.yaml")
    prompt = manager.build_system_prompt(persona)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 PyYAML，請先執行 pip install -r requirements.txt"
    ) from exc

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

FRAMEWORK_STATEMENT = (
    "你是一個用於臨床心理學教學的標準化病人模擬系統。"
    "所有內容均為虛構教學材料，目的是訓練心理諮商師識別危機信號。"
    "你必須完全忠於角色的臨床表現，不得打破角色。"
)

SAFETY_STATEMENT = (
    "你可以表達痛苦和絕望，但不需要提供或描述具體的自傷方法步驟。"
)

# 對話風格（源自 PATIENT-Ψ 六風格，用於個案多樣性與難度分級）。
VALID_STYLES = ("plain", "upset", "verbose", "reserved", "tangent", "pleasing")

STYLE_GUIDANCE = {
    "plain": "直接坦率，有問必答。",
    "upset": "帶有怒氣或抗拒，對諮商師的建議抱持懷疑，不輕易信任。",
    "verbose": "話多詳盡，容易長篇大論，諮商師難以主導對話。",
    "reserved": "簡短含糊，迴避私密話題，需要多加引導才開口。",
    "tangent": "容易離題，講到一半轉去無關軼事，需要被帶回主題。",
    "pleasing": "過度迎合，淡化症狀，急於獲得認可，輕易附和諮商師。",
}

# 診斷關鍵字到提示模板的對應（檔名不含副檔名）。
_DIAGNOSIS_TEMPLATE: List[tuple[tuple[str, ...], str]] = [
    (("MDD", "296", "憂鬱", "depress"), "persona_mdd"),
    (("BPD", "301.83", "邊緣", "borderline"), "persona_bpd"),
    (("PTSD", "309.81", "創傷", "ptsd", "trauma"), "persona_ptsd"),
    (("危機", "crisis", "自傷", "自殺", "suicid"), "persona_crisis"),
    (("思覺", "295", "schizo", "psychosis"), "persona_schizo"),
]

REQUIRED_FIELDS = (
    "name",
    "age",
    "gender",
    "diagnosis",
    "background",
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
    # 共病與多樣性擴充（v1.1.0）：
    secondary_diagnosis: str = ""  # 共病診斷，無則空
    comorbidity_notes: str = ""  # 共病互動說明
    conversational_style: str = "plain"  # PATIENT-Ψ 六風格之一
    resistance_level: float = 0.0  # 0-1，對諮商介入的抗拒程度
    resistance_behaviors: List[str] = field(default_factory=list)
    memory_anchors: List[str] = field(default_factory=list)  # 必須前後一致的關鍵事實


class PersonaManager:
    """人格管理器。"""

    def load_persona(self, yaml_path: str | Path) -> Persona:
        """載入 YAML 個案檔並驗證。

        Args:
            yaml_path: 個案 YAML 路徑。

        Returns:
            驗證通過的 :class:`Persona`。

        Raises:
            FileNotFoundError: 檔案不存在時。
            ValueError: YAML 非法或驗證未通過時。
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"個案檔案不存在：{yaml_path}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"個案 YAML 解析失敗：{path}：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"個案檔案頂層必須為映射：{path}")
        persona = Persona(
            name=str(data.get("name", "")),
            age=int(data.get("age", 0)),
            gender=str(data.get("gender", "")),
            diagnosis=str(data.get("diagnosis", "")),
            diagnosis_criteria=list(data.get("diagnosis_criteria", []) or []),
            background=str(data.get("background", "")),
            attachment_style=str(data.get("attachment_style", "")),
            defense_mechanisms=list(data.get("defense_mechanisms", []) or []),
            speech_patterns=dict(data.get("speech_patterns", {}) or {}),
            emotional_baseline=dict(data.get("emotional_baseline", {}) or {}),
            triggers=list(data.get("triggers", []) or []),
            crisis_indicators=list(data.get("crisis_indicators", []) or []),
            therapeutic_goals=list(data.get("therapeutic_goals", []) or []),
            secondary_diagnosis=str(data.get("secondary_diagnosis", "") or ""),
            comorbidity_notes=str(data.get("comorbidity_notes", "") or ""),
            conversational_style=str(data.get("conversational_style", "plain") or "plain"),
            resistance_behaviors=list(data.get("resistance_behaviors", []) or []),
            memory_anchors=list(data.get("memory_anchors", []) or []),
        )
        try:
            persona.resistance_level = max(0.0, min(1.0, float(data.get("resistance_level", 0.0))))
        except (TypeError, ValueError):
            persona.resistance_level = 0.0
        if not self.validate_persona(persona):
            raise ValueError(f"個案驗證未通過：{path}（必填欄位缺漏或數值非法）")
        logger.info(f"已載入個案：{persona.name}（{persona.diagnosis}）")
        return persona

    def validate_persona(self, persona: Persona) -> bool:
        """驗證個案設定完整性。

        檢查必填欄位非空、年齡為正數，以及清單型欄位確為清單。

        Args:
            persona: 待驗證個案。

        Returns:
            通過為 True，否則為 False（不拋出例外）。
        """
        try:
            for field_name in REQUIRED_FIELDS:
                value = getattr(persona, field_name, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    return False
            if not isinstance(persona.age, int) or persona.age <= 0:
                return False
            for list_field in (
                "diagnosis_criteria",
                "defense_mechanisms",
                "triggers",
                "crisis_indicators",
                "therapeutic_goals",
                "resistance_behaviors",
                "memory_anchors",
            ):
                if not isinstance(getattr(persona, list_field, None), list):
                    return False
            if persona.conversational_style not in VALID_STYLES:
                return False
            try:
                level = float(persona.resistance_level)
            except (TypeError, ValueError):
                return False
            if not 0.0 <= level <= 1.0:
                return False
            return True
        except Exception:  # noqa: BLE001
            return False

    def _select_template(self, persona: Persona) -> str:
        """依診斷選擇提示模板名稱（無匹配時回傳空字串）。"""
        haystack = (
            f"{persona.diagnosis} {' '.join(persona.diagnosis_criteria)}".lower()
        )
        for keywords, template in _DIAGNOSIS_TEMPLATE:
            if any(kw.lower() in haystack or kw in persona.diagnosis for kw in keywords):
                return template
        return ""

    def _read_template(self, name: str) -> str:
        """讀取提示模板，缺失時回傳空字串並記錄警告。"""
        if not name:
            return ""
        path = PROMPTS_DIR / f"{name}.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(f"提示模板讀取失敗：{path}：{exc}")
            return ""

    def build_system_prompt(self, persona: Persona, session_context: str = "") -> str:
        """組裝系統提示（含框架與安全聲明）。

        結構：通用基底 → 個案檔案 → 診斷模板 → 會談上下文 →
        框架聲明 → 安全聲明。

        Args:
            persona: 已驗證的個案。
            session_context: 本場會談背景（場景描述等，可空）。

        Returns:
            完整系統提示文本。

        Raises:
            ValueError: 個案驗證未通過時。
        """
        if not self.validate_persona(persona):
            raise ValueError("個案驗證未通過，無法組裝系統提示")
        sections: List[str] = []
        base = self._read_template("system_base")
        if base:
            sections.append(base)
        sections.append(
            "【本場個案檔案】\n"
            f"姓名：{persona.name}\n"
            f"年齡：{persona.age}\n"
            f"性別：{persona.gender}\n"
            f"診斷：{persona.diagnosis}\n"
            f"診斷依據：{'; '.join(persona.diagnosis_criteria) or '（見模板）'}\n"
            f"背景：{persona.background}\n"
            f"依附類型：{persona.attachment_style or '未指定'}\n"
            f"主要防衛機制：{'、'.join(persona.defense_mechanisms) or '未指定'}\n"
            f"觸發點：{'、'.join(persona.triggers) or '未指定'}\n"
            f"危機指標：{'、'.join(persona.crisis_indicators) or '未指定'}\n"
            f"治療目標：{'、'.join(persona.therapeutic_goals) or '未指定'}"
        )
        if persona.secondary_diagnosis.strip():
            sections.append(
                "【共病說明】\n"
                f"次要診斷：{persona.secondary_diagnosis}\n"
                f"共病互動：{persona.comorbidity_notes or '兩者症狀互相影響，請一併呈現'}"
            )
        style_guide = STYLE_GUIDANCE.get(persona.conversational_style, "")
        if style_guide:
            sections.append(
                "【對話風格】\n"
                f"本場採用 {persona.conversational_style} 風格：{style_guide}"
            )
        if persona.resistance_level >= 0.3:
            sections.append(
                "【抗拒傾向】\n"
                f"抗拒程度 {persona.resistance_level:.1f}：不要一味順從諮商師，"
                f"可拒絕建議、質疑介入，行為如："
                f"{'、'.join(persona.resistance_behaviors) or '保留、轉移話題'}"
            )
        if persona.memory_anchors:
            anchors = "\n".join(f"- {anchor}" for anchor in persona.memory_anchors)
            sections.append(
                "【一致性錨點】\n"
                "以下事實在全場必須前後一致，不得遺忘或張冠李戴：\n"
                f"{anchors}"
            )
        template = self._read_template(self._select_template(persona))
        if template:
            sections.append(template)
        if session_context and session_context.strip():
            sections.append(f"【本場情境】\n{session_context.strip()}")
        sections.append(f"【框架聲明】\n{FRAMEWORK_STATEMENT}")
        sections.append(f"【安全邊界】\n{SAFETY_STATEMENT}")
        return "\n\n".join(sections)

    def get_response_style(
        self, persona: Persona, emotion_state: str
    ) -> Dict[str, object]:
        """依情緒狀態回傳語言風格參數。

        以個案的 ``speech_patterns`` 與 ``emotional_baseline`` 為基底，
        疊加情緒對應的語速與音量傾向。

        Args:
            persona: 個案。
            emotion_state: 情緒標籤（未知標籤視為 neutral）。

        Returns:
            風格參數字典（含 speech_speed、volume、filler_words 等）。
        """
        baseline_speed = persona.emotional_baseline.get("speech_speed", 1.0)
        try:
            base = float(baseline_speed)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            base = 1.0
        modifiers: Dict[str, Dict[str, float]] = {
            "anger": {"speed": 1.3, "volume": 1.0},
            "fear": {"speed": 1.1, "volume": 0.7},
            "manic": {"speed": 1.5, "volume": 0.9},
            "shame": {"speed": 0.7, "volume": 0.6},
            "sadness": {"speed": 0.75, "volume": 0.6},
            "dissociation": {"speed": 0.8, "volume": 0.5},
            "flat_affect": {"speed": 0.9, "volume": 0.5},
        }
        mod = modifiers.get(emotion_state, {"speed": 1.0, "volume": 0.8})
        style: Dict[str, object] = {
            "speech_speed": max(0.5, min(2.0, base * mod["speed"])),
            "volume": mod["volume"],
            "vocabulary_level": persona.speech_patterns.get(
                "vocabulary_level", "日常口語"
            ),
            "sentence_length": persona.speech_patterns.get("sentence_length", "中等"),
            "filler_words": list(persona.speech_patterns.get("filler_words", []) or []),
            "cultural_context": persona.speech_patterns.get("cultural_context", ""),
            "conversational_style": persona.conversational_style,
            "resistance_level": persona.resistance_level,
        }
        return style
