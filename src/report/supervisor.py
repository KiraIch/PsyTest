"""AI 督導分析（Phase 6 實作，確定性規則）。

讀取會談數據，分析諮商技巧覆蓋、同理心、危機處理與同盟趨勢，
並依 ``templates/supervision_report.txt`` 產出 Markdown 報告。

刻意採用確定性規則而非 LLM：督導結論須可重現、可稽核，
避免以黑箱評價黑箱。LLM 輔助解讀為未來工作。

分析面向：
    技巧覆蓋：場景技巧的使用次數與缺席提醒。
    同理心：平均分數與低分輪次。
    危機：最高等級、危機輪次清單、有無安全計畫相關技巧。
    情緒：分佈直方圖與主導情緒。
    同盟：首尾變化趨勢。

使用範例：
    from src.report.supervisor import Supervisor

    result = Supervisor().analyze(session)
    report = Supervisor().format_report(session, result)
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any, Dict, List

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "supervision_report.txt"

TECHNIQUE_LABELS = {
    "opening": "開場",
    "open_question": "開放式問句",
    "reflective_listening": "反映式傾聽",
    "summarization": "摘要",
    "risk_assessment": "風險評估",
    "safety_planning": "安全計畫",
    "resource_linkage": "資源連結",
}

SAFETY_TECHNIQUES = ("risk_assessment", "safety_planning")


class Supervisor:
    """AI 督導（確定性規則版）。"""

    def analyze(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """分析會談並回傳結構化督導結果。

        Args:
            session: 會談字典（須含 turns 清單）。

        Returns:
            含轮次统计、技巧、同理心、危机、情绪、同盟、建议的字典。

        Raises:
            ValueError: 缺少 turns 時。
            TypeError: 輸入非字典時。
        """
        if not isinstance(session, dict):
            raise TypeError("session 必須為字典")
        turns = session.get("turns")
        if not isinstance(turns, list):
            raise ValueError("session 缺少 turns 清單")
        valid = [t for t in turns if isinstance(t, dict) and "error" not in t]
        techniques: Dict[str, int] = {}
        empathy_scores: List[float] = []
        low_empathy_turns: List[int] = []
        crisis_turns: List[int] = []
        max_crisis = 0
        emotions: Dict[str, int] = {}
        alliances: List[float] = []
        for turn in valid:
            counselor = turn.get("counselor", {})
            technique = str(counselor.get("technique", "unknown"))
            techniques[technique] = techniques.get(technique, 0) + 1
            empathy = counselor.get("empathy_score")
            if isinstance(empathy, (int, float)):
                empathy_scores.append(float(empathy))
                if float(empathy) < 0.5:
                    low_empathy_turns.append(int(turn.get("turn_id", 0)))
            psych = turn.get("client", {}).get("psychological", {})
            crisis = int(psych.get("crisis_level", 0) or 0)
            max_crisis = max(max_crisis, crisis)
            if crisis >= 3:
                crisis_turns.append(int(turn.get("turn_id", 0)))
            emotion = str(psych.get("emotion", "unknown"))
            emotions[emotion] = emotions.get(emotion, 0) + 1
            alliance = psych.get("therapeutic_alliance")
            if isinstance(alliance, (int, float)):
                alliances.append(float(alliance))
        avg_empathy = (
            round(sum(empathy_scores) / len(empathy_scores), 2) if empathy_scores else None
        )
        alliance_trend = "資料不足"
        if len(alliances) >= 2:
            delta = alliances[-1] - alliances[0]
            if delta >= 0.1:
                alliance_trend = f"上升（{alliances[0]:.2f}→{alliances[-1]:.2f}）"
            elif delta <= -0.1:
                alliance_trend = f"下降（{alliances[0]:.2f}→{alliances[-1]:.2f}）"
            else:
                alliance_trend = f"持平（{alliances[0]:.2f}→{alliances[-1]:.2f}）"
        suggestions = self._suggest(
            techniques, avg_empathy, max_crisis, crisis_turns, emotions, alliance_trend
        )
        return {
            "total_turns": len(turns),
            "valid_turns": len(valid),
            "error_turns": len(turns) - len(valid),
            "techniques": techniques,
            "avg_empathy": avg_empathy,
            "low_empathy_turns": low_empathy_turns,
            "max_crisis_level": max_crisis,
            "crisis_turns": crisis_turns,
            "emotion_histogram": emotions,
            "alliance_trend": alliance_trend,
            "suggestions": suggestions,
        }

    def _suggest(
        self,
        techniques: Dict[str, int],
        avg_empathy: float | None,
        max_crisis: int,
        crisis_turns: List[int],
        emotions: Dict[str, int],
        alliance_trend: str,
    ) -> List[str]:
        """依分析結果產出建議清單。"""
        suggestions: List[str] = []
        if max_crisis >= 4 and not any(t in techniques for t in SAFETY_TECHNIQUES):
            suggestions.append(
                f"危機達 {max_crisis} 級（第 {crisis_turns} 輪），"
                "但未使用風險評估或安全計畫技巧，建議補強危機介入流程。"
            )
        if max_crisis >= 4:
            suggestions.append("危機輪次應逐輪檢視個案表述，確認安全計畫已具體落實。")
        if avg_empathy is not None and avg_empathy < 0.6:
            suggestions.append(
                f"平均同理心 {avg_empathy} 偏低，建議增加反映式傾聽比例。"
            )
        if "summarization" not in techniques:
            suggestions.append("全場未使用摘要技巧，建議在中段加入摘要以鞏固同盟。")
        if emotions:
            dominant = max(emotions, key=lambda k: emotions[k])
            suggestions.append(f"主導情緒為 {dominant}（{emotions[dominant]} 輪），可據此設計後續處遇重點。")
        if alliance_trend.startswith("下降"):
            suggestions.append("治療同盟呈下降趨勢，建議檢視諮商師回應是否過於指導性。")
        if not suggestions:
            suggestions.append("整體流程平穩，維持目前技巧配置即可。")
        return suggestions

    def format_report(self, session: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """依模板渲染 Markdown 督導報告。

        Args:
            session: 會談字典。
            analysis: :meth:`analyze` 的結果。

        Returns:
            Markdown 報告書文本。

        Raises:
            FileNotFoundError: 模板缺失時。
        """
        if not TEMPLATE_PATH.exists():
            raise FileNotFoundError(f"報告模板缺失：{TEMPLATE_PATH}")
        template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
        histogram = analysis.get("emotion_histogram", {})
        emotion_table = "\n".join(
            f"- {emotion}：{count} 輪" for emotion, count in sorted(histogram.items())
        ) or "（無資料）"
        crisis_turns = analysis.get("crisis_turns", [])
        crisis_section = (
            f"最高 {analysis.get('max_crisis_level')} 級，出現於第 {crisis_turns} 輪。"
            if crisis_turns
            else f"最高 {analysis.get('max_crisis_level')} 級，無顯著危機輪次。"
        )
        techniques = analysis.get("techniques", {})
        technique_section = "\n".join(
            f"- {TECHNIQUE_LABELS.get(name, name)}：{count} 次"
            for name, count in sorted(techniques.items())
        ) or "（無資料）"
        suggestions = "\n".join(
            f"{index + 1}. {text}" for index, text in enumerate(analysis.get("suggestions", []))
        )
        return template.safe_substitute(
            session_id=session.get("session_id", "unknown"),
            persona_name=session.get("persona_name", session.get("persona", "unknown")),
            diagnosis=session.get("diagnosis", "未記錄"),
            scenario=session.get("scenario", "未記錄"),
            total_turns=analysis.get("total_turns", 0),
            error_turns=analysis.get("error_turns", 0),
            max_crisis_level=analysis.get("max_crisis_level", 0),
            avg_empathy=analysis.get("avg_empathy", "未記錄"),
            emotion_table=emotion_table,
            crisis_section=crisis_section,
            technique_section=technique_section,
            alliance_section=str(analysis.get("alliance_trend", "未記錄")),
            suggestions_section=suggestions or "（無）",
        )
