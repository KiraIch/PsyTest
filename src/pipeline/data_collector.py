"""多模態數據收集器（Phase 5 實作）。

負責彙整文字、音訊、視覺、心理標註與嵌入向量，
正規化為統一的輪次數據結構（欄位定義見規格書 4.6 節）。
所有數值欄位皆驗證範圍，缺漏以 None 標示而非偽造。

使用範例：
    from src.pipeline.data_collector import DataCollector

    turn = DataCollector().collect_turn({...})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

REQUIRED_TOP_FIELDS = ("session_id", "turn_id", "timestamp")

EMOTION_LEVEL_RANGE = (0.0, 1.0)
CRISIS_LEVEL_RANGE = (0, 5)


def _clamp_float(value: object, default: Optional[float] = None) -> Optional[float]:
    """轉浮點，失敗回傳預設。"""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result


class DataCollector:
    """數據收集器。"""

    def collect_turn(self, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集並正規化單輪數據。

        Args:
            turn_data: 原始輪次數據，至少含 session_id、turn_id、timestamp。

        Returns:
            正規化後的輪次字典（含 counselor、client、laya_decisions、
            embedding_vectors 四大區塊，缺漏區塊以空字典補齊）。

        Raises:
            ValueError: 缺少頂層必要欄位時。
            TypeError: 輸入非字典時。
        """
        if not isinstance(turn_data, dict):
            raise TypeError("turn_data 必須為字典")
        for key in REQUIRED_TOP_FIELDS:
            if key not in turn_data:
                raise ValueError(f"缺少必要欄位：{key}")
        normalized: Dict[str, Any] = {
            "session_id": str(turn_data["session_id"]),
            "turn_id": int(turn_data["turn_id"]),
            "timestamp": str(turn_data["timestamp"]),
            "counselor": self._normalize_counselor(turn_data.get("counselor", {})),
            "client": self._normalize_client(turn_data.get("client", {})),
            "laya_decisions": self._normalize_laya(turn_data.get("laya_decisions", {})),
            "embedding_vectors": self._normalize_embeddings(
                turn_data.get("embedding_vectors", {})
            ),
        }
        if "error" in turn_data:
            normalized["error"] = str(turn_data["error"])
        return normalized

    def _normalize_counselor(self, data: object) -> Dict[str, Any]:
        """正規化諮商師區塊。"""
        raw = data if isinstance(data, dict) else {}
        empathy = _clamp_float(raw.get("empathy_score"))
        if empathy is not None:
            empathy = max(0.0, min(1.0, empathy))
        return {
            "text": str(raw.get("text", "")),
            "technique": str(raw.get("technique", "open_question")),
            "empathy_score": empathy,
        }

    def _normalize_client(self, data: object) -> Dict[str, Any]:
        """正規化個案區塊（含心理標註範圍檢查）。"""
        raw = data if isinstance(data, dict) else {}
        psych = raw.get("psychological", {})
        psych = psych if isinstance(psych, dict) else {}
        intensity = _clamp_float(psych.get("emotion_intensity"))
        if intensity is not None:
            intensity = max(0.0, min(1.0, intensity))
        crisis = psych.get("crisis_level", 0)
        try:
            crisis = max(0, min(5, int(crisis)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            crisis = 0
        alliance = _clamp_float(psych.get("therapeutic_alliance"))
        if alliance is not None:
            alliance = max(0.0, min(1.0, alliance))
        audio = raw.get("audio", {})
        audio = audio if isinstance(audio, dict) else {}
        visual = raw.get("visual", {})
        visual = visual if isinstance(visual, dict) else {}
        return {
            "text": str(raw.get("text", "")),
            "director_notes": dict(raw.get("director_notes", {}) or {}),
            "audio": {
                "file_path": audio.get("file_path"),
                "duration": _clamp_float(audio.get("duration")),
                "features": dict(audio.get("features", {}) or {}),
            },
            "visual": {
                "gaze_direction": str(visual.get("gaze_direction", "direct")),
                "facial_expression": str(visual.get("facial_expression", "")),
                "body_posture": str(visual.get("body_posture", "open")),
                "arousal_level": _clamp_float(visual.get("arousal_level"), 0.5),
            },
            "psychological": {
                "emotion": str(psych.get("emotion", "neutral")),
                "emotion_intensity": intensity,
                "defense_mechanism": str(psych.get("defense_mechanism", "unknown")),
                "crisis_level": crisis,
                "therapeutic_alliance": alliance,
            },
        }

    def _normalize_laya(self, data: object) -> Dict[str, Any]:
        """正規化 Laya 決策區塊。"""
        raw = data if isinstance(data, dict) else {}
        pause = _clamp_float(raw.get("pause_duration"), 0.0)
        pause = max(0.0, min(30.0, pause if pause is not None else 0.0))
        urgency = _clamp_float(raw.get("response_urgency"), 0.5)
        urgency = max(0.0, min(1.0, urgency if urgency is not None else 0.5))
        return {
            "pause_duration": pause,
            "should_respond": bool(raw.get("should_respond", True)),
            "response_urgency": urgency,
            "pacing": str(raw.get("pacing", "moderate")),
        }

    def _normalize_embeddings(self, data: object) -> Dict[str, Any]:
        """正規化嵌入向量區塊（僅保留清單或 None）。"""
        raw = data if isinstance(data, dict) else {}
        normalized: Dict[str, Any] = {}
        for key in ("audio_embedding", "text_embedding", "multimodal_embedding"):
            value = raw.get(key)
            if value is None:
                normalized[key] = None
            elif isinstance(value, dict) and "data" in value:
                normalized[key] = value  # 已量化格式原樣保留
            else:
                try:
                    normalized[key] = [float(v) for v in value]  # type: ignore[union-attr]
                except (TypeError, ValueError):
                    normalized[key] = None
        return normalized

    def collect_session(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """正規化整場會談的所有輪次。

        Args:
            turns: 原始輪次清單。

        Returns:
            正規化後的輪次清單（單輪失敗不中斷，錯誤記入該輪）。
        """
        collected: List[Dict[str, Any]] = []
        for index, turn in enumerate(turns):
            try:
                collected.append(self.collect_turn(turn))
            except (ValueError, TypeError) as exc:
                collected.append(
                    {
                        "session_id": turn.get("session_id", "unknown") if isinstance(turn, dict) else "unknown",
                        "turn_id": index + 1,
                        "timestamp": "",
                        "error": f"正規化失敗：{exc}",
                    }
                )
        return collected
