"""語音導演（Phase 4 實作）。

將導演指令轉換為各 TTS 引擎的具體參數。輸入可為
:class:`DirectorNotes` 或 ``Director.parse`` 回傳的字典，
情緒基底取自 :data:`EMOTION_TO_VOICE`，明確指定的數值優先。

各引擎參數形狀：
    chattts: oral、laugh、break（0 至 7）、speed（註記用）。
    kokoro: voice（sid 整數）、speed、pause_ms（停頓清單）。
    pyttsx3: rate（每分鐘字數）、volume、pause_ms。
    orpheus: tags（``<tag>`` 清單）。

使用範例：
    from src.slow_think.director import Director
    from src.voice.voice_director import VoiceDirector

    dialog, notes = Director().parse(llm_output)
    params = VoiceDirector().to_engine_params(notes.to_dict(), "chattts")
"""

from __future__ import annotations

from typing import Dict, List, Union

from src.slow_think.director import Director, DirectorNotes

EMOTION_TO_VOICE: Dict[str, Dict[str, object]] = {
    "shame": {"volume": 0.6, "speed": 0.7, "pitch_shift": -0.10},
    "anger": {"volume": 1.0, "speed": 1.3, "pitch_shift": 0.05},
    "fear": {"volume": 0.7, "speed": 1.1, "pitch_shift": 0.15, "tremor": 0.3},
    "dissociation": {"volume": 0.5, "speed": 0.8, "pitch_shift": 0.0, "monotone": True},
    "manic": {"volume": 0.9, "speed": 1.5, "pitch_variance": "high"},
    "flat_affect": {"volume": 0.5, "speed": 0.9, "pitch_shift": 0.0, "monotone": True},
}

_VALID_ENGINES = ("chattts", "kokoro", "pyttsx3", "orpheus")

# 語氣關鍵字到情緒的回推（用於僅有 tone 時的基底選擇）。
_TONE_EMOTION = (
    ("shame", ("subdued", "ashamed", "羞愧")),
    ("anger", ("harsh", "angry", "憤怒")),
    ("fear", ("trembling", "fearful", "害怕")),
    ("dissociation", ("hollow", "解離")),
    ("manic", ("pressured", "躁")),
    ("flat_affect", ("mechanical", "monotone", "平板")),
)


def _infer_emotion(tone: str) -> str:
    """由語氣字串回推情緒，未命中時回傳空字串。"""
    lowered = tone.strip().lower()
    for emotion, keywords in _TONE_EMOTION:
        if any(kw in lowered for kw in keywords):
            return emotion
    return ""


def _clamp(value: object, low: float, high: float, default: float) -> float:
    """轉浮點並鉗制，失敗回傳預設。"""
    try:
        return max(low, min(high, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _pause_list(notes: Dict[str, object]) -> List[float]:
    """由 pauses 欄位萃取停頓秒數清單（容忍多種格式）。"""
    result: List[float] = []
    raw = notes.get("pauses", [])
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                result.append(max(0.0, float(item[1]) - float(item[0])))
                continue
            except (TypeError, ValueError):
                pass
        try:
            result.append(max(0.0, float(item)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return result


class VoiceDirector:
    """語音導演。"""

    def from_emotion(self, emotion: str) -> Dict[str, object]:
        """回傳指定情緒的基底語音參數。

        Args:
            emotion: 情緒標籤，未知時回傳中性預設。

        Returns:
            含 volume、speed 等的字典（副本，修改不影響常數）。
        """
        base = EMOTION_TO_VOICE.get(emotion, {"volume": 0.8, "speed": 1.0})
        return dict(base)

    def to_engine_params(
        self,
        director_notes: Union[DirectorNotes, Dict[str, object]],
        engine: str = "chattts",
        emotion: str = "",
    ) -> Dict[str, object]:
        """轉換導演指令為指定引擎參數。

        Args:
            director_notes: 導演指令（物件或字典）。
            engine: 目標引擎，chattts、kokoro、pyttsx3、orpheus。
            emotion: 情緒標籤，未指定時由 tone 回推，皆無則用中性。

        Returns:
            引擎參數字典。

        Raises:
            ValueError: 引擎名稱非法時。
        """
        target = (engine or "chattts").strip().lower()
        if target not in _VALID_ENGINES:
            raise ValueError(
                f"未知的語音引擎：{engine!r}，允許值為 {list(_VALID_ENGINES)}"
            )
        notes: Dict[str, object] = (
            director_notes.to_dict()
            if isinstance(director_notes, DirectorNotes)
            else dict(director_notes)
        )
        resolved_emotion = (emotion or "").strip() or _infer_emotion(
            str(notes.get("tone", ""))
        )
        base = self.from_emotion(resolved_emotion) if resolved_emotion else {
            "volume": 0.8,
            "speed": 1.0,
        }
        volume = _clamp(notes.get("volume", base.get("volume", 0.8)), 0.0, 1.0, 0.8)
        speed = _clamp(notes.get("speed", base.get("speed", 1.0)), 0.5, 2.0, 1.0)
        pauses = _pause_list(notes)
        total_pause = sum(pauses)
        nonverbal = [
            str(item).strip().lower()
            for item in (notes.get("nonverbal", []) or [])
            if isinstance(item, str) or isinstance(item, (int, float))
        ]

        if target == "chattts":
            laugh = 2 if any(k in ("sniffle", "sob", "cry", "laugh") for k in nonverbal) else 0
            return {
                "oral": 2,
                "laugh": laugh,
                "break": min(7, int(total_pause)),
                "speed": speed,
                "volume": volume,
            }
        if target == "kokoro":
            return {
                "voice": notes.get("voice", "sid:46"),
                "speed": speed,
                "volume": volume,
                "pause_ms": [int(p * 1000) for p in pauses],
            }
        if target == "pyttsx3":
            return {
                "rate": int(200 * speed),
                "volume": volume,
                "pause_ms": [int(p * 1000) for p in pauses],
            }
        tags = Director().to_tts_tags(
            DirectorNotes(
                tone=str(notes.get("tone", "neutral")),
                volume=volume,
                speed=speed,
                breathing=str(notes.get("breathing", "normal")),
                nonverbal=list(notes.get("nonverbal", []) or []),
            ),
            engine="orpheus",
        )
        return {"tags": tags, "volume": volume, "speed": speed}
