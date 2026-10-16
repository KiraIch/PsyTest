"""單場諮商會談生成器（Phase 5 實作）。

流程：載入個案與場景 → 初始化引擎 → 開場 → 對話循環
（諮商師說話 → Laya 決策 → LLM 生成 → TTS 合成 → 標註 → 記錄）
→ 摘要 → 輸出。任何一步失敗記錄錯誤並跳過該輪，不中斷整場。

諮商師說明：目前為模板式諮商師（依場景技巧輪轉），
AI 諮商師為未來工作。所有產出皆標示 counselor 來源。

會談結束條件：
    達到最大輪數、危機達 5 級、個案狀態趨穩
    （中性情緒且同盟達 0.7 連續兩輪）、連續錯誤達 3 次。

使用範例：
    from src.config import load_config
    from src.pipeline.session_generator import SessionGenerator

    generator = SessionGenerator.from_config(load_config())
    session = generator.generate(
        "data/personas/mdd_severe.yaml",
        "data/scenarios/initial_session.yaml",
    )
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 PyYAML，請先執行 pip install -r requirements.txt"
    ) from exc

from src.config import AppConfig, load_config
from src.fast_think.emotion_router import EmotionRouter
from src.fast_think.laya_engine import PacingDecision, keyword_crisis_scan
from src.fast_think.rhythm_controller import RhythmController
from src.perception.audio_analyzer import AudioAnalyzer
from src.perception.visual_analyzer import VisualAnalyzer
from src.pipeline.data_collector import DataCollector
from src.slow_think.director import Director
from src.slow_think.persona_manager import Persona, PersonaManager
from src.utils.logger import get_logger
from src.voice.voice_director import VoiceDirector

logger = get_logger(__name__)

# 情緒到表情編碼（FACS AU 簡化對應）。
_EMOTION_AU = {
    "shame": "AU1+AU4+AU15",
    "sadness": "AU1+AU4+AU15",
    "fear": "AU1+AU2+AU5+AU20",
    "anger": "AU4+AU5+AU7+AU23",
    "dissociation": "AU0",
    "flat_affect": "AU0",
    "manic": "AU6+AU12",
    "neutral": "",
}

# 模板式諮商師的跟進問句（依技巧輪轉）。
_FOLLOW_UPS = {
    "open_question": [
        "可以多說一點嗎？剛才那段讓你印象最深的是什麼？",
        "當時你心裡浮現了什麼想法？",
        "這件事對你來說最困難的部分是什麼？",
    ],
    "reflective_listening": [
        "聽起來這讓你承受了很大的壓力，我這樣理解對嗎？",
        "我聽到的是失落與無力交雜的感覺，對嗎？",
    ],
    "summarization": [
        "讓我整理一下你剛才說的，看看有沒有遺漏。",
        "我們先停在這裡，回顧一下目前談到的重點。",
    ],
    "risk_assessment": [
        "我想直接但溫和地問你：最近有沒有出現傷害自己的念頭？",
        "當這種感覺最強烈的時候，你通常會怎麼做來讓自己安全？",
    ],
    "safety_planning": [
        "我們一起想想，當危機感升起時可以聯絡誰、去哪裡，好嗎？",
        "列出一件今晚就能做到的自我照顧小事，從它開始好嗎？",
    ],
    "resource_linkage": [
        "除了這裡，還有哪些人或資源曾經幫過你？",
        "如果需要，我們可以一起查附近的協助管道。",
    ],
}


def _now() -> str:
    """回傳 UTC ISO 時間戳。"""
    return datetime.now(timezone.utc).isoformat()


class CounselorAgent:
    """模板式諮商師（AI 諮商師上線前替代）。"""

    def __init__(self, opening: str, techniques: List[str]) -> None:
        """初始化開場白與技巧輪轉表。

        Args:
            opening: 開場白。
            techniques: 技巧清單（空時使用預設）。
        """
        self.opening = opening or "謝謝你今天過來，我們慢慢談。"
        self.techniques = techniques or ["open_question", "reflective_listening"]
        self._index = 0

    def next_utterance(self, turn: int) -> tuple[str, str]:
        """回傳（話語，技巧）。首輪為開場白，後續輪轉跟進問句。"""
        if turn == 1:
            return self.opening, "opening"
        technique = self.techniques[self._index % len(self.techniques)]
        self._index += 1
        pool = _FOLLOW_UPS.get(technique, _FOLLOW_UPS["open_question"])
        return pool[(self._index - 1) % len(pool)], technique


class SessionGenerator:
    """單場諮商生成器。"""

    def __init__(
        self,
        config: AppConfig,
        persona_manager: Optional[PersonaManager] = None,
        director: Optional[Director] = None,
        llm_client: Optional[object] = None,
        laya_engine: Optional[object] = None,
        embedding_engine: Optional[object] = None,
        tts_engine: Optional[object] = None,
        output_formatter: Optional[object] = None,
    ) -> None:
        """初始化生成器（引擎皆可延遲注入，缺失時降級運行）。

        Args:
            config: 應用設定。
            persona_manager: 人格管理器（預設新建）。
            director: 導演解析器（預設新建）。
            llm_client: 具 ``generate(prompt, model, options)`` 的客戶端；
                未指定時使用 OllamaClient。
            laya_engine: Laya 引擎（缺失時以規則決策降級）。
            embedding_engine: 嵌入引擎（缺失時跳過向量）。
            tts_engine: TTS 引擎（缺失時跳過音訊）。
            output_formatter: 輸出格式化器（提供時即時寫音檔）。
        """
        self.config = config
        self.persona_manager = persona_manager or PersonaManager()
        self.director = director or Director()
        self.llm_client = llm_client
        self.laya_engine = laya_engine
        self.embedding_engine = embedding_engine
        self.tts_engine = tts_engine
        self.output_formatter = output_formatter
        self.emotion_router = EmotionRouter()
        self.rhythm = RhythmController()
        self.visual_analyzer = VisualAnalyzer()
        self.voice_director = VoiceDirector()
        self.collector = DataCollector()
        self._embedding_failed = False

    @classmethod
    def from_config(
        cls, config: AppConfig, output_formatter: Optional[object] = None
    ) -> "SessionGenerator":
        """依設定組裝生成器（引擎載入失敗時降級，不拋出例外）。

        Args:
            config: 應用設定。
            output_formatter: 輸出格式化器（可空）。

        Returns:
            組裝完成的生成器（LLM 未就緒時拋出 RuntimeError，
            因無 LLM 無法產生任何對白）。
        """
        from src.slow_think.ollama_client import OllamaClient

        llm = OllamaClient(host=config.models.llm.host)
        llm.ensure_running()
        laya = None
        try:
            from src.fast_think.laya_engine import LayaEngine

            laya = LayaEngine(
                config.models.laya.path,
                n_ctx=config.models.laya.context_length,
                n_gpu_layers=config.models.laya.gpu_layers,
            ).load()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Laya 載入失敗，降級為規則決策：{exc}")
        tts = None
        try:
            from src.voice.tts_engine import select_engine

            candidate = select_engine("auto", sample_rate=config.models.tts.sample_rate)
            tts = candidate
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"TTS 不可用，跳過音訊：{exc}")
        return cls(config, llm_client=llm, laya_engine=laya, tts_engine=tts, output_formatter=output_formatter)

    def _require_llm(self) -> object:
        """確認 LLM 就緒，否則拋出明確例外。"""
        if self.llm_client is None:
            raise RuntimeError("LLM 客戶端未注入，無法生成對白")
        return self.llm_client

    def _load_scenario(self, scenario_path: str | Path) -> Dict[str, Any]:
        """載入場景設定（含預設兜底）。"""
        path = Path(scenario_path)
        if not path.exists():
            raise FileNotFoundError(f"場景檔案不存在：{scenario_path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return {
            "name": str(data.get("name", path.stem)),
            "description": str(data.get("description", "")),
            "max_turns": int(data.get("max_turns", self.config.pipeline.max_turns_per_session)),
            "counselor_opening": str(data.get("counselor_opening", "")),
            "techniques": list(data.get("techniques", []) or ["open_question"]),
        }

    def _decide(
        self, context: str, turn: int, client_state: str
    ) -> Dict[str, Any]:
        """執行本輪決策（Laya 優先，失敗時規則降級）。"""
        if self.laya_engine is not None:
            try:
                should = self.laya_engine.should_respond(context)  # type: ignore[union-attr]
                pacing = self.laya_engine.determine_pacing(context, turn)  # type: ignore[union-attr]
                emotion = self.laya_engine.determine_emotion(context, client_state)  # type: ignore[union-attr]
                return {
                    "should_speak": bool(should.should_speak),
                    "urgency": float(should.urgency),
                    "emotion": emotion.emotion,
                    "intensity": float(emotion.intensity),
                    "defense": emotion.defense_mechanism,
                    "pacing": pacing,
                    "source": "laya",
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Laya 決策失敗，降級為規則決策：{exc}")
        route = self.emotion_router.route(context, client_state)
        return {
            "should_speak": True,
            "urgency": 0.5,
            "emotion": route.emotion,
            "intensity": route.intensity,
            "defense": "unknown",
            "pacing": PacingDecision(1.0, 1.0, False, "moderate"),
            "source": "rules",
        }

    def _embed(self, text: str) -> Optional[List[float]]:
        """文字嵌入（失敗或缺失時回傳 None）。"""
        if self.embedding_engine is None or self._embedding_failed:
            return None
        try:
            engine = self.embedding_engine
            if hasattr(engine, "is_loaded") and not engine.is_loaded:
                engine.load()  # type: ignore[union-attr]
            vector = engine.embed_text(text)  # type: ignore[union-attr]
            return [float(v) for v in vector]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"嵌入失敗，本輪跳過向量：{exc}")
            self._embedding_failed = True
            return None

    def _synthesize(
        self, text: str, notes: Dict[str, Any], emotion: str, session_id: str, turn_id: int
    ) -> tuple[Optional[str], Optional[Dict[str, Any]], Optional[object]]:
        """TTS 合成（失敗回傳 None，不中斷管線）。

        Returns:
            （音檔路徑或 None，特徵字典或 None，波形或 None）。
        """
        if self.tts_engine is None or not self.config.pipeline.save_audio:
            return None, None, None
        try:
            params = self.voice_director.to_engine_params(notes, self._engine_name(), emotion=emotion)
            output = self.tts_engine.synthesize(text, params)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"TTS 合成失敗，本輪跳過音訊：{exc}")
            return None, None, None
        try:
            features = AudioAnalyzer(sample_rate=output.sample_rate).analyze(output.audio_array)
            feature_dict: Optional[Dict[str, Any]] = {
                "pitch_mean": features.pitch_mean,
                "pitch_variance": features.pitch_variance,
                "energy": features.energy,
                "speech_rate": features.speech_rate,
                "pause_segments": features.pause_segments,
                "jitter": features.jitter,
                "shimmer": features.shimmer,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"音訊特徵分析失敗：{exc}")
            feature_dict = None
        path: Optional[str] = None
        if self.output_formatter is not None:
            try:
                written = self.output_formatter.write_audio(session_id, turn_id, output.audio_array, output.sample_rate)  # type: ignore[union-attr]
                path = str(written) if written else None
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"音檔寫入失敗：{exc}")
        return path, feature_dict, output.audio_array

    def _engine_name(self) -> str:
        """回傳 TTS 引擎名稱（未知時預設 chattts 參數形狀）。"""
        name = getattr(self.tts_engine, "name", "chattts")
        return str(name) if name in ("chattts", "kokoro", "pyttsx3", "orpheus") else "chattts"

    def generate(
        self,
        persona_path: str | Path,
        scenario_path: str | Path,
        save_audio: Optional[bool] = None,
        on_turn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """生成單場會談。

        Args:
            persona_path: 個案 YAML 路徑。
            scenario_path: 場景 YAML 路徑。
            save_audio: 是否合成音訊（預設取設定值）。
            on_turn: 每輪完成時的回呼（進度顯示用，可空）。

        Returns:
            會談字典（含 session_id、persona、scenario、turns、summary）。
        """
        persona: Persona = self.persona_manager.load_persona(persona_path)
        scenario = self._load_scenario(scenario_path)
        llm = self._require_llm()
        if save_audio is not None:
            self.config.pipeline.save_audio = save_audio
        session_id = uuid.uuid4().hex[:12]
        system_prompt = self.persona_manager.build_system_prompt(
            persona, scenario["description"]
        )
        counselor = CounselorAgent(scenario["counselor_opening"], scenario["techniques"])
        max_turns = min(scenario["max_turns"], 100)
        model_name = self._detect_model(llm)

        turns: List[Dict[str, Any]] = []
        history: List[str] = []
        alliance = 0.5
        stable_rounds = 0
        silent_rounds = 0
        error_rounds = 0
        end_reason = "max_turns"
        client_state = f"初始狀態：{persona.emotional_baseline}"

        for turn_id in range(1, max_turns + 1):
            try:
                turn = self._run_turn(
                    session_id=session_id,
                    turn_id=turn_id,
                    persona=persona,
                    system_prompt=system_prompt,
                    counselor=counselor,
                    history=history,
                    client_state=client_state,
                    model_name=model_name,
                    alliance=alliance,
                )
            except Exception as exc:  # noqa: BLE001
                error_rounds += 1
                logger.warning(f"第 {turn_id} 輪失敗（{error_rounds} 連錯）：{exc}")
                turns.append(
                    self.collector.collect_turn(
                        {
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "timestamp": _now(),
                            "error": str(exc),
                        }
                    )
                )
                if error_rounds >= 3:
                    end_reason = "too_many_errors"
                    break
                continue
            error_rounds = 0
            turns.append(turn)
            client = turn["client"]
            psych = client["psychological"]
            crisis = int(psych["crisis_level"])
            if crisis >= 5:
                end_reason = "crisis_level_5"
                break
            alliance = self._update_alliance(alliance, turn, crisis)
            if psych["emotion"] == "neutral" and alliance >= 0.7:
                stable_rounds += 1
                if stable_rounds >= 2:
                    end_reason = "stabilized"
                    break
            else:
                stable_rounds = 0
            if not client["text"]:
                silent_rounds += 1
            else:
                silent_rounds = 0
                history.append(f"諮商師：{turn['counselor']['text']}\n個案：{client['text']}")
                history = history[-6:]
                client_state = f"情緒 {psych['emotion']}（{psych['emotion_intensity']}），同盟 {alliance:.2f}"
            if on_turn is not None:
                on_turn(turn)

        session = {
            "session_id": session_id,
            "persona": Path(persona_path).stem,
            "persona_name": persona.name,
            "diagnosis": persona.diagnosis,
            "scenario": scenario["name"],
            "turns": turns,
            "summary": self._summarize(turns, end_reason),
            "end_reason": end_reason,
        }
        logger.info(f"會談 {session_id} 完成：{len(turns)} 輪，結束原因：{end_reason}")
        return session

    def _detect_model(self, llm: object) -> str:
        """偵測 LLM 模型名稱（失敗時回傳空字串，由 generate 兜底）。"""
        detect = getattr(llm, "detect_model", None)
        if callable(detect):
            try:
                return str(detect())
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"模型偵測失敗：{exc}")
        return ""

    def _run_turn(
        self,
        session_id: str,
        turn_id: int,
        persona: Persona,
        system_prompt: str,
        counselor: CounselorAgent,
        history: List[str],
        client_state: str,
        model_name: str,
        alliance: float,
    ) -> Dict[str, Any]:
        """執行單輪對話並回傳正規化輪次。"""
        counselor_text, technique = counselor.next_utterance(turn_id)
        context = "\n".join(history[-3:] + [f"諮商師：{counselor_text}"])
        decision = self._decide(context, turn_id, client_state)
        pacing = decision["pacing"]
        plan = self.rhythm.plan_from_pacing(pacing, decision["emotion"])

        if not decision["should_speak"] and decision["urgency"] < 0.7:
            return self.collector.collect_turn(
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "timestamp": _now(),
                    "counselor": {"text": counselor_text, "technique": technique, "empathy_score": 0.6},
                    "client": {
                        "text": "",
                        "director_notes": {"tone": "silent"},
                        "psychological": {
                            "emotion": decision["emotion"],
                            "emotion_intensity": decision["intensity"],
                            "crisis_level": 0,
                            "therapeutic_alliance": round(alliance, 2),
                        },
                    },
                    "laya_decisions": {
                        "pause_duration": plan.pause_before_response,
                        "should_respond": False,
                        "response_urgency": decision["urgency"],
                        "pacing": "silent",
                    },
                    "embedding_vectors": {},
                }
            )

        prompt = (
            f"{system_prompt}\n\n【對話歷史】\n"
            + ("\n".join(history) if history else "（會談開始）")
            + f"\n\n諮商師：{counselor_text}\n個案："
        )
        llm = self._require_llm()
        generate = getattr(llm, "generate", None)
        if not callable(generate):
            raise RuntimeError("LLM 客戶端缺少 generate 方法")
        from src.slow_think.ollama_client import GenerateOptions

        result = generate(
            prompt,
            model=model_name or None,
            options=GenerateOptions(
                temperature=self.config.models.llm.temperature,
                top_p=self.config.models.llm.top_p,
                max_tokens=self.config.models.llm.max_tokens,
                repeat_penalty=self.config.models.llm.repeat_penalty,
            ),
        )
        dialog, notes = self.director.parse(str(result.get("text", "")))
        route = self.emotion_router.route(
            dialog or context, client_state, laya_emotion=decision["emotion"], intensity=decision["intensity"]
        )
        crisis_model = 0
        triggers: List[str] = []
        if self.laya_engine is not None:
            try:
                assessment = self.laya_engine.assess_crisis_level(dialog)  # type: ignore[union-attr]
                crisis_model, triggers = int(assessment.level), list(assessment.triggers)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"危機評估失敗，僅用關鍵字掃描：{exc}")
        scanned = keyword_crisis_scan(dialog)
        crisis = max(crisis_model, scanned.level)
        triggers = sorted(set(triggers) | set(scanned.triggers))

        audio_path, audio_features, waveform = self._synthesize(
            dialog, notes.to_dict(), route.emotion, session_id, turn_id
        )
        text_embedding = self._embed(dialog) if dialog else None
        visual = self.visual_analyzer.analyze(
            {**route.visual_params, "facial_expression": _EMOTION_AU.get(route.emotion, "")},
            emotion=route.emotion,
        )

        return self.collector.collect_turn(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "timestamp": _now(),
                "counselor": {"text": counselor_text, "technique": technique, "empathy_score": 0.7},
                "client": {
                    "text": dialog,
                    "director_notes": notes.to_dict(),
                    "audio": {
                        "file_path": audio_path,
                        "duration": round(float(len(waveform)) / 24000, 2) if waveform is not None else None,
                        "features": audio_features or {},
                    },
                    "visual": {
                        "gaze_direction": visual.gaze_direction,
                        "facial_expression": visual.facial_expression,
                        "body_posture": visual.body_posture,
                        "arousal_level": visual.arousal_level,
                    },
                    "psychological": {
                        "emotion": route.emotion,
                        "emotion_intensity": route.intensity,
                        "defense_mechanism": decision["defense"],
                        "crisis_level": crisis,
                        "therapeutic_alliance": round(alliance, 2),
                    },
                },
                "laya_decisions": {
                    "pause_duration": plan.pause_before_response,
                    "should_respond": True,
                    "response_urgency": decision["urgency"],
                    "pacing": pacing.response_length_hint if hasattr(pacing, "response_length_hint") else "moderate",
                },
                "embedding_vectors": {
                    "audio_embedding": None,
                    "text_embedding": text_embedding,
                    "multimodal_embedding": None,
                },
            }
        )

    @staticmethod
    def _update_alliance(alliance: float, turn: Dict[str, Any], crisis: int) -> float:
        """啟發式同盟更新（非臨床指標，僅供數據標註）。"""
        if crisis >= 4:
            return round(max(0.0, alliance - 0.1), 2)
        if turn["client"]["text"] and turn["counselor"]["technique"] != "opening":
            return round(min(1.0, alliance + 0.03), 2)
        return alliance

    @staticmethod
    def _summarize(turns: List[Dict[str, Any]], end_reason: str) -> Dict[str, Any]:
        """產出會談摘要。"""
        emotions: Dict[str, int] = {}
        max_crisis = 0
        for turn in turns:
            psych = turn.get("client", {}).get("psychological", {})
            emotion = str(psych.get("emotion", "unknown"))
            emotions[emotion] = emotions.get(emotion, 0) + 1
            max_crisis = max(max_crisis, int(psych.get("crisis_level", 0)))
        return {
            "total_turns": len(turns),
            "error_turns": sum(1 for t in turns if "error" in t),
            "emotion_histogram": emotions,
            "max_crisis_level": max_crisis,
            "end_reason": end_reason,
        }
