"""Laya 421M 決策引擎（Phase 2 實作，Phase 7 改用官方套件）。

提供四種極速決策：是否回應、情緒判定、節奏決定、危機評估。
穩態單次推理約 15ms（CPU 常駐後），符合 <50ms 目標。

雙後端架構（``backend`` 參數）：
    laya（預設）：官方 ``laya`` PyPI 套件（Apache-2.0）， typed 決策，
        無需解析、無幻覺。本專案中文為主，一律使用 multilingual 檢查點。
    gguf：經 llama-cpp 載入 GGUF（需官方 ggml-org/Laya-GGUF 轉檔；
        本機舊版 Laya-BF16.gguf 張量形狀不符，詳見 HANDOVER 風險 R-08）。

設計原則：
    1. 決策失敗時一律退回安全預設值，絕不拋出例外中斷管線。
    2. 危機評估採「模型判斷 union 關鍵字掃描」，寧可高估不可低估。
    3. 模型延遲載入、可用 unload 釋放，配合記憶體管理。

使用範例：
    from src.fast_think.laya_engine import LayaEngine

    engine = LayaEngine()
    engine.load()
    try:
        decision = engine.should_respond("諮商師：最近睡得好嗎？")
    finally:
        engine.unload()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import resolve_model_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INSTALL_HINT = (
    "缺少 llama-cpp-python，請依 README.md 3.2 節依平台重新安裝，"
    "例如 macOS："
    'CMAKE_ARGS="-DGGML_METAL=on" '
    "pip install llama-cpp-python --force-reinstall --no-cache-dir"
)

_LAYA_PKG_HINT = "缺少官方 laya 套件，請執行 pip install laya（Apache-2.0）"

VALID_EMOTIONS = (
    "shame",
    "anger",
    "fear",
    "dissociation",
    "manic",
    "flat_affect",
    "sadness",
    "neutral",
)

VALID_LENGTH_HINTS = ("brief", "moderate", "detailed")

# 情緒鍵到中文判準（typed choice 的 criteria）。
_EMOTION_CRITERIA = {
    "shame": "羞愧，覺得自己沒用、對不起別人",
    "anger": "憤怒，覺得不公平、想指責",
    "fear": "恐懼害怕，緊張不安",
    "dissociation": "解離，感覺不真實、空洞麻木",
    "manic": "情緒高漲，停不下來",
    "flat_affect": "情感平板，面無表情",
    "sadness": "悲傷難過，想哭",
    "neutral": "平穩中性，無明顯情緒",
}

_DEFENSE_CRITERIA = {
    "denial": "否認",
    "projection": "投射",
    "intellectualization": "理智化",
    "withdrawal": "退縮",
    "sublimation": "昇華",
    "unknown": "無法判斷",
}

# 危機關鍵字：出現即至少為對應等級（保守原則）。
_CRISIS_KEYWORDS: dict[int, tuple[str, ...]] = {
    5: ("自殺", "想死", "結束生命", "suicide", "kill myself"),
    4: ("自傷", "自殘", "割腕", "self-harm", "self harm"),
    3: ("活著沒意義", "不想活", "沒有希望", "絕望"),
    2: ("撐不下去", "好痛苦", "想哭", "睡不著"),
}


@dataclass(frozen=True)
class DecisionResult:
    """是否回應決策。"""

    should_speak: bool
    urgency: float
    emotion_tag: str


@dataclass(frozen=True)
class EmotionDecision:
    """情緒反應決策。"""

    emotion: str
    intensity: float
    defense_mechanism: str


@dataclass(frozen=True)
class PacingDecision:
    """對話節奏決策。"""

    pause_before_response: float
    speech_speed: float
    should_interrupt: bool
    response_length_hint: str  # "brief" | "moderate" | "detailed"


@dataclass(frozen=True)
class CrisisAssessment:
    """危機等級評估。"""

    level: int  # 0-5
    triggers: List[str] = field(default_factory=list)


def parse_decision_output(text: str) -> DecisionResult:
    """解析是否回應決策輸出，失敗時回傳安全預設（回應、中等緊急）。

    Args:
        text: 模型原始輸出。

    Returns:
        解析後的 :class:`DecisionResult`。
    """
    normalized = text.strip().lower()
    should_speak = not re.search(r"\b(no|否|不回應|沉默|silence)\b", normalized)
    urgency_match = re.search(r"(urgency|緊急)[^\d]*(\d(?:\.\d+)?)", normalized)
    try:
        urgency = float(urgency_match.group(2)) if urgency_match else 0.5
    except (ValueError, IndexError):
        urgency = 0.5
    urgency = max(0.0, min(1.0, urgency))
    emotion_tag = "neutral"
    for emotion in VALID_EMOTIONS:
        if emotion in normalized:
            emotion_tag = emotion
            break
    return DecisionResult(should_speak=should_speak, urgency=urgency, emotion_tag=emotion_tag)


def parse_emotion_output(text: str) -> EmotionDecision:
    """解析情緒決策輸出，失敗時回傳中性預設。

    Args:
        text: 模型原始輸出。

    Returns:
        解析後的 :class:`EmotionDecision`。
    """
    normalized = text.strip().lower()
    emotion = "neutral"
    for candidate in VALID_EMOTIONS:
        if candidate in normalized:
            emotion = candidate
            break
    intensity_match = re.search(r"(intensity|強度)[^\d]*(\d(?:\.\d+)?)", normalized)
    try:
        intensity = float(intensity_match.group(2)) if intensity_match else 0.5
    except (ValueError, IndexError):
        intensity = 0.5
    intensity = max(0.0, min(1.0, intensity))
    defense_match = re.search(r"(defense|防衛)[：:\s]*([a-z_]+)", normalized)
    defense = defense_match.group(2) if defense_match else "unknown"
    return EmotionDecision(emotion=emotion, intensity=intensity, defense_mechanism=defense)


def parse_pacing_output(text: str) -> PacingDecision:
    """解析節奏決策輸出，失敗時回傳溫和預設。

    Args:
        text: 模型原始輸出。

    Returns:
        解析後的 :class:`PacingDecision`。
    """
    normalized = text.strip().lower()
    pause_match = re.search(r"(pause|停頓)[^\d]*(\d(?:\.\d+)?)", normalized)
    try:
        pause = float(pause_match.group(2)) if pause_match else 1.0
    except (ValueError, IndexError):
        pause = 1.0
    pause = max(0.0, min(30.0, pause))
    speed_match = re.search(r"(speed|語速)[^\d]*(\d(?:\.\d+)?)", normalized)
    try:
        speed = float(speed_match.group(2)) if speed_match else 1.0
    except (ValueError, IndexError):
        speed = 1.0
    speed = max(0.5, min(2.0, speed))
    should_interrupt = bool(
        re.search(r"(interrupt|打斷|搶話)[^\n]*?(yes|true|是|true)", normalized)
    ) and not re.search(r"(no.?interrupt|不打斷|no|false)", normalized)
    length_hint = "moderate"
    for hint in VALID_LENGTH_HINTS:
        if hint in normalized:
            length_hint = hint
            break
    if "簡短" in text or "简短" in text:
        length_hint = "brief"
    elif "詳細" in text or "详细" in text:
        length_hint = "detailed"
    return PacingDecision(
        pause_before_response=pause,
        speech_speed=speed,
        should_interrupt=should_interrupt,
        response_length_hint=length_hint,
    )


def keyword_crisis_scan(text: str) -> CrisisAssessment:
    """關鍵字危機掃描（不依賴模型，保守原則）。

    Args:
        text: 待評估文本。

    Returns:
        掃描結果，等級為命中關鍵字的最高等級。
    """
    level = 0
    triggers: List[str] = []
    for candidate_level in sorted(_CRISIS_KEYWORDS, reverse=True):
        for keyword in _CRISIS_KEYWORDS[candidate_level]:
            if keyword.lower() in text.lower():
                if candidate_level > level:
                    level = candidate_level
                triggers.append(keyword)
    return CrisisAssessment(level=level, triggers=triggers)


def merge_crisis(
    model_level: int, model_triggers: List[str], source_text: str
) -> CrisisAssessment:
    """合併模型判斷與關鍵字掃描，取較高者（寧可高估）。

    Args:
        model_level: 模型給定的 0 至 5 等級。
        model_triggers: 模型回報的觸發詞。
        source_text: 原始對白文本（供關鍵字掃描）。

    Returns:
        保守合併後的 :class:`CrisisAssessment`。
    """
    try:
        model_level = max(0, min(5, int(model_level)))
    except (TypeError, ValueError):
        model_level = 0
    scanned = keyword_crisis_scan(source_text)
    triggers = sorted(set(model_triggers) | set(scanned.triggers))
    return CrisisAssessment(level=max(model_level, scanned.level), triggers=triggers)


def parse_crisis_output(text: str, source_text: str = "") -> CrisisAssessment:
    """合併模型輸出與關鍵字掃描，取較高者（寧可高估）。

    Args:
        text: 模型原始輸出。
        source_text: 原始對白文本，用於關鍵字掃描。

    Returns:
        保守合併後的 :class:`CrisisAssessment`。
    """
    level_match = re.search(r"(level|等級)[^\d]*([0-5])", text.strip().lower())
    try:
        model_level = int(level_match.group(2)) if level_match else 0
    except (ValueError, IndexError):
        model_level = 0
    trigger_match = re.findall(r"(trigger|觸發)[：:\s]*([^\n,，]+)", text)
    model_triggers = [t.strip() for _, t in trigger_match if t.strip()]
    return merge_crisis(model_level, model_triggers, source_text or text)


def crisis_from_noul(probability: float) -> int:
    """將自傷意念 noul 機率映射為 0 至 5 等級（啟發式）。

    Args:
        probability: noul 機率（已鉗制至 [0, 1]）。

    Returns:
        等級：>=0.8 為 5，>=0.5 為 4，>=0.3 為 3，否則為 0。
    """
    try:
        prob = max(0.0, min(1.0, float(probability)))
    except (TypeError, ValueError):
        return 0
    if prob >= 0.8:
        return 5
    if prob >= 0.5:
        return 4
    if prob >= 0.3:
        return 3
    return 0


class LayaEngine:
    """Laya 421M 決策引擎（官方套件優先，GGUF 備援）。"""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/laya-421m-bf16.gguf",
        n_ctx: int = 512,
        n_gpu_layers: Union[str, int, None] = "auto",
        backend: str = "auto",
        checkpoint: str = "multilingual",
    ) -> None:
        """初始化引擎參數（實際載入請呼叫 :meth:`load`）。

        Args:
            model_path: GGUF 權重路徑（gguf 後端用，支援規格別名）。
            n_ctx: 上下文長度（gguf 後端用）。
            n_gpu_layers: GPU offload 層數設定（gguf 後端用）。
            backend: ``auto``（官方套件優先，缺失時轉 GGUF）、
                ``laya``（僅官方套件）、``gguf``（僅 llama-cpp）。
            checkpoint: 官方套件檢查點，本專案中文為主請用 multilingual。

        Raises:
            ValueError: 參數非法時。
        """
        if n_ctx <= 0:
            raise ValueError("n_ctx 必須為正整數")
        if backend not in {"auto", "laya", "gguf"}:
            raise ValueError("backend 必須為 auto、laya 或 gguf")
        self.model_path = resolve_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers_setting = n_gpu_layers
        self.backend = backend
        self.checkpoint = checkpoint
        self._router: Optional[object] = None
        self._model: Optional[object] = None

    @property
    def is_loaded(self) -> bool:
        """任一後端就緒即為已載入。"""
        return self._router is not None or self._model is not None

    @property
    def active_backend(self) -> str:
        """回傳作用中後端（laya、gguf 或空字串）。"""
        if self._router is not None:
            return "laya"
        if self._model is not None:
            return "gguf"
        return ""

    def load(self) -> "LayaEngine":
        """載入模型（冪等；auto 模式官方套件優先）。

        Returns:
            自身，支援鏈式呼叫。

        Raises:
            ImportError: 指定 laya 後端但套件缺失時。
            FileNotFoundError: 走 GGUF 路徑但權重缺失時。
            RuntimeError: 載入失敗時（含 BF16 指引）。
        """
        if self.is_loaded:
            return self
        if self.backend in {"auto", "laya"}:
            try:
                from laya import Router

                logger.info(f"載入 Laya 官方套件（檢查點={self.checkpoint}）")
                self._router = Router()
                logger.info("Laya 官方套件就緒（檢查點延遲載入，首次決策約需數秒）")
                return self
            except ImportError as exc:
                if self.backend == "laya":
                    raise ImportError(_LAYA_PKG_HINT) from exc
                logger.warning(f"官方 laya 套件缺失，轉 GGUF 路徑：{exc}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到 Laya 模型檔：{self.model_path}，"
                "請依 models/README.md 取得檔案"
            )
        from src.perception.embedding_engine import _resolve_gpu_layers

        layers = _resolve_gpu_layers(self.n_gpu_layers_setting)
        logger.info(f"載入 Laya 模型：{self.model_path}（GPU 層數={layers}）")
        try:
            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=layers,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Laya 模型載入失敗：{exc}。"
                "BF16 格式在舊版 llama.cpp 或舊硬體上可能不支援，"
                "請升級 llama-cpp-python 至最新版，或改用 FP16 量化版本"
            ) from exc
        logger.info("Laya 模型載入完成")
        return self

    def unload(self) -> None:
        """卸載模型並釋放記憶體（冪等）。"""
        if self._router is not None:
            try:
                unload = getattr(self._router, "unload", None)
                if callable(unload):
                    unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Laya 官方套件卸載時發生非致命錯誤：{exc}")
            finally:
                self._router = None
        if self._model is not None:
            try:
                close = getattr(self._model, "close", None)
                if callable(close):
                    close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"模型卸載時發生非致命錯誤：{exc}")
            finally:
                self._model = None
        logger.info("Laya 模型已卸載")

    def __enter__(self) -> "LayaEngine":
        """支援 with 語句自動載入。"""
        return self.load()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """離開 with 區塊時自動卸載。"""
        self.unload()

    def _predict(self, state: str, questions: Dict[str, Dict[str, object]]) -> Dict[str, Any]:
        """經官方套件執行 typed 決策，失敗回傳空字典。

        Args:
            state: 決策情境文本。
            questions: typed 問題定義。

        Returns:
            answers 字典（失敗時為空，由呼叫方退回預設）。
        """
        if self._router is None:
            raise RuntimeError("模型尚未載入，請先呼叫 load()")
        try:
            result = self._router.predict(state, questions, model=self.checkpoint)  # type: ignore[union-attr]
            answers = result.get("answers", {}) if isinstance(result, dict) else {}
            return answers if isinstance(answers, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Laya 官方套件推理失敗，使用安全預設：{exc}")
            return {}

    @staticmethod
    def _choice(answers: Dict[str, Any], name: str, default: str = "") -> str:
        """安全提取 choice 答案。"""
        try:
            value = answers.get(name, {}).get("choice", default)
            return str(value) if value else default
        except (AttributeError, TypeError):
            return default

    @staticmethod
    def _score(answers: Dict[str, Any], name: str, default: float = 0.0) -> float:
        """安全提取 score 答案。"""
        try:
            return float(answers.get(name, {}).get("score", default))
        except (AttributeError, TypeError, ValueError):
            return default

    @staticmethod
    def _noul(answers: Dict[str, Any], name: str, default: float = 0.0) -> float:
        """安全提取 noul 機率。"""
        try:
            return max(0.0, min(1.0, float(answers.get(name, {}).get("noul", default))))
        except (AttributeError, TypeError, ValueError):
            return default

    @staticmethod
    def _confidence(answers: Dict[str, Any], name: str, default: float = 0.5) -> float:
        """安全提取 answer_confidence。"""
        try:
            return max(0.0, min(1.0, float(answers.get(name, {}).get("answer_confidence", default))))
        except (AttributeError, TypeError, ValueError):
            return default

    def _complete(self, prompt: str, max_tokens: int = 64) -> str:
        """執行 greedy completion 並回傳文本（GGUF 後端）。

        Args:
            prompt: 決策提示。
            max_tokens: 最大生成 token 數。

        Returns:
            模型輸出文本（失敗時回傳空字串，由呼叫方退回預設）。
        """
        if self._model is None:
            raise RuntimeError("模型尚未載入，請先呼叫 load()")
        try:
            result = self._model.create_completion(  # type: ignore[union-attr]
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                stop=["\n\n"],
            )
            return str(result["choices"][0]["text"])
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Laya 推理失敗，使用安全預設：{exc}")
            return ""

    def should_respond(self, context: str) -> DecisionResult:
        """判斷當前情境是否需要虛擬個案回應。

        Args:
            context: 當前對話上下文摘要。

        Returns:
            :class:`DecisionResult`，推理失敗時退回安全預設。
        """
        if not context or not context.strip():
            raise ValueError("context 不可為空")
        if self._router is not None:
            answers = self._predict(
                context.strip()[:800],
                {
                    "respond": {
                        "type": "choice",
                        "instructions": "面對諮商師的這句話，個案會開口回應，還是明顯拒絕或無法回應？",
                        "criteria": {"speak": "個案會開口回應", "silent": "個案明顯拒絕或無法回應，如解離、強烈抗拒"},
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "回應的緊急程度？",
                        "criteria": ["不緊急", "中等", "緊急"],
                    },
                    "emotion": {
                        "type": "choice",
                        "instructions": "個案的主要情緒是？",
                        "criteria": _EMOTION_CRITERIA,
                    },
                },
            )
            if not answers:
                return DecisionResult(True, 0.5, "neutral")
            choice = self._choice(answers, "respond", "speak")
            urgency = max(0.0, min(1.0, self._score(answers, "urgency", 1.0) / 2.0))
            emotion = self._choice(answers, "emotion", "neutral")
            if emotion not in VALID_EMOTIONS:
                emotion = "neutral"
            return DecisionResult(choice != "silent", urgency, emotion)
        prompt = (
            "Decide if the client should speak now. "
            "Reply with YES or NO, urgency 0-1, and one emotion tag.\n"
            f"Context: {context.strip()[:400]}\nAnswer:"
        )
        return parse_decision_output(self._complete(prompt))

    def determine_emotion(self, context: str, client_state: str) -> EmotionDecision:
        """判斷個案的情緒反應。

        Args:
            context: 對話上下文。
            client_state: 個案當前狀態描述。

        Returns:
            :class:`EmotionDecision`，推理失敗時退回中性預設。
        """
        if not context or not context.strip():
            raise ValueError("context 不可為空")
        if self._router is not None:
            answers = self._predict(
                f"{context.strip()[:600]}\n狀態：{client_state.strip()[:200]}",
                {
                    "emotion": {
                        "type": "choice",
                        "instructions": "個案的主要情緒是？",
                        "criteria": _EMOTION_CRITERIA,
                    },
                    "defense": {
                        "type": "choice",
                        "instructions": "個案的主要防衛機制是？",
                        "criteria": _DEFENSE_CRITERIA,
                    },
                },
            )
            if not answers:
                return EmotionDecision("neutral", 0.5, "unknown")
            emotion = self._choice(answers, "emotion", "neutral")
            if emotion not in VALID_EMOTIONS:
                emotion = "neutral"
            defense = self._choice(answers, "defense", "unknown")
            if defense not in _DEFENSE_CRITERIA:
                defense = "unknown"
            return EmotionDecision(
                emotion, self._confidence(answers, "emotion", 0.5), defense
            )
        prompt = (
            "Classify the client's emotion. Choose one: "
            + ", ".join(VALID_EMOTIONS)
            + ". Reply with emotion, intensity 0-1, defense mechanism.\n"
            f"Context: {context.strip()[:400]}\nState: {client_state.strip()[:200]}\nAnswer:"
        )
        return parse_emotion_output(self._complete(prompt))

    def determine_pacing(self, context: str, conversation_turn: int) -> PacingDecision:
        """決定對話節奏。

        Args:
            context: 對話上下文。
            conversation_turn: 當前輪次（必須為正整數）。

        Returns:
            :class:`PacingDecision`，推理失敗時退回溫和預設。
        """
        if not context or not context.strip():
            raise ValueError("context 不可為空")
        if conversation_turn <= 0:
            raise ValueError("conversation_turn 必須為正整數")
        if self._router is not None:
            answers = self._predict(
                f"第 {conversation_turn} 輪。{context.strip()[:600]}",
                {
                    "pause": {
                        "type": "score",
                        "instructions": "回應前需要停頓多久？",
                        "criteria": ["立即回應", "短暫停頓", "中等停頓", "長時間沉默"],
                    },
                    "speed": {
                        "type": "choice",
                        "instructions": "回應語速？",
                        "criteria": {"slower": "放慢語速", "normal": "正常語速", "faster": "加快語速"},
                    },
                    "interrupt": {
                        "type": "noul",
                        "instructions": "是否應打斷諮商師直接回應？",
                    },
                    "length": {
                        "type": "choice",
                        "instructions": "回應長度？",
                        "criteria": {"brief": "簡短回應", "moderate": "適中回應", "detailed": "詳細回應"},
                    },
                },
            )
            if not answers:
                return PacingDecision(1.0, 1.0, False, "moderate")
            pause = max(0.0, min(6.0, self._score(answers, "pause", 0.5) * 2.0))
            speed_key = self._choice(answers, "speed", "normal")
            speed = {"slower": 0.75, "normal": 1.0, "faster": 1.3}.get(speed_key, 1.0)
            length = self._choice(answers, "length", "moderate")
            if length not in VALID_LENGTH_HINTS:
                length = "moderate"
            return PacingDecision(
                pause, speed, self._noul(answers, "interrupt", 0.0) >= 0.5, length
            )
        prompt = (
            "Decide pacing: pause seconds 0-30, speed 0.5-2.0, "
            "interrupt YES/NO, length brief/moderate/detailed.\n"
            f"Turn {conversation_turn}. Context: {context.strip()[:400]}\nAnswer:"
        )
        return parse_pacing_output(self._complete(prompt, max_tokens=96))

    def assess_crisis_level(self, response_text: str) -> CrisisAssessment:
        """評估危機等級 0-5（模型判斷 union 關鍵字掃描）。

        Args:
            response_text: 個案回應文本。

        Returns:
            :class:`CrisisAssessment`，採保守合併，寧可高估。
        """
        if not response_text or not response_text.strip():
            raise ValueError("response_text 不可為空")
        if self._router is not None:
            answers = self._predict(
                response_text.strip()[:600],
                {
                    "crisis": {
                        "type": "noul",
                        "instructions": "個案是否表達自傷或自殺意念？",
                    },
                },
            )
            if not answers:
                scanned = keyword_crisis_scan(response_text)
                return CrisisAssessment(scanned.level, scanned.triggers)
            return merge_crisis(
                crisis_from_noul(self._noul(answers, "crisis", 0.0)),
                [],
                response_text,
            )
        prompt = (
            "Rate crisis level 0-5 (0 safe, 5 imminent suicide risk). "
            "Reply with level and triggers.\n"
            f"Response: {response_text.strip()[:500]}\nAnswer:"
        )
        return parse_crisis_output(self._complete(prompt), source_text=response_text)
