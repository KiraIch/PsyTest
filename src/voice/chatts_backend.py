"""ChatTTS 後端（Phase 4 實作，已於本機驗證）。

ChatTTS 原生理解 ``[laugh]``、``[uv_break]`` 等標籤，
情感表達能力最佳，但 CPU 推理較慢且授權為 AGPLv3+。
本模組僅在使用者安裝後動態調用，未捆綁散佈，
授權說明見 NOTICE。首次載入會由 HF 下載模型（數 GB）。

固定音色：初始化時以固定種子抽取說話人嵌入並重複使用，
確保同一批數據的音色一致。

驗證紀錄：中文情感合成成功（3.75 秒，24kHz，峰值 0.995）。

使用範例：
    from src.voice.chatts_backend import ChatTTSBackend

    engine = ChatTTSBackend()
    output = engine.synthesize("你好[laugh]，最近好嗎？", {"oral": 2})
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.utils.logger import get_logger
from src.voice.tts_engine import AudioOutput, TTSEngine, parse_inline_tags, strip_stage_directions

logger = get_logger(__name__)

# <tag> 到 ChatTTS 原生寫法的轉換（無對應者移除）。
_ANGLE_TO_BRACKET = {
    "hesitate": "[uv_break]",
    "breath": "[uv_break]",
    "cry": "[laugh]",
}


class ChatTTSBackend(TTSEngine):
    """ChatTTS 實作。"""

    name = "chattts"

    def __init__(
        self,
        model_source: str = "huggingface",
        oral: int = 2,
        laugh: int = 0,
        breakthrough: int = 0,
        sample_rate: int = 24000,
    ) -> None:
        """初始化後端參數（模型延至首次合成時載入）。

        Args:
            model_source: ``huggingface``（自動下載）或本地模型目錄。
            oral: 口語化程度預設。
            laugh: 笑聲強度預設。
            breakthrough: 停頓強度預設（對應 break，避開關鍵字）。
            sample_rate: 目標取樣率（ChatTTS 原生 24000）。

        Raises:
            ValueError: 參數非法時。
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate 必須為正整數")
        self.model_source = model_source
        self.oral = oral
        self.laugh = laugh
        self.breakthrough = breakthrough
        self.sample_rate = sample_rate
        self._chat: object | None = None
        self._speaker: object | None = None

    def unavailable_reason(self) -> str:
        """回傳不可用原因。"""
        try:
            import ChatTTS  # noqa: F401
        except ImportError:
            return "缺少 ChatTTS，請執行 pip install ChatTTS（授權 AGPLv3+，見 NOTICE）"
        return ""

    def is_available(self) -> bool:
        """檢查 ChatTTS 是否已安裝（模型於載入時取得）。"""
        return self.unavailable_reason() == ""

    def get_voice_options(self) -> List[str]:
        """回傳固定音色說明（說話人嵌入於初始化時抽取）。"""
        return ["default-fixed-speaker"]

    def _require_chat(self) -> object:
        """載入（或回傳已快取的）ChatTTS 實例與固定音色。"""
        if self._chat is not None:
            return self._chat
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(f"ChatTTS 不可用：{reason}")
        import ChatTTS
        import torch

        logger.info("載入 ChatTTS 模型（首次可能需由 HF 下載數 GB）")
        try:
            chat = ChatTTS.Chat()
            if not chat.load(source=self.model_source):  # type: ignore[arg-type]
                raise RuntimeError("ChatTTS 模型載入回傳失敗")
        except Exception as exc:
            raise RuntimeError(f"ChatTTS 模型載入失敗：{exc}") from exc
        torch.manual_seed(42)
        try:
            self._speaker = chat.sample_random_speaker()
        except Exception as exc:
            raise RuntimeError(f"ChatTTS 音色抽取失敗：{exc}") from exc
        self._chat = chat
        logger.info("ChatTTS 模型載入完成")
        return self._chat

    def _adapt_text(self, text: str) -> str:
        """轉換 <tag> 為 ChatTTS 原生標籤並移除無對應者與舞台指示。"""
        adapted = strip_stage_directions(text)
        for angle, bracket in _ANGLE_TO_BRACKET.items():
            adapted = adapted.replace(f"<{angle}>", f" {bracket} ")
            adapted = adapted.replace(f"<{angle.upper()}>", f" {bracket} ")
        adapted, _ = parse_inline_tags(adapted)
        return " ".join(adapted.split())

    def synthesize(self, text: str, director_notes: Dict[str, object]) -> AudioOutput:
        """合成情感語音。

        Args:
            text: 輸入文本（可含兩種標籤寫法）。
            director_notes: 可含 oral、laugh、break 覆寫。

        Returns:
            :class:`AudioOutput`（float32 單聲道，24kHz）。

        Raises:
            ValueError: 文本為空或清理後為空時。
            RuntimeError: 引擎不可用或合成失敗時。
        """
        if not text or not text.strip():
            raise ValueError("輸入文本不可為空")
        chat = self._require_chat()
        adapted = self._adapt_text(text.strip())
        if not adapted:
            raise ValueError("文本經清理後為空，無法合成")
        try:
            oral = int(director_notes.get("oral", self.oral))  # type: ignore[arg-type]
            laugh = int(director_notes.get("laugh", self.laugh))  # type: ignore[arg-type]
            brk = int(director_notes.get("break", director_notes.get("breakthrough", self.breakthrough)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            oral, laugh, brk = self.oral, self.laugh, self.breakthrough
        oral = max(0, min(9, oral))
        laugh = max(0, min(9, laugh))
        brk = max(0, min(7, brk))
        # ChatTTS 0.2.x 以行內特殊標籤控制情感（非 infer 參數）。
        tagged = f"[oral_{oral}][laugh_{laugh}][break_{brk}]{adapted}"
        params = {"oral": oral, "laugh": laugh, "break": brk}
        try:
            import ChatTTS as _ChatTTS

            infer_code = _ChatTTS.Chat.InferCodeParams(spk_emb=self._speaker)
            wavs = chat.infer([tagged], params_infer_code=infer_code)  # type: ignore[union-attr]
            waveform = np.asarray(wavs[0]).squeeze().astype(np.float32).ravel()
        except Exception as exc:
            raise RuntimeError(f"ChatTTS 合成失敗：{exc}") from exc
        if waveform.size == 0:
            raise RuntimeError("ChatTTS 回傳空音訊")
        peak = float(np.abs(waveform).max())
        if peak > 1.0:
            waveform = waveform / peak
        return AudioOutput(
            audio_array=waveform,
            sample_rate=24000,
            duration=float(len(waveform) / 24000),
            metadata={
                "engine": "chattts",
                "params": params,
                "peak": round(peak, 3),
            },
        )

    def unload(self) -> None:
        """釋放模型與音色（冪等，供管線記憶體管理使用）。"""
        self._chat = None
        self._speaker = None
        logger.info("ChatTTS 模型已卸載")
