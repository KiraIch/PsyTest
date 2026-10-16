"""Ollama API 客戶端（骨架，Phase 2 實作）。

預定行為：
    1. 偵測 Ollama 是否運行（GET /api/tags），未運行則嘗試跨平台啟動。
    2. 模糊匹配 Gemma 4 Uncensored 模型（含 gemma、uncensored、
       hauhau、aggressive 關鍵字），支援 e2b/e4b 變體。
    3. 數據生成用非串流，即時互動用串流；失敗重試最多 3 次。
    4. 支援環境變數 OLLAMA_HOST 覆寫。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_HOST = "http://localhost:11434"
MODEL_KEYWORDS = ("gemma", "uncensored", "hauhau", "aggressive")


@dataclass
class GenerateOptions:
    """生成參數。"""

    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = 512
    repeat_penalty: float = 1.1
    stream: bool = False


class OllamaClient:
    """Ollama API 客戶端（Phase 2 實作）。"""

    def __init__(self, host: Optional[str] = None) -> None:
        """初始化客戶端。

        Args:
            host: Ollama 位址，未指定時讀取 OLLAMA_HOST，預設本機。
        """
        import os

        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")

    def ensure_running(self, timeout: int = 5) -> bool:
        """確保 Ollama 運行中（Phase 2 實作）。"""
        raise NotImplementedError("OllamaClient.ensure_running 排定於 Phase 2 實作")

    def list_models(self) -> List[str]:
        """列出已安裝模型（Phase 2 實作）。"""
        raise NotImplementedError("OllamaClient.list_models 排定於 Phase 2 實作")

    def detect_model(self, preference: str = "auto") -> str:
        """模糊匹配目標模型（Phase 2 實作）。"""
        raise NotImplementedError("OllamaClient.detect_model 排定於 Phase 2 實作")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GenerateOptions] = None,
    ) -> Dict[str, Any]:
        """生成回應（Phase 2 實作）。"""
        raise NotImplementedError("OllamaClient.generate 排定於 Phase 2 實作")
