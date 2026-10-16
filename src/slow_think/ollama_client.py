"""Ollama API 客戶端（Phase 2 實作）。

負責與本機 Ollama 服務通訊，預設使用 e4b 變體
``fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest``
（6.3GB，本機已驗證存在）。模型偵測採模糊匹配，優先順序為
e4b 明確標籤、本地已存在的相容模型、e2b 標籤、其他 4B 以上模型。

行為要點：
    1. 啟動時偵測 Ollama 是否運行，未運行則嘗試跨平台自動啟動。
    2. 數據生成使用非串流（效率優先），即時互動可使用串流。
    3. 失敗自動重試最多 3 次，記憶體不足時提供降載指引。
    4. 位址支援環境變數 OLLAMA_HOST 覆寫。

使用範例：
    from src.slow_think.ollama_client import OllamaClient, GenerateOptions

    client = OllamaClient()
    client.ensure_running()
    model = client.detect_model()
    result = client.generate("個案說：我最近睡不好", model=model)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 requests，請先執行 pip install -r requirements.txt"
    ) from exc

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HOST = "http://localhost:11434"

# 本專案鎖定的 e4b 變體（本機已驗證存在，6.3GB）。
PINNED_MODEL = "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest"

MODEL_KEYWORDS = ("gemma", "uncensored", "hauhau", "aggressive")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


@dataclass
class GenerateOptions:
    """生成參數。

    Attributes:
        think: 是否啟用思考模式。數據生成預設關閉（效率優先），
            因為思考 token 會佔用 num_predict 配額，小配額下可能
            導致回應為空；即時互動可視需要開啟。
    """

    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = 512
    repeat_penalty: float = 1.1
    stream: bool = False
    think: bool = False


def _platform() -> str:
    """回傳正規化平台名稱。"""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def _try_start_ollama() -> None:
    """嘗試以跨平台方式啟動 Ollama（失敗不拋出例外，由呼叫方重試確認）。"""
    current = _platform()
    try:
        if current == "macos":
            subprocess.run(["open", "-a", "Ollama"], check=False)
        elif current == "linux":
            subprocess.run(
                ["ollama", "serve"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif current == "windows":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"自動啟動 Ollama 失敗：{exc}")


class OllamaClient:
    """Ollama API 客戶端。"""

    def __init__(self, host: Optional[str] = None, timeout: int = 120) -> None:
        """初始化客戶端。

        Args:
            host: Ollama 位址，未指定時讀取 OLLAMA_HOST，預設本機。
            timeout: 單次請求逾時（秒）。

        Raises:
            ValueError: timeout 非正數時。
        """
        if timeout <= 0:
            raise ValueError("timeout 必須為正數")
        self.host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.timeout = timeout

    def _tags_url(self) -> str:
        """回傳模型列表端點。"""
        return f"{self.host}/api/tags"

    def is_running(self) -> bool:
        """檢查 Ollama 是否運行中（不嘗試啟動）。"""
        try:
            response = requests.get(self._tags_url(), timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def ensure_running(self, wait_seconds: int = 10) -> bool:
        """確保 Ollama 運行中，未運行則嘗試自動啟動並等待。

        Args:
            wait_seconds: 啟動後等待秒數。

        Returns:
            最終是否可用。

        Raises:
            RuntimeError: 仍無法連線時，提供各平台安裝指引。
        """
        if self.is_running():
            return True
        logger.info("Ollama 未運行，嘗試自動啟動")
        _try_start_ollama()
        time.sleep(wait_seconds)
        if self.is_running():
            logger.info("Ollama 已啟動")
            return True
        raise RuntimeError(
            "無法連接 Ollama。請確認已安裝並啟動："
            "macOS 請開啟 Ollama 應用程式；"
            "Linux 請執行 ollama serve；"
            "Windows 請啟動 Ollama；"
            "或以 OLLAMA_HOST 環境變數指定正確位址"
        )

    def list_models(self) -> List[str]:
        """列出已安裝模型名稱。

        Returns:
            模型名稱清單。

        Raises:
            RuntimeError: 連線失敗或回應格式異常時。
        """
        try:
            response = requests.get(self._tags_url(), timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"無法取得 Ollama 模型列表（{self.host}）：{exc}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(f"Ollama 回應非合法 JSON：{exc}") from exc
        models = data.get("models", [])
        if not isinstance(models, list):
            raise RuntimeError("Ollama 回應格式異常：缺少 models 清單")
        names = [m.get("name", "") for m in models if isinstance(m, dict)]
        return [name for name in names if name]

    def detect_model(self, preference: str = "auto") -> str:
        """偵測要使用的模型，優先 e4b 變體。

        Args:
            preference: ``auto`` 表自動偵測，否則直接回傳指定名稱
                （仍會驗證其存在，不存在時拋出例外）。

        Returns:
            模型完整名稱。

        Raises:
            RuntimeError: 無可用模型，或指定模型不存在時。
        """
        available = self.list_models()
        if not available:
            raise RuntimeError(
                "Ollama 尚未安裝任何模型，請執行："
                "ollama pull fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive"
            )
        if preference != "auto":
            if preference in available:
                return preference
            raise RuntimeError(
                f"指定模型不存在：{preference}。可用模型：{available}"
            )
        # 第一優先：本專案鎖定的 e4b 變體。
        if PINNED_MODEL in available:
            return PINNED_MODEL

        def score(name: str) -> tuple[int, str]:
            lowered = name.lower()
            keywords = sum(1 for kw in MODEL_KEYWORDS if kw in lowered)
            if "e4b" in lowered and keywords >= 2:
                rank = 0
            elif keywords >= 3:
                rank = 1
            elif "e2b" in lowered and keywords >= 2:
                rank = 2
            elif keywords >= 1:
                rank = 3
            else:
                rank = 4
            return (rank, name)

        ranked = sorted((score(name) for name in available))
        best_rank, best_name = ranked[0]
        if best_rank >= 4:
            logger.warning(
                f"找不到 Gemma 4 Uncensored 相容模型，回退使用 {best_name}。"
                "建議執行 ollama pull "
                "fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive"
            )
        else:
            logger.info(f"自動選用模型：{best_name}")
        return best_name

    def _build_payload(
        self,
        prompt: str,
        model: str,
        options: GenerateOptions,
    ) -> Dict[str, Any]:
        """組裝 /api/generate 請求內容。"""
        return {
            "model": model,
            "prompt": prompt,
            "stream": options.stream,
            "think": options.think,
            "options": {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "num_predict": options.max_tokens,
                "repeat_penalty": options.repeat_penalty,
            },
        }

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        """由回應提取文本，格式異常時拋出例外。"""
        if not isinstance(data, dict) or "response" not in data:
            raise RuntimeError(f"Ollama 回應格式異常：{str(data)[:200]}")
        return str(data["response"])

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GenerateOptions] = None,
    ) -> Dict[str, Any]:
        """非串流生成，失敗自動重試最多 3 次。

        Args:
            prompt: 輸入提示，不可為空。
            model: 模型名稱，未指定時自動偵測。
            options: 生成參數，未指定時使用預設。

        Returns:
            包含 ``text`` 與 ``model`` 的字典。

        Raises:
            ValueError: prompt 為空時。
            RuntimeError: 重試耗盡仍失敗時。
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt 不可為空")
        opts = options or GenerateOptions(stream=False)
        opts.stream = False
        target = model or self.detect_model()
        payload = self._build_payload(prompt.strip(), target, opts)
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                text = self._extract_text(response.json())
                return {"text": text, "model": target}
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                logger.warning(f"生成失敗（第 {attempt}/{MAX_RETRIES} 次）：{exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        hint = ""
        if last_error is not None and "memory" in str(last_error).lower():
            hint = "疑似記憶體不足，請降低 max_tokens 或改用更小的模型變體（e2b）。"
        raise RuntimeError(
            f"Ollama 生成失敗，已重試 {MAX_RETRIES} 次：{last_error}。{hint}"
        ) from last_error

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GenerateOptions] = None,
    ) -> Iterator[str]:
        """串流生成，逐段產出文本（即時互動用）。

        Args:
            prompt: 輸入提示，不可為空。
            model: 模型名稱，未指定時自動偵測。
            options: 生成參數。

        Yields:
            文本片段。

        Raises:
            ValueError: prompt 為空時。
            RuntimeError: 連線失敗時。
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt 不可為空")
        opts = options or GenerateOptions()
        opts.stream = True
        target = model or self.detect_model()
        payload = self._build_payload(prompt.strip(), target, opts)
        try:
            with requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    import json as _json

                    try:
                        chunk = _json.loads(line.decode("utf-8"))
                    except ValueError:
                        continue
                    if isinstance(chunk, dict) and "response" in chunk:
                        yield str(chunk["response"])
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama 串流生成失敗：{exc}") from exc
