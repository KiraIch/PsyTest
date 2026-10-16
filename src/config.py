"""PsyTest 全域設定管理。

負責載入 YAML 設定檔、合併環境變數覆寫、解析模型路徑別名，
並對設定內容進行驗證。所有設定讀取應經由此模組，避免各處自行解析。

模型檔名別名：
    規格書使用 ``embeddinggemma2-q8.gguf`` 與 ``laya-421m-bf16.gguf``，
    本機實際檔名為 ``embeddinggemma-2-Q8_0.gguf`` 與 ``Laya-BF16.gguf``。
    :func:`resolve_model_path` 會依序嘗試別名，回傳第一個存在的檔案。

環境變數覆寫（見 ``.env.example``）：
    OLLAMA_HOST、OLLAMA_MODEL、PSYTEST_OUTPUT_DIR、PSYTEST_SAVE_AUDIO、
    PSYTEST_SAVE_EMBEDDINGS、PSYTEST_MAX_MEMORY_GB、PSYTEST_GPU_LAYERS、
    PSYTEST_TEMPERATURE_LIMIT、PSYTEST_LOG_LEVEL
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 PyYAML，請先執行 pip install -r requirements.txt"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型檔名別名表：規格名稱 -> 本機實際可能檔名（依序嘗試）。
MODEL_ALIASES: dict[str, list[str]] = {
    "models/embeddinggemma2-q8.gguf": [
        "models/embeddinggemma2-q8.gguf",
        "models/embeddinggemma-2-Q8_0.gguf",
    ],
    "models/laya-421m-bf16.gguf": [
        "models/laya-421m-bf16.gguf",
        "models/Laya-BF16.gguf",
    ],
}


def get_platform() -> str:
    """回傳正規化平台名稱：``macos``、``windows`` 或 ``linux``。"""
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def get_project_root() -> Path:
    """回傳專案根目錄。"""
    return PROJECT_ROOT


def resolve_model_path(relative_path: str | Path) -> Path:
    """解析模型路徑，支援規格檔名與本機實際檔名別名。

    Args:
        relative_path: 相對於專案根目錄的路徑。

    Returns:
        解析後的絕對路徑（第一個存在的候選；若皆不存在則回傳首選）。

    Raises:
        ValueError: 路徑為空時。
    """
    if not str(relative_path).strip():
        raise ValueError("模型路徑不可為空")
    key = str(relative_path).replace("\\", "/")
    candidates = MODEL_ALIASES.get(key, [key])
    for candidate in candidates:
        full = PROJECT_ROOT / candidate
        if full.exists():
            return full
    return PROJECT_ROOT / candidates[0]


def find_existing_model_path(relative_path: str | Path) -> Optional[Path]:
    """回傳存在的模型路徑，不存在時回傳 None（不拋出例外）。"""
    try:
        resolved = resolve_model_path(relative_path)
    except ValueError:
        return None
    return resolved if resolved.exists() else None


# ---------------------------------------------------------------------------
# 設定資料結構
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """計算後端設定。"""

    auto_detect: bool = True
    force: Optional[str] = None
    gpu_layers: str | int = "auto"
    max_memory_gb: Optional[float] = None
    temperature_limit: float = 85.0


@dataclass
class EmbeddingModelConfig:
    """EmbeddingGemma 2 模型設定。"""

    path: str = "models/embeddinggemma2-q8.gguf"
    context_length: int = 2048
    batch_size: int = 32
    gpu_layers: str | int = "auto"


@dataclass
class LayaModelConfig:
    """Laya 421M 決策模型設定。"""

    path: str = "models/laya-421m-bf16.gguf"
    context_length: int = 512
    temperature: float = 0.0
    gpu_layers: str | int = "auto"


@dataclass
class LLMConfig:
    """慢想層 Ollama LLM 設定。"""

    provider: str = "ollama"
    model_name: str = "auto"
    host: str = "http://localhost:11434"
    temperature: float = 0.9
    top_p: float = 0.9
    max_tokens: int = 512
    repeat_penalty: float = 1.1


@dataclass
class TTSConfig:
    """語音合成設定。"""

    engine: str = "auto"
    sample_rate: int = 24000
    voice: str = "default"


@dataclass
class ModelsConfig:
    """全部模型設定。"""

    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    laya: LayaModelConfig = field(default_factory=LayaModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


@dataclass
class PipelineConfig:
    """數據生成管線設定。"""

    max_turns_per_session: int = 30
    max_duration_minutes: int = 50
    save_audio: bool = True
    save_embeddings: bool = True
    embedding_save_format: str = "float32"


@dataclass
class ThermalConfig:
    """溫度監控設定。"""

    enabled: bool = True
    check_interval_seconds: int = 30
    throttle_threshold: float = 85.0
    cooldown_seconds: int = 120


@dataclass
class LoggingConfig:
    """日誌設定。"""

    level: str = "INFO"
    file: str = "logs/psytest.log"
    console: bool = True


@dataclass
class AppConfig:
    """應用程式完整設定。"""

    project_name: str = "PsyTest"
    project_version: str = "1.0.3"
    output_dir: str = "data/output"
    backend: BackendConfig = field(default_factory=BackendConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: Optional[str] = None


# ---------------------------------------------------------------------------
# 載入與合併
# ---------------------------------------------------------------------------


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"環境變數 {name} 必須為數字，實際值：{raw!r}") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    """讀取 YAML 檔案並回傳字典。

    Raises:
        FileNotFoundError: 檔案不存在時。
        ValueError: 內容不是合法映射時。
    """
    full = Path(path)
    if not full.is_absolute():
        full = PROJECT_ROOT / full
    if not full.exists():
        raise FileNotFoundError(f"設定檔不存在：{full}")
    try:
        with open(full, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"設定檔 YAML 解析失敗：{full}：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"設定檔頂層必須為映射：{full}")
    return data


def _build_config(data: dict[str, Any]) -> AppConfig:
    """由字典建構 AppConfig（缺漏鍵值以預設補齊）。"""
    project = data.get("project", {}) or {}
    backend = data.get("backend", {}) or {}
    models = data.get("models", {}) or {}
    embedding = models.get("embedding", {}) or {}
    laya = models.get("laya", {}) or {}
    llm = models.get("llm", {}) or {}
    tts = models.get("tts", {}) or {}
    pipeline = data.get("pipeline", {}) or {}
    thermal = data.get("thermal", {}) or {}
    logging_cfg = data.get("logging", {}) or {}

    return AppConfig(
        project_name=str(project.get("name", "PsyTest")),
        project_version=str(project.get("version", "1.0.3")),
        output_dir=str(project.get("output_dir", "data/output")),
        backend=BackendConfig(
            auto_detect=bool(backend.get("auto_detect", True)),
            force=backend.get("force"),
            gpu_layers=backend.get("gpu_layers", "auto"),
            max_memory_gb=backend.get("max_memory_gb"),
            temperature_limit=float(backend.get("temperature_limit", 85)),
        ),
        models=ModelsConfig(
            embedding=EmbeddingModelConfig(
                path=str(embedding.get("path", "models/embeddinggemma2-q8.gguf")),
                context_length=int(embedding.get("context_length", 2048)),
                batch_size=int(embedding.get("batch_size", 32)),
                gpu_layers=embedding.get("gpu_layers", "auto"),
            ),
            laya=LayaModelConfig(
                path=str(laya.get("path", "models/laya-421m-bf16.gguf")),
                context_length=int(laya.get("context_length", 512)),
                temperature=float(laya.get("temperature", 0.0)),
                gpu_layers=laya.get("gpu_layers", "auto"),
            ),
            llm=LLMConfig(
                provider=str(llm.get("provider", "ollama")),
                model_name=str(llm.get("model_name", "auto")),
                host=str(llm.get("host", "http://localhost:11434")),
                temperature=float(llm.get("temperature", 0.9)),
                top_p=float(llm.get("top_p", 0.9)),
                max_tokens=int(llm.get("max_tokens", 512)),
                repeat_penalty=float(llm.get("repeat_penalty", 1.1)),
            ),
            tts=TTSConfig(
                engine=str(tts.get("engine", "auto")),
                sample_rate=int(tts.get("sample_rate", 24000)),
                voice=str(tts.get("voice", "default")),
            ),
        ),
        pipeline=PipelineConfig(
            max_turns_per_session=int(pipeline.get("max_turns_per_session", 30)),
            max_duration_minutes=int(pipeline.get("max_duration_minutes", 50)),
            save_audio=bool(pipeline.get("save_audio", True)),
            save_embeddings=bool(pipeline.get("save_embeddings", True)),
            embedding_save_format=str(pipeline.get("embedding_save_format", "float32")),
        ),
        thermal=ThermalConfig(
            enabled=bool(thermal.get("enabled", True)),
            check_interval_seconds=int(thermal.get("check_interval_seconds", 30)),
            throttle_threshold=float(thermal.get("throttle_threshold", 85)),
            cooldown_seconds=int(thermal.get("cooldown_seconds", 120)),
        ),
        logging=LoggingConfig(
            level=str(logging_cfg.get("level", "INFO")),
            file=str(logging_cfg.get("file", "logs/psytest.log")),
            console=bool(logging_cfg.get("console", True)),
        ),
    )


def apply_env_overrides(config: AppConfig) -> AppConfig:
    """套用環境變數覆寫並回傳同一物件。

    Raises:
        ValueError: 環境變數格式錯誤時。
    """
    if os.getenv("OLLAMA_HOST"):
        config.models.llm.host = os.getenv("OLLAMA_HOST", "").strip()
    if os.getenv("OLLAMA_MODEL"):
        config.models.llm.model_name = os.getenv("OLLAMA_MODEL", "").strip()
    if os.getenv("PSYTEST_OUTPUT_DIR"):
        config.output_dir = os.getenv("PSYTEST_OUTPUT_DIR", "").strip()
    if os.getenv("PSYTEST_SAVE_AUDIO") is not None:
        config.pipeline.save_audio = _get_env_bool("PSYTEST_SAVE_AUDIO", True)
    if os.getenv("PSYTEST_SAVE_EMBEDDINGS") is not None:
        config.pipeline.save_embeddings = _get_env_bool(
            "PSYTEST_SAVE_EMBEDDINGS", True
        )
    max_mem = _get_env_float("PSYTEST_MAX_MEMORY_GB")
    if max_mem is not None:
        config.backend.max_memory_gb = max_mem
    if os.getenv("PSYTEST_GPU_LAYERS"):
        raw = os.getenv("PSYTEST_GPU_LAYERS", "").strip()
        config.backend.gpu_layers = int(raw) if raw.lstrip("-").isdigit() else raw
    temp_limit = _get_env_float("PSYTEST_TEMPERATURE_LIMIT")
    if temp_limit is not None:
        config.backend.temperature_limit = temp_limit
        config.thermal.throttle_threshold = temp_limit
    if os.getenv("PSYTEST_LOG_LEVEL"):
        config.logging.level = os.getenv("PSYTEST_LOG_LEVEL", "INFO").strip().upper()
    return config


def validate_config(config: AppConfig) -> list[str]:
    """驗證設定合理性，回傳警告訊息清單（錯誤直接拋出例外）。

    Raises:
        ValueError: 存在致命設定錯誤時。
    """
    warnings: list[str] = []
    if config.backend.force not in (None, "metal", "cuda", "rocm", "cpu"):
        raise ValueError(
            f"backend.force 非法：{config.backend.force!r}，"
            "允許值為 null、metal、cuda、rocm、cpu"
        )
    if config.models.llm.max_tokens <= 0:
        raise ValueError("llm.max_tokens 必須為正整數")
    if not 0.0 <= config.models.llm.temperature <= 2.0:
        warnings.append(
            f"llm.temperature={config.models.llm.temperature} 超出常見範圍 [0, 2]"
        )
    if config.models.embedding.context_length <= 0:
        raise ValueError("embedding.context_length 必須為正整數")
    if config.models.laya.context_length <= 0:
        raise ValueError("laya.context_length 必須為正整數")
    if config.pipeline.max_turns_per_session <= 0:
        raise ValueError("pipeline.max_turns_per_session 必須為正整數")
    if config.pipeline.embedding_save_format not in {"float32", "float16", "int8"}:
        raise ValueError(
            "pipeline.embedding_save_format 必須為 float32、float16 或 int8"
        )
    if config.logging.level.upper() not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise ValueError(f"logging.level 非法：{config.logging.level!r}")
    for label, rel in (
        ("embedding", config.models.embedding.path),
        ("laya", config.models.laya.path),
    ):
        if find_existing_model_path(rel) is None:
            warnings.append(
                f"找不到 {label} 模型檔（已嘗試別名）：{rel}，"
                "相關功能將無法執行，請依 models/README.md 取得檔案"
            )
    return warnings


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    """載入設定檔並套用環境變數覆寫與驗證。

    Args:
        path: 設定檔路徑（相對專案根目錄或絕對路徑）。

    Returns:
        驗證通過的 :class:`AppConfig`。

    Raises:
        FileNotFoundError: 設定檔不存在時。
        ValueError: 設定內容非法時。
    """
    data = load_yaml(path)
    config = _build_config(data)
    config.config_path = str(path)
    apply_env_overrides(config)
    warnings = validate_config(config)
    if warnings:
        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        for message in warnings:
            logger.warning(message)
    return config
