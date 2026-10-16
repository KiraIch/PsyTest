"""PsyTest 日誌系統。

提供全專案統一的日誌設定，支援同時輸出至終端與檔案，
並可經由環境變數 ``PSYTEST_LOG_LEVEL`` 覆寫日誌等級。

使用範例：
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("管線啟動")
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_INITIALIZED_LOGGERS: set[str] = set()

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _resolve_level(level: Optional[str]) -> str:
    """解析日誌等級，優先順序：參數 > 環境變數 > 預設 INFO。"""
    candidate = level or os.getenv("PSYTEST_LOG_LEVEL", "INFO")
    normalized = candidate.strip().upper()
    if normalized not in _VALID_LEVELS:
        raise ValueError(
            f"無效的日誌等級：{candidate!r}，允許值為 {sorted(_VALID_LEVELS)}"
        )
    return normalized


def get_logger(
    name: str = "psytest",
    level: Optional[str] = None,
    log_file: Optional[str | Path] = None,
    console: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """取得（或建立）具名 logger，保證 handler 不重複掛載。

    Args:
        name: logger 名稱，建議傳入 ``__name__``。
        level: 日誌等級，未指定時讀取 ``PSYTEST_LOG_LEVEL``，預設 INFO。
        log_file: 日誌檔路徑，未指定時僅輸出至終端。
        console: 是否輸出至標準輸出。
        max_bytes: 單一檔案上限（輪替用）。
        backup_count: 保留的輪替檔案數。

    Returns:
        設定完成的 :class:`logging.Logger`。

    Raises:
        ValueError: 日誌等級無效時。
        OSError: 日誌檔父目錄無法建立時。
    """
    resolved = _resolve_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, resolved))
    logger.propagate = False

    if name in _INITIALIZED_LOGGERS:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        if console:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(getattr(logging, resolved))
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if log_file is not None:
            path = Path(log_file)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise OSError(f"無法建立日誌目錄：{path.parent}：{exc}") from exc
            file_handler = RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(getattr(logging, resolved))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    except Exception:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        raise

    _INITIALIZED_LOGGERS.add(name)
    return logger


def reset_logger(name: str = "psytest") -> None:
    """移除指定 logger 的所有 handler（主要供測試使用）。"""
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _INITIALIZED_LOGGERS.discard(name)
