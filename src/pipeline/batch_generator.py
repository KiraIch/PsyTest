"""批量生成排程器（Phase 5 實作）。

管理多場會談的生成：讀取生成計畫、斷點續傳、進度顯示。
預設串行生成（安全優先），並行控制為未來工作。

檢查點檔案：``<output_dir>/checkpoints/progress.json``，
記錄已完成的（個案，場次）組合，中斷後以 ``--resume`` 續跑。

使用範例：
    from src.config import load_config
    from src.pipeline.batch_generator import BatchGenerator
    from src.pipeline.output_formatter import OutputFormatter
    from src.pipeline.session_generator import SessionGenerator

    config = load_config()
    formatter = OutputFormatter("data/output/batch_001/")
    generator = SessionGenerator.from_config(config, formatter)
    BatchGenerator(config, generator, formatter).run(
        ["data/personas/mdd_severe.yaml"], sessions_per_persona=2,
        output_dir="data/output/batch_001/",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchGenerator:
    """批量生成排程器。"""

    def __init__(
        self,
        config: AppConfig,
        session_generator: Optional[object] = None,
        output_formatter: Optional[object] = None,
        session_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        """初始化排程器。

        Args:
            config: 應用設定。
            session_generator: 單場生成器（與 session_factory 二選一）。
            output_formatter: 輸出格式化器（可空，由生成器持有亦可）。
            session_factory: 生成器工廠（測試注入或多進程擴充用）。

        Raises:
            ValueError: 生成器與工廠皆未提供時。
        """
        if session_generator is None and session_factory is None:
            raise ValueError("須提供 session_generator 或 session_factory")
        self.config = config
        self.session_generator = session_generator
        self.session_factory = session_factory
        self.output_formatter = output_formatter

    def _checkpoint_path(self, output_dir: str | Path) -> Path:
        """回傳檢查點路徑（自動建目錄）。"""
        path = Path(output_dir) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path / "progress.json"

    def _load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """讀取檢查點，缺失或損壞時回傳空進度。"""
        if not path.exists():
            return {"completed": []}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("completed"), list):
                return data
        except (OSError, ValueError) as exc:
            logger.warning(f"檢查點讀取失敗，重新開始：{exc}")
        return {"completed": []}

    def _save_checkpoint(self, path: Path, completed: List[str]) -> None:
        """寫入檢查點（失敗僅記錄警告，不中斷生成）。"""
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"completed": completed}, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning(f"檢查點寫入失敗：{exc}")

    def _resolve_generator(self) -> object:
        """取得本輪使用的生成器。"""
        if self.session_factory is not None:
            return self.session_factory()
        assert self.session_generator is not None
        return self.session_generator

    def run(
        self,
        personas: List[str],
        sessions_per_persona: int,
        output_dir: str,
        scenario: str = "data/scenarios/initial_session.yaml",
        resume: bool = False,
        on_session: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """執行批量生成。

        Args:
            personas: 個案 YAML 路徑清單。
            sessions_per_persona: 每個案場次數（正整數）。
            output_dir: 輸出目錄。
            scenario: 場景 YAML 路徑。
            resume: 是否由檢查點續跑。
            on_session: 每場完成時的回呼（可空）。

        Returns:
            統計字典（含 total、succeeded、failed、sessions）。

        Raises:
            ValueError: 參數非法時。
        """
        if not personas:
            raise ValueError("personas 不可為空")
        if sessions_per_persona <= 0:
            raise ValueError("sessions_per_persona 必須為正整數")
        if not Path(scenario).exists():
            raise ValueError(f"場景檔案不存在：{scenario}")
        checkpoint = self._checkpoint_path(output_dir)
        completed: List[str] = list(self._load_checkpoint(checkpoint)["completed"]) if resume else []
        sessions: List[Dict[str, Any]] = []
        failed = 0
        total = len(personas) * sessions_per_persona
        done = 0
        for persona_path in personas:
            for index in range(1, sessions_per_persona + 1):
                key = f"{Path(persona_path).stem}#{index}"
                done += 1
                if key in completed:
                    logger.info(f"跳過已完成：{key}（{done}/{total}）")
                    continue
                logger.info(f"生成中：{key}（{done}/{total}）")
                try:
                    generator = self._resolve_generator()
                    session = generator.generate(persona_path, scenario)  # type: ignore[union-attr]
                    sessions.append(session)
                    if self.output_formatter is not None:
                        try:
                            self.output_formatter.write_session(session)  # type: ignore[union-attr]
                            self.output_formatter.write_annotations(  # type: ignore[union-attr]
                                session["session_id"],
                                {"summary": session.get("summary", {}), "end_reason": session.get("end_reason", "")},
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"會談輸出寫入失敗：{exc}")
                    completed.append(key)
                    self._save_checkpoint(checkpoint, completed)
                    if on_session is not None:
                        on_session(session)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.warning(f"會談 {key} 生成失敗（繼續下一場）：{exc}")
        result = {
            "total": total,
            "succeeded": len(sessions),
            "failed": failed,
            "sessions": sessions,
        }
        logger.info(f"批量生成完成：成功 {len(sessions)}/{total}，失敗 {failed}")
        return result
