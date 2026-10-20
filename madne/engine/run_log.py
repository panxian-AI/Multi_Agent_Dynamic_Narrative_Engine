"""结构化运行日志：写入 SQLite（可选 JSONL 兼容模式）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage import SessionDatabase


@dataclass(slots=True)
class RunLogEntry:
    cycle: int
    step_id: str
    agent: str
    status: str
    attempt: int = 1
    tokens_used: int = 0
    duration_ms: int = 0
    error: str | None = None
    detail: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RunLog:
    """内存中保留条目；优先写数据库，未注入数据库时退回 JSONL。"""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        database: "SessionDatabase | None" = None,
        session_id: str = "",
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.database = database
        self.session_id = session_id
        self.entries: list[RunLogEntry] = []
        if database is not None and session_id:
            self.entries = [
                RunLogEntry(**row)
                for row in database.load_run_log(session_id)
            ]

    def record(self, **kwargs) -> RunLogEntry:
        entry = RunLogEntry(**kwargs)
        self.entries.append(entry)
        if self.database is not None and self.session_id:
            self.database.save_run_log(self.session_id, asdict(entry))
        elif self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def for_step(self, step_id: str) -> list[RunLogEntry]:
        return [entry for entry in self.entries if entry.step_id == step_id]

    def statuses(self) -> list[str]:
        return [entry.status for entry in self.entries]
