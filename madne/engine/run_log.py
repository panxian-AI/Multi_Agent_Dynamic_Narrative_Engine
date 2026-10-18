"""结构化运行日志：每次步骤尝试都追加落盘（JSONL）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


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
    """内存中保留全部条目，同时按需追加到 JSONL 文件。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.entries: list[RunLogEntry] = []

    def record(self, **kwargs) -> RunLogEntry:
        entry = RunLogEntry(**kwargs)
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def for_step(self, step_id: str) -> list[RunLogEntry]:
        return [entry for entry in self.entries if entry.step_id == step_id]

    def statuses(self) -> list[str]:
        return [entry.status for entry in self.entries]
