"""M10 记忆系统：完整事件流（JSONL）+ 压缩记忆（JSON）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from ..agents.prompts import MEMORY_COMPRESSION_SYSTEM
from ..llm.base import LLMClient, extract_json
from ..models import CompressedMemory, MemoryEvent


class MemorySystem:
    """读路径走压缩记忆，完整事件仅用于回溯与新压缩。"""

    def __init__(
        self,
        session_dir: str | Path,
        llm: LLMClient,
        compress_every: int = 10,
        session_id: str = "",
    ) -> None:
        self.session_dir = Path(session_dir)
        self.llm = llm
        self.compress_every = max(1, compress_every)
        self.session_id = session_id
        self.memory_dir = self.session_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.full_path = self.memory_dir / "full.jsonl"
        self.compressed_path = self.memory_dir / "compressed.json"
        self.meta_path = self.memory_dir / "meta.json"
        self.last_error: str | None = None

    def record(self, event: MemoryEvent) -> None:
        with self.full_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def read_events(self, limit: int | None = None) -> list[MemoryEvent]:
        if not self.full_path.exists():
            return []
        events: list[MemoryEvent] = []
        for line in self.full_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(MemoryEvent.model_validate_json(line))
            except ValidationError:
                continue
        if limit is not None:
            return events[-limit:]
        return events

    def event_count(self) -> int:
        return len(self.read_events())

    def load_compressed(self) -> CompressedMemory | None:
        if not self.compressed_path.exists():
            return None
        try:
            return CompressedMemory.model_validate_json(
                self.compressed_path.read_text(encoding="utf-8")
            )
        except (ValidationError, OSError, json.JSONDecodeError):
            return None

    def save_compressed(self, memory: CompressedMemory) -> None:
        self.compressed_path.write_text(
            memory.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def maybe_compress(self, force: bool = False) -> CompressedMemory | None:
        """每 N 个事件（或 force）触发一次压缩，并合并历史压缩记忆。"""
        self.last_error = None
        count = self.event_count()
        if count == 0:
            return self.load_compressed()

        last_count = self._load_meta().get("last_compressed_event_count", 0)
        if not force and count - last_count < self.compress_every:
            return self.load_compressed()

        events = self.read_events(limit=max(self.compress_every, 20))
        event_text = "\n".join(
            event.model_dump_json() for event in events
        )
        prompt = (
            "【近期事件流】\n"
            f"{event_text}\n\n"
            "请提炼 CompressedMemory；session_id 可留空。"
        )

        try:
            raw = self.llm.complete(
                prompt,
                system=MEMORY_COMPRESSION_SYSTEM,
                json_schema=CompressedMemory.model_json_schema(),
            )
            fresh = CompressedMemory.model_validate_json(extract_json(raw))
        except Exception as exc:  # 压缩失败不应中断叙事主流程
            self.last_error = f"记忆压缩失败: {exc}"
            return self.load_compressed()

        previous = self.load_compressed() or CompressedMemory(
            session_id=self.session_id
        )
        merged = self.merge(previous, fresh)
        self.save_compressed(merged)
        self._save_meta({"last_compressed_event_count": count})
        return merged

    def merge(
        self,
        previous: CompressedMemory,
        fresh: CompressedMemory,
    ) -> CompressedMemory:
        return CompressedMemory(
            session_id=self.session_id or fresh.session_id or previous.session_id,
            player_preferences=self._merge_lists(
                fresh.player_preferences, previous.player_preferences
            ),
            unresolved_hooks=self._merge_lists(
                fresh.unresolved_hooks, previous.unresolved_hooks
            ),
            relationship_changes=self._merge_lists(
                fresh.relationship_changes, previous.relationship_changes
            ),
            long_term_goals=self._merge_lists(
                fresh.long_term_goals, previous.long_term_goals
            ),
        )

    @staticmethod
    def _merge_lists(primary: Iterable[str], secondary: Iterable[str]) -> list[str]:
        merged: list[str] = []
        for item in [*primary, *secondary]:
            if item and item not in merged:
                merged.append(item)
        return merged

    def _load_meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_meta(self, meta: dict) -> None:
        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
