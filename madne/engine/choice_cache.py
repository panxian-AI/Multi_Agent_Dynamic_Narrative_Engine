"""两层选项缓存持久化（SQLite 优先，文件模式兼容）。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..models import ChoiceCache

if TYPE_CHECKING:
    from ..storage import SessionDatabase


class ChoiceCacheStore:
    def __init__(
        self,
        session_dir: str | Path,
        *,
        database: "SessionDatabase | None" = None,
        session_id: str = "",
    ) -> None:
        self.session_dir = Path(session_dir)
        self.path = self.session_dir / "choice_cache.json"
        self.database = database
        self.session_id = session_id

    def load(self) -> ChoiceCache | None:
        if self.database is not None and self.session_id:
            payload = self.database.load_choice_cache(self.session_id)
            if payload is None:
                return None
            try:
                return ChoiceCache.model_validate(payload)
            except (ValidationError, ValueError):
                return None
        if not self.path.exists():
            return None
        try:
            return ChoiceCache.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError):
            return None

    def save(self, cache: ChoiceCache) -> Path:
        if self.database is not None and self.session_id:
            self.database.save_choice_cache(
                self.session_id, cache.model_dump(mode="json")
            )
            return self.database.path
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            cache.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return self.path

    def clear(self) -> None:
        if self.database is not None and self.session_id:
            self.database.clear_choice_cache(self.session_id)
            return
        if self.path.exists():
            self.path.unlink()
