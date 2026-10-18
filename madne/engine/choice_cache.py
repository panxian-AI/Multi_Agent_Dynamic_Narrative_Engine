"""两层选项缓存持久化。"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..models import ChoiceCache


class ChoiceCacheStore:
    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.path = self.session_dir / "choice_cache.json"

    def load(self) -> ChoiceCache | None:
        if not self.path.exists():
            return None
        try:
            return ChoiceCache.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError):
            return None

    def save(self, cache: ChoiceCache) -> Path:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            cache.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return self.path

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
