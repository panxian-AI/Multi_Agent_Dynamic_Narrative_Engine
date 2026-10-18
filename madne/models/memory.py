from datetime import datetime, timezone

from pydantic import Field

from .base import MADNEModel


class MemoryEvent(MADNEModel):
    """完整记忆库中的一条追加事件。"""

    session_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_content: str
    player_choice: str | None = None


class CompressedMemory(MADNEModel):
    """压缩记忆：创作阶段唯一注入的长期记忆。"""

    session_id: str = ""
    player_preferences: list[str] = Field(default_factory=list)
    unresolved_hooks: list[str] = Field(default_factory=list)
    relationship_changes: list[str] = Field(default_factory=list)
    long_term_goals: list[str] = Field(default_factory=list)
