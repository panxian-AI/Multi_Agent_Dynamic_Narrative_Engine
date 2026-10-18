from pydantic import Field

from .base import MADNEModel


class DialogueLine(MADNEModel):
    """单句台词。"""

    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    inner_monologue: str | None = None
    emotion: str = "neutral"


class DialogueScript(MADNEModel):
    """一场戏的完整台本。"""

    scene_id: str = Field(min_length=1)
    lines: list[DialogueLine] = Field(min_length=1)
