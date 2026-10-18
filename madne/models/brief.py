from pydantic import Field

from .base import MADNEModel


class CreationBrief(MADNEModel):
    """IntentAI 的产出：只描述需求，不包含故事内容。"""

    genre: str
    style: str
    core_elements: list[str] = Field(default_factory=list)
    tone: str
    constraints: list[str] = Field(default_factory=list)
