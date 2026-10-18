from pydantic import Field

from .base import MADNEModel


class WorldBible(MADNEModel):
    """世界观圣经。boundaries 为后续所有 Agent 必须遵守的红线。"""

    era: str
    tech_level: str
    social_structure: str
    geography: str
    rules: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
