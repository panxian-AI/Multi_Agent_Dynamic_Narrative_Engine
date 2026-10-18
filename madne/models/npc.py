from pydantic import Field, field_validator

from .base import MADNEModel


class NPCProfile(MADNEModel):
    """NPC 档案。traits 为 0-100 的动态权重。"""

    npc_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str
    is_major: bool = True
    traits: dict[str, int] = Field(default_factory=dict)

    @field_validator("traits")
    @classmethod
    def _clamp_traits(cls, value: dict[str, int]) -> dict[str, int]:
        return {key: max(0, min(100, int(weight))) for key, weight in value.items()}
