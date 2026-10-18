from pydantic import Field, field_validator

from .base import MADNEModel


class NPCProfile(MADNEModel):
    """NPC 档案。traits 为 0-100 的动态权重。

    ``appearance`` 是跨回合稳定的生图外观描述；``reference_image`` 可选，
    指向本地定妆照（相对 data_dir 或绝对路径），用于 InstantID 身份注入。
    """

    npc_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str
    is_major: bool = True
    traits: dict[str, int] = Field(default_factory=dict)
    appearance: str = ""
    reference_image: str | None = None

    @field_validator("traits")
    @classmethod
    def _clamp_traits(cls, value: dict[str, int]) -> dict[str, int]:
        return {key: max(0, min(100, int(weight))) for key, weight in value.items()}


class CastRoster(MADNEModel):
    """从 StoryFramework 抽取的角色池，作为 NPCSystem 的初始化数据。"""

    characters: list[NPCProfile] = Field(min_length=1)
