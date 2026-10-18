from pydantic import Field

from .base import MADNEModel


class SceneOutline(MADNEModel):
    """场景大纲：只描述场景、动作与事件，显式禁止台词。"""

    scene_id: str = Field(min_length=1)
    summary: str
    setting: str
    actions: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    involved_npcs: list[str] = Field(default_factory=list)
    # 供文生图使用的英文镜头描述（不参与叙事语义，可为空）
    visual_prompt: str = ""
