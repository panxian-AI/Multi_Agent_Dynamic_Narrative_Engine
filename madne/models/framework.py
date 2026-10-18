from typing import Literal

from pydantic import Field

from .base import MADNEModel

BeatStatus = Literal["pending", "active", "completed"]
RiskLevel = Literal["low", "medium", "high"]


class Beat(MADNEModel):
    """主线中的关键节拍。"""

    beat_id: str = Field(min_length=1)
    title: str
    summary: str
    order: int = 0
    status: BeatStatus = "pending"


class Branch(MADNEModel):
    """有限分支，直接供游戏引擎渲染为选项。"""

    branch_id: str = Field(min_length=1)
    description: str
    risk_level: RiskLevel = "medium"
    requirements: list[str] = Field(default_factory=list)


class StoryFramework(MADNEModel):
    """主线 + 关键节拍 + 有限分支（禁止穷举分支树）。"""

    main_plot: str
    key_beats: list[Beat] = Field(min_length=1)
    branches: list[Branch] = Field(min_length=3, max_length=5)
