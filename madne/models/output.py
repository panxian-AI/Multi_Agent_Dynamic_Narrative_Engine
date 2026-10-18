from typing import Any

from pydantic import Field, model_validator

from .base import MADNEModel
from .framework import Branch
from .image import Illustration
from .scene import SceneOutline
from .script import DialogueScript


class GameOutput(MADNEModel):
    """交付给游戏引擎的最终结构。"""

    scene: SceneOutline
    script: DialogueScript
    choices: list[Branch] = Field(min_length=1, max_length=5)
    world_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    session_id: str = Field(min_length=1)
    illustration: Illustration | None = None

    @model_validator(mode="after")
    def _scene_script_match(self) -> "GameOutput":
        if self.scene.scene_id != self.script.scene_id:
            raise ValueError("SceneOutline.scene_id 与 DialogueScript.scene_id 不一致")
        return self

    def to_engine_dict(self) -> dict[str, Any]:
        """转换为规格中的游戏引擎 JSON 形态。"""
        payload = self.model_dump(mode="json")
        payload["choices"] = [
            {"id": branch.branch_id, "text": branch.description}
            for branch in self.choices
        ]
        return payload
