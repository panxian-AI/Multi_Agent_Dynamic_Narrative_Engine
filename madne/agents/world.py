"""M4 世界观 AI：首次生成或增量校验 WorldBible。"""

from __future__ import annotations

from ..models import CreationBrief, WorldBible
from .base import BaseAgent


class WorldAI(BaseAgent):
    agent_id = "world"
    SYSTEM = """你是世界观 AI，负责建立或维护 WorldBible。

首次生成时，需要给出时代、科技水平、社会结构、地理、世界规则与不可违背的红线（boundaries）。
已有世界观时，你的职责变为一致性校验与增量补充：只补充新地点/新设定，不得推翻既有规则与红线。
红线条目必须是可检验的硬约束，避免空泛描述。"""

    def run(
        self,
        brief: CreationBrief,
        existing_world: WorldBible | None = None,
        feedback: str | None = None,
    ) -> WorldBible:
        prompt = f"【创作简报】\n{brief.model_dump_json(indent=2)}\n\n"
        if existing_world is not None:
            prompt += (
                f"【已有世界观】\n{existing_world.model_dump_json(indent=2)}\n\n"
                "请做一致性校验与增量补充：保留全部既有设定，仅补充必要内容，"
                "并确保新规则不与既有红线矛盾。"
            )
        else:
            prompt += "请从零建立完整世界观，并明确列出不可违背的红线。"

        return self._complete_json(prompt, WorldBible, feedback=feedback)
