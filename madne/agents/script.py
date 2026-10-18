"""M7 台本 AI：SceneOutline + NPC traits -> DialogueScript。"""

from __future__ import annotations

from ..models import DialogueScript, NPCProfile, SceneOutline, WorldBible
from .base import BaseAgent
from .prompts import world_redlines


class ScriptAI(BaseAgent):
    agent_id = "script"
    SYSTEM = """你是台本 AI，负责为场景撰写角色对白。

要求：
1. 每句台词必须标注 speaker、text、emotion，可选 inner_monologue。
2. 必须严格体现 NPC 的 traits 权重：权重 >= 70 的特质要在措辞、态度或潜台词中明显体现。
3. 台词要与场景动作、事件经过一致，不得新增未铺垫的重大设定。
4. 不得违反世界观红线。"""

    def run(
        self,
        scene: SceneOutline,
        npcs: list[NPCProfile],
        world: WorldBible | None = None,
        feedback: str | None = None,
    ) -> DialogueScript:
        profiles = [npc.model_dump(mode="json") for npc in npcs]
        high_weights = {
            npc.name: {
                trait: weight
                for trait, weight in npc.traits.items()
                if weight >= 70
            }
            for npc in npcs
        }
        prompt = (
            f"【场景大纲】\n{scene.model_dump_json(indent=2)}\n\n"
            f"【角色完整档案（含当前 traits）】\n{profiles}\n\n"
            f"【必须体现的高权重特质】\n{high_weights}\n\n"
            "请输出完整 DialogueScript，scene_id 必须与场景一致。"
        )
        extra = world_redlines(world) if world is not None else None
        return self._complete_json(
            prompt,
            DialogueScript,
            extra_system=extra,
            feedback=feedback,
        )
