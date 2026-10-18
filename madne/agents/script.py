"""M7 台本 AI：SceneOutline + NPC traits -> DialogueScript。"""

from __future__ import annotations

from ..exceptions import AgentOutputError
from ..models import DialogueScript, NPCProfile, SceneOutline, WorldBible
from .base import BaseAgent
from .prompts import world_redlines


class ScriptAI(BaseAgent):
    agent_id = "script"
    SYSTEM = """你是台本 AI，负责为场景撰写角色对白。

要求：
1. 每句台词必须标注 speaker、text、emotion，可选 inner_monologue。
2. speaker 必须与给定 NPC 档案中的 name 完全一致，禁止使用旁白、未登记角色或临时改名。
3. text 只能是角色实际说出口的话；动作、环境、舞台指示、括号说明一律不得写进 text。
4. 内心活动只能写入 inner_monologue。
5. 必须严格体现 NPC 的 traits 权重：权重 >= 70 的特质要在措辞、态度或潜台词中明显体现。
6. 台词要与场景动作、事件经过一致，不得新增未铺垫的重大设定。
7. 不得违反世界观红线。"""

    def run(
        self,
        scene: SceneOutline,
        npcs: list[NPCProfile],
        world: WorldBible | None = None,
        feedback: str | None = None,
    ) -> DialogueScript:
        profiles = [npc.model_dump(mode="json") for npc in npcs]
        known_names = sorted(npc.name for npc in npcs)
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
            f"【允许的 speaker（必须完全一致）】{known_names}\n\n"
            f"【必须体现的高权重特质】\n{high_weights}\n\n"
            "请输出完整 DialogueScript，scene_id 必须与场景一致。"
        )
        extra = world_redlines(world) if world is not None else None
        script = self._complete_json(
            prompt,
            DialogueScript,
            extra_system=extra,
            feedback=feedback,
        )

        unknown_speakers = sorted(
            {
                line.speaker
                for line in script.lines
                if line.speaker not in known_names
            }
        )
        if unknown_speakers:
            corrective_prompt = (
                prompt
                + f"\n\n【硬性纠正】\n上一次输出使用了未登记的 speaker："
                f"{unknown_speakers}。只能使用：{known_names}。"
                "请保持剧情不变，修正 speaker 后重新输出完整 JSON。"
            )
            script = self._complete_json(
                corrective_prompt,
                DialogueScript,
                extra_system=extra,
                retries=0,
            )
            remaining = sorted(
                {
                    line.speaker
                    for line in script.lines
                    if line.speaker not in known_names
                }
            )
            if remaining:
                raise AgentOutputError(
                    f"ScriptAI 连续两次输出未登记的 speaker: {remaining}"
                )
        return script
