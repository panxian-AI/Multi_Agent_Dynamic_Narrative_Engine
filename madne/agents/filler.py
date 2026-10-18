"""M6 内容填充 AI：Beat -> SceneOutline，显式禁止台词。"""

from __future__ import annotations

from ..exceptions import AgentOutputError
from ..models import (
    Beat,
    NPCProfile,
    SceneOutline,
    StoryFramework,
    WorldBible,
)
from .base import BaseAgent
from .prompts import world_redlines

_QUOTE_CHARS = ('"', "“", "”", "「", "」", "『", "』")


class FillerAI(BaseAgent):
    agent_id = "filler"
    SYSTEM = """你是内容填充 AI，负责把单个故事节拍扩写为 SceneOutline。

硬性约束：
1. 只描述场景、环境、动作与事件经过。
2. 严禁任何角色对白，严禁使用引号包裹台词，严禁写剧本格式。
3. 场景必须推进当前节拍，并涉及给定 NPC。
4. 不得违反世界观红线。"""

    def run(
        self,
        framework: StoryFramework,
        world: WorldBible,
        npcs: list[NPCProfile],
        feedback: str | None = None,
    ) -> SceneOutline:
        beat = self._select_beat(framework)
        profiles = [npc.model_dump(mode="json") for npc in npcs]
        prompt = (
            f"【当前节拍】\n{beat.model_dump_json(indent=2)}\n\n"
            f"【完整框架】\n{framework.model_dump_json(indent=2)}\n\n"
            f"【涉及角色档案】\n{profiles}\n\n"
            "请输出该节拍对应的场景大纲。"
        )
        extra = world_redlines(world)
        outline = self._complete_json(
            prompt,
            SceneOutline,
            extra_system=extra,
            feedback=feedback,
        )

        if self._contains_dialogue(outline):
            corrective_prompt = (
                prompt
                + "\n\n【硬性纠正】\n上一次输出包含引号对白，违反禁止台词约束。"
                "请移除所有对白与引号内容，只保留动作和事件叙述，重新输出完整 JSON。"
            )
            outline = self._complete_json(
                corrective_prompt,
                SceneOutline,
                extra_system=extra,
                retries=0,
            )
            if self._contains_dialogue(outline):
                raise AgentOutputError("FillerAI 连续两次输出包含对白，违反 SceneOutline 约束")
        return outline

    @staticmethod
    def _select_beat(framework: StoryFramework) -> Beat:
        ordered = sorted(framework.key_beats, key=lambda beat: beat.order)
        for beat in ordered:
            if beat.status != "completed":
                return beat
        return ordered[-1]

    @staticmethod
    def _contains_dialogue(outline: SceneOutline) -> bool:
        texts = [outline.summary, outline.setting, *outline.actions, *outline.events]
        return any(char in text for text in texts for char in _QUOTE_CHARS)
