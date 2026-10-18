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
4. 不得违反世界观红线。
5. visual_prompt 用英文写一句用于文生图的镜头描述：人物外观、构图、光线、
   镜头与氛围；不得包含台词、字幕或画面内文字。"""

    def run(
        self,
        framework: StoryFramework,
        world: WorldBible,
        npcs: list[NPCProfile],
        previous_scene: SceneOutline | None = None,
        feedback: str | None = None,
    ) -> SceneOutline:
        beat = self._select_beat(framework)
        profiles = [npc.model_dump(mode="json") for npc in npcs]
        known_ids = sorted(npc.npc_id for npc in npcs)
        known_names = sorted(npc.name for npc in npcs)
        prompt = (
            f"【当前节拍】\n{beat.model_dump_json(indent=2)}\n\n"
            f"【完整框架】\n{framework.model_dump_json(indent=2)}\n\n"
            f"【角色档案】\n{profiles}\n\n"
            f"【可用 npc_id】{known_ids}\n"
            f"【角色姓名必须完全一致】{known_names}\n\n"
        )
        if previous_scene is not None:
            prompt += (
                f"【上一场戏】\n{previous_scene.model_dump_json(indent=2)}\n\n"
                "必须从上一场戏的结尾继续：不得重复其 setting、actions 或 events；"
                "必须体现玩家选择的直接后果，并推动当前节拍向前发展。\n\n"
            )
        prompt += (
            "请输出该节拍对应的场景大纲。involved_npcs 只能填写上述 npc_id；"
            "场景中的人物姓名必须与档案完全一致，不得为角色改名或发明新的核心角色。"
        )
        extra = world_redlines(world)
        outline = self._complete_json(
            prompt,
            SceneOutline,
            extra_system=extra,
            feedback=feedback,
        )

        problems = self._collect_problems(outline, known_ids)
        if problems:
            corrective_prompt = (
                prompt
                + "\n\n【硬性纠正】\n上一次输出存在以下问题：\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\n请修复上述问题，重新输出完整 JSON。"
            )
            outline = self._complete_json(
                corrective_prompt,
                SceneOutline,
                extra_system=extra,
                retries=0,
            )
            remaining = self._collect_problems(outline, known_ids)
            if remaining:
                raise AgentOutputError(
                    "FillerAI 连续两次输出违反约束: " + "；".join(remaining)
                )
        return outline

    @staticmethod
    def _select_beat(framework: StoryFramework) -> Beat:
        ordered = sorted(framework.key_beats, key=lambda beat: beat.order)
        for beat in ordered:
            if beat.status != "completed":
                return beat
        return ordered[-1]

    @classmethod
    def _collect_problems(
        cls,
        outline: SceneOutline,
        known_ids: list[str],
    ) -> list[str]:
        problems: list[str] = []
        unknown_ids = [
            npc_id
            for npc_id in outline.involved_npcs
            if npc_id not in known_ids
        ]
        if unknown_ids:
            problems.append(
                f"involved_npcs 使用了未登记 ID：{unknown_ids}；"
                f"只能使用：{known_ids}"
            )
        if cls._contains_dialogue(outline):
            problems.append("文本包含引号对白；SceneOutline 禁止任何台词")
        return problems

    @staticmethod
    def _contains_dialogue(outline: SceneOutline) -> bool:
        texts = [outline.summary, outline.setting, *outline.actions, *outline.events]
        return any(char in text for text in texts for char in _QUOTE_CHARS)
