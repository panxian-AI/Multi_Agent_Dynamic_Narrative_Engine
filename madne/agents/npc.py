"""NPC 角色抽取 AI：让角色池跟随 StoryFramework，而不是使用固定演示名单。"""

from __future__ import annotations

from ..models import (
    CastRoster,
    CreationBrief,
    NPCProfile,
    StoryFramework,
    WorldBible,
)
from .base import BaseAgent
from .prompts import world_redlines


class NPCAI(BaseAgent):
    agent_id = "npc"
    SYSTEM = """你是 NPC 设定 AI。请从 StoryFramework 中抽取本次故事的角色池，输出 CastRoster。

硬性要求：
1. name 必须与 StoryFramework 中使用的称呼完全一致，不得改名、合并或替换人物。
2. npc_id 使用稳定的 ASCII 标识（npc_ 前缀），同一人物在后续回合中不得更换 ID。
3. 必须包含玩家主角与当前节拍涉及的核心角色；背景角色可以省略。
4. role 简洁描述身份与剧情功能；is_major 标记核心角色。
5. traits 为 0-100 整数，至少包含 honesty、selfishness、loyalty、curiosity；
   数值必须能合理解释角色在框架中的行为。
6. appearance 用一句稳定的外观描述（年龄、发型、服装、体态、标志性配饰、
   义体或伤疤等），供后续插图保持人物一致性；同一角色在后续回合不得改变。
7. 不得添加框架中不存在的核心角色，不得违反世界观红线。"""

    def run(
        self,
        brief: CreationBrief,
        world: WorldBible,
        framework: StoryFramework,
        existing_npcs: list[NPCProfile] | None = None,
        feedback: str | None = None,
    ) -> CastRoster:
        existing = [
            profile.model_dump(mode="json")
            for profile in (existing_npcs or [])
        ]
        prompt = (
            f"【创作简报】\n{brief.model_dump_json(indent=2)}\n\n"
            f"【StoryFramework】\n{framework.model_dump_json(indent=2)}\n\n"
            f"【已有角色池】\n{existing}\n\n"
            "请抽取/更新角色池。若已有角色仍出现在框架中，保留其 npc_id，"
            "并根据框架微调 role 与 traits；允许新增框架中明确出现的新角色。"
        )
        return self._complete_json(
            prompt,
            CastRoster,
            extra_system=world_redlines(world),
            feedback=feedback,
        )
