"""M5 框架 AI：主线 + 关键节拍 + 有限分支。"""

from __future__ import annotations

from ..models import (
    CompressedMemory,
    PlayerInput,
    StoryFramework,
    WorldBible,
)
from .base import BaseAgent
from .prompts import world_redlines


class FrameworkAI(BaseAgent):
    agent_id = "framework"
    SYSTEM = """你是情节框架 AI，负责输出 StoryFramework。

要求：
1. main_plot：一条清晰主线。
2. key_beats：3-6 个有序关键节拍，每个节拍有可推进的 summary。
3. branches：3-5 个有限分支，禁止穷举分支树；分支将直接渲染为玩家选项。
4. 必须衔接压缩记忆中的未解伏笔与人物关系。
5. 不得违反世界观红线。"""

    def run(
        self,
        world: WorldBible,
        compressed_memory: CompressedMemory | None,
        player_input: PlayerInput,
        previous_framework: StoryFramework | None = None,
        feedback: str | None = None,
    ) -> StoryFramework:
        history = (
            compressed_memory.model_dump_json(indent=2)
            if compressed_memory is not None
            else "无历史压缩记忆"
        )
        prompt = (
            f"【玩家本次输入】\n{player_input.text}\n\n"
            f"【压缩记忆】\n{history}\n\n"
            f"【世界观】\n{world.model_dump_json(indent=2)}\n\n"
        )
        if previous_framework is not None:
            prompt += (
                f"【上一轮框架】\n{previous_framework.model_dump_json(indent=2)}\n\n"
                "请基于玩家选择推进主线：将已完成节拍标记为 completed，"
                "推进到下一个 pending 节拍，并更新分支。"
            )
        else:
            prompt += "请建立主线、关键节拍与当前可选分支。"

        return self._complete_json(
            prompt,
            StoryFramework,
            extra_system=world_redlines(world),
            feedback=feedback,
        )
