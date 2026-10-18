"""两层选项预取 AI：为当前每个分支提前生成未来两层可执行选项。"""

from __future__ import annotations

import json

from ..models import (
    ChoiceCache,
    CompressedMemory,
    SceneOutline,
    StoryFramework,
    WorldBible,
)
from .base import BaseAgent
from .prompts import world_redlines


class ChoiceCacheAI(BaseAgent):
    agent_id = "choice_cache"
    SYSTEM = """你是选项预取 AI。你的任务不是写剧情，而是为当前每个玩家选项预生成未来两层的可执行选择。

硬性要求：
1. 对每个当前分支，生成 3-4 个下一层选项；再为每个下一层选项生成 3 个第二层选项。
2. path 必须包含当前分支 ID 作为根：
   - 下一层：["当前分支ID", "下一层ID"]
   - 第二层：["当前分支ID", "下一层ID", "第二层ID"]
   所有 ID 必须稳定、简短、不含空格。
3. 每个选项必须是具体行动：动词 + 明确目标 + 代价/风险；禁止空泛情绪或纯氛围描述。
4. 必须衔接当前场景、压缩记忆中的未解伏笔与人物关系；不得违反世界红线。
5. 选项要推动主线，避免死路；每个选项都应能自然引出下一场戏。
6. requirements 写清执行前提；risk_level 只能是 low / medium / high。
7. 只输出 ChoiceCache JSON，不要生成场景、台词或故事正文。"""

    def run(
        self,
        world: WorldBible,
        framework: StoryFramework,
        scene: SceneOutline,
        compressed_memory: CompressedMemory | None = None,
        feedback: str | None = None,
    ) -> ChoiceCache:
        history = (
            compressed_memory.model_dump_json(indent=2)
            if compressed_memory is not None
            else "无历史压缩记忆"
        )
        current_branches = json.dumps(
            [branch.model_dump(mode="json") for branch in framework.branches],
            ensure_ascii=False,
        )
        prompt = (
            f"【当前分支（必须作为 path 根节点）】\n"
            f"{current_branches}\n\n"
            f"【当前场景】\n{scene.model_dump_json(indent=2)}\n\n"
            f"【当前框架】\n{framework.model_dump_json(indent=2)}\n\n"
            f"【压缩记忆】\n{history}\n\n"
            "请为每个当前分支预生成两层选项，输出 ChoiceCache。"
        )
        return self._complete_json(
            prompt,
            ChoiceCache,
            extra_system=world_redlines(world),
            feedback=feedback,
        )
