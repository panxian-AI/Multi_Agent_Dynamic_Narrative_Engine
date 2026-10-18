"""M5 框架 AI：主线 + 关键节拍 + 有限分支。"""

from __future__ import annotations

from ..models import (
    Branch,
    CachedChoice,
    CompressedMemory,
    PlayerInput,
    SceneOutline,
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
5. 不得违反世界观红线；不得引入 WorldBible 未定义的新技术、新组织规则或新的硬性设定，
   剧情需要时只能在既有规则范围内做合理延伸。"""

    def run(
        self,
        world: WorldBible,
        compressed_memory: CompressedMemory | None,
        player_input: PlayerInput,
        previous_framework: StoryFramework | None = None,
        previous_scene: SceneOutline | None = None,
        prefetched_choices: list[CachedChoice] | None = None,
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
                f"【上一轮框架（正史，不得推翻）】\n"
                f"{previous_framework.model_dump_json(indent=2)}\n\n"
            )
            if previous_scene is not None:
                prompt += (
                    f"【上一场戏的结尾】\n"
                    f"{previous_scene.model_dump_json(indent=2)}\n\n"
                )
            prompt += (
                "请从上一场戏的结尾继续推进：completed 节拍保持 completed；"
                "把当前 active/pending 节拍推进一步；必须体现玩家选择的直接后果；"
                "不得重启故事、不得重复同一节拍、不得重置人物关系或已建立的伏笔。"
            )
        else:
            prompt += "请建立主线、关键节拍与当前可选分支。"

        if prefetched_choices:
            prompt += (
                "\n\n【预取选项（必须作为本次 branches）】\n"
                f"{[choice.model_dump(mode='json') for choice in prefetched_choices]}\n"
                "本次 StoryFramework.branches 必须使用这些预取选项的 ID 与描述，"
                "不得改写、删除或另造分支；请围绕它们编写 main_plot 与 key_beats。"
            )

        framework = self._complete_json(
            prompt,
            StoryFramework,
            extra_system=world_redlines(world),
            feedback=feedback,
        )
        return self._apply_prefetched_branches(framework, prefetched_choices)

    @staticmethod
    def _apply_prefetched_branches(
        framework: StoryFramework,
        prefetched_choices: list[CachedChoice] | None,
    ) -> StoryFramework:
        if not prefetched_choices:
            return framework
        layer_one = [
            choice
            for choice in prefetched_choices
            if len(choice.path) == 2
        ][:5]
        if len(layer_one) < 3:
            return framework

        generated = {branch.branch_id: branch for branch in framework.branches}
        branches = [
            Branch(
                branch_id=choice.path[-1],
                description=choice.description,
                risk_level=choice.risk_level,
                requirements=(
                    choice.requirements
                    or generated.get(choice.path[-1], Branch(
                        branch_id=choice.path[-1],
                        description=choice.description,
                    )).requirements
                ),
            )
            for choice in layer_one
        ]
        return framework.model_copy(update={"branches": branches})
