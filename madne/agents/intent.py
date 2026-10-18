"""M1 意图理解 AI：只做分析，不创作。"""

from __future__ import annotations

from ..exceptions import AgentOutputError, IntentParseError
from ..models import CompressedMemory, CreationBrief, PlayerInput
from .base import BaseAgent


class IntentAI(BaseAgent):
    agent_id = "intent"
    SYSTEM = """你是意图理解 AI，只做需求分析，禁止创作任何故事内容、角色或情节。

从玩家输入与历史压缩记忆中抽取：
- genre：题材
- style：叙事风格
- core_elements：核心元素
- tone：情感基调
- constraints：约束条件

信息缺失时基于玩家输入做合理推断，不要向玩家提问，不要输出分析过程。"""

    def run(
        self,
        player_input: PlayerInput,
        compressed_memory: CompressedMemory | None = None,
    ) -> CreationBrief:
        history = (
            compressed_memory.model_dump_json(indent=2)
            if compressed_memory is not None
            else "无历史记忆"
        )
        prompt = (
            f"【玩家输入】\n{player_input.text}\n\n"
            f"【历史压缩记忆】\n{history}\n\n"
            "请分析玩家需求，输出 CreationBrief。"
        )
        try:
            return self._complete_json(prompt, CreationBrief, retries=1)
        except AgentOutputError as exc:
            raise IntentParseError(str(exc)) from exc
