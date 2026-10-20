"""M2 任务规划 AI：动态编排 Agent 链，输出可调度的 TaskPlan。"""

from __future__ import annotations

from ..engine.templates import build_plan, catalog_text, compile_plan
from ..exceptions import EngineError
from ..models import (
    CompressedMemory,
    CreationBrief,
    PlannerDecision,
    TaskPlan,
)
from .base import BaseAgent


class PlannerAI(BaseAgent):
    agent_id = "planner"
    SYSTEM = """你是任务规划 AI。你的职责是根据本轮上下文动态编排叙事工作流。

硬性约束：
1. 只能使用【Agent 能力表】中列出的 agent，禁止发明新步骤或修改 Agent 名称。
2. steps 必须符合依赖关系：消费者排在生产者之后。
3. 必须覆盖交付链路：framework -> filler -> script；首次创作还需要先产出 world。
4. 已有世界观时不要重复生成 world；角色更新使用 npc_update。
5. 步骤数量控制在 4-6 个，不要包含纯展示或可选的冗余步骤。
6. 只输出 PlannerDecision JSON；steps 每项只填 agent 与 reason。"""

    def plan(
        self,
        brief: CreationBrief,
        compressed_memory: CompressedMemory | None = None,
        has_existing_world: bool = False,
        available_keys: set[str] | None = None,
        max_steps: int = 8,
        dynamic: bool = True,
    ) -> TaskPlan:
        history = (
            compressed_memory.model_dump_json(indent=2)
            if compressed_memory is not None
            else "无历史记忆"
        )
        keys_text = "、".join(sorted(available_keys or ())) or "无"
        prompt = (
            f"【创作简报】\n{brief.model_dump_json(indent=2)}\n\n"
            f"【历史压缩记忆】\n{history}\n\n"
            f"【是否已有世界观】{'是' if has_existing_world else '否'}\n\n"
            f"【当前可用上下文键】\n{keys_text}\n\n"
            f"【Agent 能力表】\n{catalog_text()}\n\n"
            "【输出要求】\n请输出 PlannerDecision：在 steps 中按执行顺序给出 "
            f"不超过 {max_steps} 个 agent，并说明 reason；"
            "priority_focus 与 pacing 用于指导后续写作。"
        )
        decision = self._complete_json(prompt, PlannerDecision, retries=1)

        focus = decision.priority_focus or brief.core_elements
        if dynamic:
            try:
                return compile_plan(
                    decision,
                    has_existing_world=has_existing_world,
                    available_keys=set(available_keys or ()),
                    priority_focus=focus,
                    pacing=decision.pacing,
                    max_steps=max_steps,
                )
            except EngineError:
                pass

        template = decision.template or (
            "continuation" if has_existing_world else "cold_start"
        )
        if template not in {"cold_start", "continuation"}:
            template = "continuation" if has_existing_world else "cold_start"
        if has_existing_world and template != "continuation":
            template = "continuation"
        return build_plan(
            template,
            priority_focus=focus,
            pacing=decision.pacing,
            rationale=decision.rationale,
            max_steps=max_steps,
        )
