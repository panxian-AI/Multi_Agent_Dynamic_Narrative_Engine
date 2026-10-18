"""M2 任务规划 AI：只选择/裁剪静态模板，不凭空生成步骤。"""

from __future__ import annotations

from ..engine.templates import TEMPLATE_STEPS, available_templates, build_plan
from ..models import (
    CompressedMemory,
    CreationBrief,
    PlannerDecision,
    TaskPlan,
)
from .base import BaseAgent


class PlannerAI(BaseAgent):
    agent_id = "planner"
    SYSTEM = """你是任务规划 AI。你的职责是为本次叙事生成选择工作流模板。

硬性约束：
1. 只能从给定模板中选择一个，禁止发明新步骤或修改 Agent 名称。
2. 首次创作（无已有世界观）选择 cold_start。
3. 续写（已有世界观与历史）选择 continuation。
4. 只输出 PlannerDecision JSON，不要输出完整步骤列表。"""

    def plan(
        self,
        brief: CreationBrief,
        compressed_memory: CompressedMemory | None = None,
        has_existing_world: bool = False,
    ) -> TaskPlan:
        template_lines = "\n".join(
            f"- {name}: " + " -> ".join(agents)
            for name, agents in available_templates().items()
        )
        history = (
            compressed_memory.model_dump_json(indent=2)
            if compressed_memory is not None
            else "无历史记忆"
        )
        prompt = (
            f"【创作简报】\n{brief.model_dump_json(indent=2)}\n\n"
            f"【历史压缩记忆】\n{history}\n\n"
            f"【是否已有世界观】{'是' if has_existing_world else '否'}\n\n"
            f"【可用模板】\n{template_lines}\n\n"
            "请选择模板并给出 priority_focus 与 pacing。"
        )
        decision = self._complete_json(prompt, PlannerDecision, retries=1)

        template = decision.template
        if template not in TEMPLATE_STEPS:
            template = "continuation" if has_existing_world else "cold_start"

        focus = decision.priority_focus or brief.core_elements
        return build_plan(template, priority_focus=focus, pacing=decision.pacing)
