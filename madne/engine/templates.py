"""静态步骤模板库。

PlannerAI 只负责选择/裁剪模板，不凭空生成全部步骤，保证引擎可调度、可回退。
"""

from __future__ import annotations

from ..exceptions import EngineError
from ..models import TaskPlan, TaskStep

TEMPLATE_STEPS: dict[str, list[dict]] = {
    "cold_start": [
        {
            "step_id": "01_world",
            "agent": "world",
            "input_keys": ["brief"],
            "output_key": "world",
            "fallback": "retry",
        },
        {
            "step_id": "02_framework",
            "agent": "framework",
            "input_keys": ["world", "compressed_memory", "player_input"],
            "output_key": "framework",
            "fallback": "retry",
        },
        {
            "step_id": "03_npc",
            "agent": "npc",
            "input_keys": ["brief", "world"],
            "output_key": "npcs",
            "fallback": "retry",
        },
        {
            "step_id": "04_filler",
            "agent": "filler",
            "input_keys": ["framework", "world", "npcs"],
            "output_key": "scene",
            "fallback": "retry",
        },
        {
            "step_id": "05_script",
            "agent": "script",
            "input_keys": ["scene", "npcs", "world"],
            "output_key": "script",
            "fallback": "retry",
        },
    ],
    "continuation": [
        {
            "step_id": "01_npc_update",
            "agent": "npc_update",
            "input_keys": ["npcs", "player_choice"],
            "output_key": "npcs",
            "fallback": "skip",
        },
        {
            "step_id": "02_framework",
            "agent": "framework",
            "input_keys": [
                "world",
                "compressed_memory",
                "player_input",
                "previous_framework",
            ],
            "output_key": "framework",
            "fallback": "retry",
        },
        {
            "step_id": "03_filler",
            "agent": "filler",
            "input_keys": ["framework", "world", "npcs"],
            "output_key": "scene",
            "fallback": "retry",
        },
        {
            "step_id": "04_script",
            "agent": "script",
            "input_keys": ["scene", "npcs", "world"],
            "output_key": "script",
            "fallback": "retry",
        },
    ],
}


def available_templates() -> dict[str, list[str]]:
    return {
        name: [step["agent"] for step in steps]
        for name, steps in TEMPLATE_STEPS.items()
    }


def build_plan(
    template: str,
    priority_focus: list[str] | None = None,
    pacing: str = "medium",
) -> TaskPlan:
    if template not in TEMPLATE_STEPS:
        raise EngineError(
            f"未知任务模板 '{template}'；可用: {', '.join(sorted(TEMPLATE_STEPS))}"
        )
    steps = [TaskStep(**definition) for definition in TEMPLATE_STEPS[template]]
    return TaskPlan(
        steps=steps,
        priority_focus=list(priority_focus or []),
        pacing=pacing,
    )
