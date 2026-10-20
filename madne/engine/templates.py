"""Agent 能力注册表与动态计划编译器。

流程：PlannerAI 只输出「需要哪些 Agent」的顺序建议，编译器负责：
1. 校验 Agent 是否合法；
2. 按 ``required_inputs`` 自动补齐上游生产步骤；
3. 保证交付必需步骤（framework / filler / script）存在；
4. 输出可调度、可回退的 ``TaskPlan``。

模板（cold_start / continuation）保留为降级路径：LLM 输出不可用时仍能产出计划。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import EngineError
from ..models import PlannedStep, PlannerDecision, TaskPlan, TaskStep
from ..models.plan import FallbackPolicy

# 由编排器直接注入、不需要 Agent 生产的上下文键。
SEED_KEYS: frozenset[str] = frozenset(
    {
        "brief",
        "player_input",
        "player_choice",
        "compressed_memory",
        "previous_framework",
        "previous_scene",
        "prefetched_choices",
        "active_choice_cache",
        "active_cached_branch_id",
        "last_npc_changes",
        "plan",
        "game_output",
        "illustration",
        "choice_cache",
    }
)

DELIVERY_AGENTS: tuple[str, ...] = ("framework", "filler", "script")


@dataclass(frozen=True, slots=True)
class AgentCapability:
    """单个 Agent 的调度契约。"""

    agent: str
    description: str
    output_key: str
    required_inputs: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    fallback: FallbackPolicy = "retry"
    phase: int = 0
    delivery: bool = False

    @property
    def input_keys(self) -> tuple[str, ...]:
        return (*self.required_inputs, *self.optional_inputs)


AGENT_CAPABILITIES: dict[str, AgentCapability] = {
    "world": AgentCapability(
        agent="world",
        description="生成或扩展世界观圣经（首次创作必需）",
        output_key="world",
        required_inputs=("brief",),
        fallback="retry",
        phase=10,
    ),
    "npc": AgentCapability(
        agent="npc",
        description="从世界观与故事框架中抽取角色池",
        output_key="npcs",
        required_inputs=("brief", "world", "framework"),
        fallback="retry",
        phase=30,
    ),
    "npc_update": AgentCapability(
        agent="npc_update",
        description="按玩家选择更新已有 NPC 的 traits（续写场景）",
        output_key="npcs",
        required_inputs=("npcs",),
        optional_inputs=("player_choice", "previous_scene"),
        fallback="skip",
        phase=20,
    ),
    "framework": AgentCapability(
        agent="framework",
        description="生成故事框架：主线、节拍与玩家分支",
        output_key="framework",
        required_inputs=("world", "player_input"),
        optional_inputs=(
            "compressed_memory",
            "previous_framework",
            "previous_scene",
            "prefetched_choices",
        ),
        fallback="retry",
        phase=40,
        delivery=True,
    ),
    "filler": AgentCapability(
        agent="filler",
        description="把框架节拍填充为单场景大纲",
        output_key="scene",
        required_inputs=("framework", "world", "npcs"),
        optional_inputs=("previous_scene",),
        fallback="retry",
        phase=50,
        delivery=True,
    ),
    "script": AgentCapability(
        agent="script",
        description="生成最终对白台本",
        output_key="script",
        required_inputs=("scene", "npcs"),
        optional_inputs=("world", "framework"),
        fallback="retry",
        phase=60,
        delivery=True,
    ),
}


def capability_for(agent: str) -> AgentCapability | None:
    return AGENT_CAPABILITIES.get(agent)


def available_agents() -> list[str]:
    return sorted(AGENT_CAPABILITIES)


def agent_for_output(key: str) -> str | None:
    """返回能生产指定上下文键的 Agent。"""
    for agent in ("world", "framework", "filler", "script", "npc", "npc_update"):
        capability = AGENT_CAPABILITIES[agent]
        if capability.output_key == key:
            return agent
    return None


def catalog_text() -> str:
    lines: list[str] = []
    for agent in available_agents():
        capability = AGENT_CAPABILITIES[agent]
        requires = "、".join(capability.required_inputs) or "无"
        optional = "、".join(capability.optional_inputs) or "无"
        lines.append(
            f"- {agent}: {capability.description}；产出 {capability.output_key}；"
            f"必需输入 {requires}；可选输入 {optional}"
        )
    return "\n".join(lines)


# 兼容旧路径的静态模板库（LLM 输出非法时降级使用）。
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
                "previous_scene",
                "prefetched_choices",
            ],
            "output_key": "framework",
            "fallback": "retry",
        },
        {
            "step_id": "03_filler",
            "agent": "filler",
            "input_keys": ["framework", "world", "npcs", "previous_scene"],
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


def make_step(
    agent: str,
    ordinal: int,
    *,
    reason: str = "",
    injected: bool = False,
) -> TaskStep:
    capability = AGENT_CAPABILITIES[agent]
    return TaskStep(
        step_id=f"{ordinal:02d}_{agent}",
        agent=agent,
        input_keys=list(capability.input_keys),
        output_key=capability.output_key,
        fallback=capability.fallback,
        required_inputs=list(capability.required_inputs),
        reason=reason,
        injected=injected,
    )


def build_plan(
    template: str,
    priority_focus: list[str] | None = None,
    pacing: str = "medium",
    *,
    rationale: str = "",
    max_steps: int = 8,
) -> TaskPlan:
    """模板降级路径：保持与旧版本一致的步骤链。"""
    if template not in TEMPLATE_STEPS:
        raise EngineError(
            f"未知任务模板 '{template}'；可用: {', '.join(sorted(TEMPLATE_STEPS))}"
        )
    steps = [
        TaskStep(
            **definition,
            required_inputs=list(
                AGENT_CAPABILITIES[definition["agent"]].required_inputs
            ),
        )
        for definition in TEMPLATE_STEPS[template]
    ]
    return TaskPlan(
        steps=steps,
        priority_focus=list(priority_focus or []),
        pacing=pacing,
        mode="template",
        rationale=rationale or f"模板降级：{template}",
        max_steps=max_steps,
    )


def compile_plan(
    decision: PlannerDecision,
    *,
    has_existing_world: bool = False,
    available_keys: set[str] | None = None,
    priority_focus: list[str] | None = None,
    pacing: str | None = None,
    max_steps: int = 8,
) -> TaskPlan:
    """把 LLM 的 Agent 选择编译成带依赖顺序的 TaskPlan。"""
    budget = max(3, max_steps)
    available = set(available_keys or ()) | SEED_KEYS
    requested = _normalize_requested(decision.steps, has_existing_world, available)

    if not requested:
        template = (
            decision.template
            or ("continuation" if has_existing_world else "cold_start")
        )
        return build_plan(
            template,
            priority_focus=priority_focus or decision.priority_focus,
            pacing=pacing or decision.pacing,
            rationale=decision.rationale,
            max_steps=budget,
        )

    delivery = list(DELIVERY_AGENTS)
    if not has_existing_world or "world" not in available:
        delivery.insert(0, "world")

    # LLM 请求的顺序优先，随后并上交付必需链；依赖排序会把生产者前移。
    ordering = _order_with_dependencies(
        [*requested, *delivery], available, has_existing_world, budget
    )
    if not set(delivery).issubset(set(ordering[:budget])):
        # 预算不足时优先保证可交付，再把可选步骤填进剩余名额。
        essential = _order_with_dependencies(
            delivery, available, has_existing_world, budget
        )
        extras = _order_with_dependencies(
            [agent for agent in requested if agent not in essential],
            available,
            has_existing_world,
            budget,
        )
        ordering = [*essential, *[a for a in extras if a not in essential]]

    reasons = {step.agent: step.reason for step in decision.steps}
    steps: list[TaskStep] = []
    for index, agent in enumerate(ordering[:budget], start=1):
        capability = AGENT_CAPABILITIES[agent]
        if agent in reasons and reasons[agent]:
            reason = reasons[agent]
        elif capability.delivery:
            reason = "交付必需步骤（编译器补齐）"
        else:
            reason = "依赖补齐（编译器自动插入）"
        steps.append(make_step(agent, index, reason=reason))

    return TaskPlan(
        steps=steps,
        priority_focus=list(priority_focus or decision.priority_focus),
        pacing=pacing or decision.pacing,
        mode="dynamic",
        rationale=decision.rationale
        or "LLM 动态选择 Agent 链："
        + " -> ".join(step.agent for step in steps),
        max_steps=budget,
    )


def _normalize_requested(
    steps: list[PlannedStep],
    has_existing_world: bool,
    available: set[str],
) -> list[str]:
    normalized: list[str] = []
    for step in steps:
        agent = step.agent.strip()
        if agent not in AGENT_CAPABILITIES:
            continue
        if agent == "npc" and has_existing_world:
            agent = "npc_update"
        if agent == "npc_update" and not has_existing_world:
            agent = "npc"
        # 已有世界观时不再重复生成，避免浪费预算与破坏设定连续性。
        if agent == "world" and "world" in available:
            continue
        if agent not in normalized:
            normalized.append(agent)
    return normalized


def _order_with_dependencies(
    agents: list[str],
    available: set[str],
    has_existing_world: bool,
    budget: int,
) -> list[str]:
    """深度优先拓扑排序：依赖的生产者一定排在消费者之前。"""
    ordered: list[str] = []

    def produced_by_selected(key: str) -> bool:
        # 只统计「已经排好序」的步骤；尚未访问的请求项由 visit 递归前移。
        return any(
            AGENT_CAPABILITIES[item].output_key == key for item in ordered
        )

    def visit(agent: str, stack: set[str]) -> None:
        if agent in ordered or agent in stack:
            return
        capability = AGENT_CAPABILITIES.get(agent)
        if capability is None:
            return
        stack.add(agent)
        for key in capability.required_inputs:
            if key in available or produced_by_selected(key):
                continue
            producer = agent_for_output(key)
            if producer is None or producer == agent:
                continue
            if producer == "npc_update" and not has_existing_world:
                producer = "npc"
            visit(producer, stack)
        stack.discard(agent)
        if agent not in ordered:
            ordered.append(agent)

    for agent in agents:
        if len(ordered) >= budget:
            break
        visit(agent, set())
    return ordered
