"""运行期动态链控制。

计划不是一次性固定的：执行每一步之前，如果它的必需输入在当前上下文中缺失，
PlanRuntime 会自动把生产该输入的步骤插到当前步骤之前（或把计划中排在后面的
生产步骤前移），从而实现「按实际状态动态生长」的步骤链。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import TaskPlan, TaskStep
from .templates import agent_for_output, capability_for, make_step


@dataclass(slots=True)
class PlanMutation:
    action: str
    agent: str
    step_id: str
    reason: str


class PlanRuntime:
    """可变计划与依赖修复器。"""

    def __init__(
        self,
        plan: TaskPlan,
        *,
        repair_limit: int = 1,
        max_steps: int | None = None,
    ) -> None:
        self.initial_plan = plan
        self.steps: list[TaskStep] = list(plan.steps)
        self.max_steps = max(
            len(self.steps),
            max_steps or plan.max_steps or len(self.steps) + 4,
        )
        self.repair_limit = max(0, repair_limit)
        self.mutations: list[PlanMutation] = []
        self._cursor = 0
        self._repairs: dict[str, int] = {}
        self._injected = 0

    # ------------------------------------------------------------------ 调度
    @property
    def cursor(self) -> int:
        return self._cursor

    def next_step(self, ctx) -> TaskStep | None:
        """返回下一个可执行步骤；必要时先插入/前移依赖生产步骤。"""
        while self._cursor < len(self.steps):
            step = self.steps[self._cursor]
            if ctx.is_completed(step.step_id):
                self._cursor += 1
                continue
            if self._repair_dependencies(ctx, step):
                continue
            return step
        return None

    def complete(self, step: TaskStep) -> None:
        for index in range(self._cursor, len(self.steps)):
            if self.steps[index].step_id == step.step_id:
                self._cursor = index + 1
                self._sync()
                return

    def append_agent(self, agent: str, *, reason: str) -> TaskStep:
        """收尾阶段补齐交付所需步骤。"""
        step = make_step(agent, len(self.steps) + 1, reason=reason)
        step = step.model_copy(
            update={
                "step_id": self._unique_id(step.step_id),
                "injected": True,
            }
        )
        self.steps.append(step)
        self._injected += 1
        self.mutations.append(
            PlanMutation(
                action="append",
                agent=agent,
                step_id=step.step_id,
                reason=reason,
            )
        )
        self._sync()
        return step

    # ------------------------------------------------------------------ 依赖修复
    def _repair_dependencies(self, ctx, step: TaskStep) -> bool:
        capability = capability_for(step.agent)
        required = list(step.required_inputs)
        if not required and capability is not None:
            required = list(capability.required_inputs)

        for key in required:
            if ctx.has(key):
                continue
            producer = agent_for_output(key)
            if producer is None or producer == step.agent:
                continue
            if producer == "npc_update" and not ctx.has("npcs"):
                producer = "npc"

            later = self._find_pending_producer(ctx, producer)
            if later is not None:
                self._move_before_cursor(later, step, key)
                return True
            if self._producer_pending(ctx, producer):
                continue
            if self._repairs.get(producer, 0) >= self.repair_limit:
                continue
            if len(self.steps) >= self.max_steps:
                continue

            new_step = make_step(
                producer,
                len(self.steps) + 1,
                reason=f"运行期补齐：{step.agent} 缺少输入 {key}",
                injected=True,
            )
            new_step = new_step.model_copy(
                update={"step_id": self._unique_id(f"repair_{producer}")}
            )
            self.steps.insert(self._cursor, new_step)
            self._repairs[producer] = self._repairs.get(producer, 0) + 1
            self._injected += 1
            self.mutations.append(
                PlanMutation(
                    action="insert",
                    agent=producer,
                    step_id=new_step.step_id,
                    reason=f"{step.agent} 缺少必需输入 {key}",
                )
            )
            self._sync()
            return True
        return False

    def _find_pending_producer(self, ctx, producer: str) -> int | None:
        for index in range(self._cursor + 1, len(self.steps)):
            candidate = self.steps[index]
            if candidate.agent != producer:
                continue
            if ctx.is_completed(candidate.step_id):
                continue
            return index
        return None

    def _producer_pending(self, ctx, producer: str) -> bool:
        for index in range(self._cursor, len(self.steps)):
            candidate = self.steps[index]
            if candidate.agent == producer and not ctx.is_completed(
                candidate.step_id
            ):
                return True
        return False

    def _move_before_cursor(
        self,
        index: int,
        consumer: TaskStep,
        key: str,
    ) -> None:
        step = self.steps.pop(index)
        self.steps.insert(self._cursor, step)
        self.mutations.append(
            PlanMutation(
                action="reorder",
                agent=step.agent,
                step_id=step.step_id,
                reason=f"{consumer.agent} 需要 {key}，生产步骤已前移",
            )
        )
        self._sync()

    # ------------------------------------------------------------------ 工具
    def _unique_id(self, base: str) -> str:
        existing = {step.step_id for step in self.steps}
        candidate = f"{base}_{self._injected + 1:02d}"
        suffix = 1
        while candidate in existing:
            suffix += 1
            candidate = f"{base}_{self._injected + 1:02d}_{suffix}"
        return candidate

    def _sync(self) -> None:
        self.initial_plan.steps = list(self.steps)

    def mutation_summary(self) -> str:
        if not self.mutations:
            return "计划在运行期未发生变更"
        return "；".join(
            f"{mutation.action} {mutation.agent}({mutation.reason})"
            for mutation in self.mutations
        )
