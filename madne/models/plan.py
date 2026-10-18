from typing import Literal

from pydantic import Field, model_validator

from .base import MADNEModel

FallbackPolicy = Literal["retry", "skip", "degrade"]


class TaskStep(MADNEModel):
    """TaskPlan 中的单个可调度步骤。"""

    step_id: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    input_keys: list[str] = Field(default_factory=list)
    output_key: str = Field(min_length=1)
    fallback: FallbackPolicy = "retry"


class TaskPlan(MADNEModel):
    """PlannerAI 的最终产出：由静态模板实例化而来。"""

    steps: list[TaskStep] = Field(min_length=1)
    priority_focus: list[str] = Field(default_factory=list)
    pacing: str = "medium"

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "TaskPlan":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("TaskPlan.steps 中存在重复的 step_id")
        return self


class PlannerDecision(MADNEModel):
    """PlannerAI 的 LLM 原始决策：只选模板，不直接生成步骤。"""

    template: str = Field(min_length=1)
    priority_focus: list[str] = Field(default_factory=list)
    pacing: str = "medium"
