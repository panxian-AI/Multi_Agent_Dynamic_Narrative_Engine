from typing import Literal

from pydantic import Field, model_validator

from .base import MADNEModel

FallbackPolicy = Literal["retry", "skip", "degrade"]
PlanMode = Literal["dynamic", "template"]


class TaskStep(MADNEModel):
    """TaskPlan 中的单个可调度步骤。"""

    step_id: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    input_keys: list[str] = Field(default_factory=list)
    output_key: str = Field(min_length=1)
    fallback: FallbackPolicy = "retry"
    # 必须已存在于上下文的键；缺失时运行期动态链会自动补齐生产步骤。
    required_inputs: list[str] = Field(default_factory=list)
    # 规划/修复原因，便于观察动态链的决策依据。
    reason: str = ""
    injected: bool = False


class TaskPlan(MADNEModel):
    """PlannerAI 的最终产出：动态编译或模板降级得到的可执行步骤链。"""

    steps: list[TaskStep] = Field(min_length=1)
    priority_focus: list[str] = Field(default_factory=list)
    pacing: str = "medium"
    mode: PlanMode = "template"
    rationale: str = ""
    max_steps: int = 8

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "TaskPlan":
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("TaskPlan.steps 中存在重复的 step_id")
        return self

    @property
    def is_dynamic(self) -> bool:
        return self.mode == "dynamic"

    def agent_sequence(self) -> list[str]:
        return [step.agent for step in self.steps]


class PlannedStep(MADNEModel):
    """LLM 动态规划时请求的一个 Agent 步骤（不含具体键位，由引擎编译）。"""

    agent: str = Field(min_length=1)
    reason: str = ""


class PlannerDecision(MADNEModel):
    """PlannerAI 的 LLM 原始决策。

    ``steps`` 为空时退回 ``template``（兼容旧提示词与降级路径）。
    """

    steps: list[PlannedStep] = Field(default_factory=list)
    template: str | None = None
    priority_focus: list[str] = Field(default_factory=list)
    pacing: str = "medium"
    rationale: str = ""
