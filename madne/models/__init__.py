"""全系统统一数据契约（M0.2）。

所有 Agent 之间只通过这些 Pydantic 模型通信。
"""

from pydantic import BaseModel

from .base import MADNEModel
from .brief import CreationBrief
from .framework import Beat, Branch, StoryFramework
from .input import PlayerInput
from .memory import CompressedMemory, MemoryEvent
from .npc import NPCProfile
from .output import GameOutput
from .plan import PlannerDecision, TaskPlan, TaskStep
from .review import Issue, ReviewReport
from .scene import SceneOutline
from .script import DialogueLine, DialogueScript
from .world import WorldBible

__all__ = [
    "MADNEModel",
    "PlayerInput",
    "CreationBrief",
    "TaskStep",
    "TaskPlan",
    "PlannerDecision",
    "WorldBible",
    "Beat",
    "Branch",
    "StoryFramework",
    "SceneOutline",
    "DialogueLine",
    "DialogueScript",
    "NPCProfile",
    "Issue",
    "ReviewReport",
    "GameOutput",
    "MemoryEvent",
    "CompressedMemory",
    "MODEL_REGISTRY",
    "model_by_name",
]

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (
        PlayerInput,
        CreationBrief,
        TaskStep,
        TaskPlan,
        PlannerDecision,
        WorldBible,
        Beat,
        Branch,
        StoryFramework,
        SceneOutline,
        DialogueLine,
        DialogueScript,
        NPCProfile,
        Issue,
        ReviewReport,
        GameOutput,
        MemoryEvent,
        CompressedMemory,
    )
}


def model_by_name(name: str) -> type[BaseModel] | None:
    return MODEL_REGISTRY.get(name)
