"""全系统统一数据契约（M0.2）。

所有 Agent 之间只通过这些 Pydantic 模型通信。
"""

from pydantic import BaseModel

from .base import MADNEModel
from .brief import CreationBrief
from .cache import CachedChoice, ChoiceCache
from .framework import Beat, Branch, StoryFramework
from .image import Illustration, ImageCharacter, ImageMetrics
from .input import PlayerInput
from .memory import CompressedMemory, MemoryEvent
from .npc import CastRoster, NPCProfile
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
    "CachedChoice",
    "ChoiceCache",
    "SceneOutline",
    "DialogueLine",
    "DialogueScript",
    "NPCProfile",
    "CastRoster",
    "Issue",
    "ReviewReport",
    "GameOutput",
    "Illustration",
    "ImageCharacter",
    "ImageMetrics",
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
        CachedChoice,
        ChoiceCache,
        Branch,
        StoryFramework,
        SceneOutline,
        DialogueLine,
        DialogueScript,
        CastRoster,
        NPCProfile,
        Illustration,
        ImageCharacter,
        ImageMetrics,
        Issue,
        ReviewReport,
        GameOutput,
        MemoryEvent,
        CompressedMemory,
    )
}


def model_by_name(name: str) -> type[BaseModel] | None:
    return MODEL_REGISTRY.get(name)
