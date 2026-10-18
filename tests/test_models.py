import hashlib

import pytest
from pydantic import ValidationError

from madne.models import (
    Beat,
    Branch,
    CachedChoice,
    ChoiceCache,
    DialogueLine,
    DialogueScript,
    GameOutput,
    NPCProfile,
    SceneOutline,
    StoryFramework,
    WorldBible,
)


def make_world() -> WorldBible:
    return WorldBible(
        era="2087",
        tech_level="神经接口",
        social_structure="企业城邦",
        geography="新港市",
        rules=["记忆可交易"],
        boundaries=["不存在超自然力量"],
    )


def make_branches(count: int = 3) -> list[Branch]:
    return [
        Branch(branch_id=chr(65 + index), description=f"选项{index}")
        for index in range(count)
    ]


def make_framework(count: int = 3) -> StoryFramework:
    return StoryFramework(
        main_plot="寻找失踪的记忆商人",
        key_beats=[Beat(beat_id="beat_01", title="委托", summary="接下委托")],
        branches=make_branches(count),
    )


def make_scene() -> SceneOutline:
    return SceneOutline(scene_id="scene_x", summary="场景摘要", setting="旧城咖啡馆")


def make_script() -> DialogueScript:
    return DialogueScript(
        scene_id="scene_x",
        lines=[DialogueLine(speaker="林默", text="这是台词。", emotion="冷静")],
    )


def test_npc_traits_are_clamped_to_0_100():
    profile = NPCProfile(
        npc_id="npc_test",
        name="测试角色",
        role="测试",
        traits={"honesty": 150, "loyalty": -10},
    )
    assert profile.traits == {"honesty": 100, "loyalty": 0}


def test_framework_requires_three_to_five_branches():
    with pytest.raises(ValidationError):
        make_framework(count=2)
    with pytest.raises(ValidationError):
        make_framework(count=6)


def test_game_output_engine_payload():
    world = make_world()
    output = GameOutput(
        scene=make_scene(),
        script=make_script(),
        choices=make_branches(),
        world_hash=hashlib.sha256(
            world.model_dump_json().encode("utf-8")
        ).hexdigest(),
        session_id="demo",
    )
    payload = output.to_engine_dict()

    assert payload["session_id"] == "demo"
    assert len(payload["world_hash"]) == 64
    assert payload["choices"] == [
        {"id": "A", "text": "选项0"},
        {"id": "B", "text": "选项1"},
        {"id": "C", "text": "选项2"},
    ]
    assert payload["scene"]["scene_id"] == payload["script"]["scene_id"]


def test_choice_cache_navigation_and_two_layers():
    cache = ChoiceCache(
        session_id="s",
        cycle=1,
        framework_hash="h",
        options=[
            CachedChoice(
                path=["B", "B1"],
                description="下一层选项",
                requirements=["需要情报"],
            ),
            CachedChoice(
                path=["B", "B1", "B1a"],
                description="第二层选项",
            ),
        ],
    )

    assert cache.next_choices("B")[0].path[-1] == "B1"
    assert cache.grandchildren("B", "B1")[0].path[-1] == "B1a"
    branches = cache.as_branches("B")
    assert branches[0].branch_id == "B1"
    assert branches[0].requirements == ["需要情报"]


def test_choice_cache_rejects_orphan_second_layer():
    with pytest.raises(ValidationError):
        ChoiceCache(
            options=[
                CachedChoice(
                    path=["B", "B1", "B1a"],
                    description="缺少父节点",
                )
            ]
        )


def test_game_output_rejects_scene_script_mismatch():
    with pytest.raises(ValidationError):
        GameOutput(
            scene=make_scene(),
            script=DialogueScript(
                scene_id="another_scene",
                lines=[DialogueLine(speaker="A", text="B")],
            ),
            choices=make_branches(),
            world_hash="a" * 64,
            session_id="demo",
        )
