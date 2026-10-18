from pathlib import Path

from madne.engine.context_store import ContextStore
from madne.llm import MockLLMClient
from madne.memory import MemorySystem
from madne.models import CompressedMemory, NPCProfile, WorldBible
from madne.npc import NPCSystem


def make_world() -> WorldBible:
    return WorldBible(
        era="2087",
        tech_level="神经接口",
        social_structure="企业城邦",
        geography="新港市",
        rules=["记忆可交易"],
        boundaries=["不存在超自然力量"],
    )


def test_npc_trait_changes_and_persistence(tmp_path: Path):
    storage = tmp_path / "npcs.json"
    system = NPCSystem(storage).load()
    system.initialize()

    before = system.get_profiles(["npc_kai"])[0]
    assert before.traits["loyalty"] == 85

    system.apply_player_choice("寻求盟友的帮助", ["npc_kai"])
    after = system.get_profiles(["npc_kai"])[0]
    assert after.traits["loyalty"] == 95
    assert after.traits["honesty"] == 65
    assert after.traits["selfishness"] == 40
    assert system.last_changes

    reloaded = NPCSystem(storage).load()
    persisted = reloaded.get_profiles(["npc_kai"])[0]
    assert persisted.traits == after.traits


def test_compressed_memory_merge_deduplicates(tmp_path: Path):
    memory = MemorySystem(
        tmp_path,
        MockLLMClient(),
        compress_every=2,
        session_id="session-x",
    )
    previous = CompressedMemory(
        session_id="session-x",
        player_preferences=["偏好调查"],
        unresolved_hooks=["旧伏笔"],
        relationship_changes=["与阿凯认识"],
        long_term_goals=["查清真相"],
    )
    fresh = CompressedMemory(
        session_id="",
        player_preferences=["偏好调查", "偏好盟友"],
        unresolved_hooks=["旧伏笔", "新伏笔"],
        relationship_changes=["与阿凯结盟"],
        long_term_goals=["查清真相"],
    )

    merged = memory.merge(previous, fresh)

    assert merged.session_id == "session-x"
    assert merged.player_preferences == ["偏好调查", "偏好盟友"]
    assert merged.unresolved_hooks == ["旧伏笔", "新伏笔"]
    assert merged.relationship_changes == ["与阿凯结盟", "与阿凯认识"]
    assert merged.long_term_goals == ["查清真相"]


def test_context_store_roundtrip_preserves_model_types(tmp_path: Path):
    context = ContextStore(session_id="session-x", cycle=2)
    world = make_world()
    profile = NPCProfile(
        npc_id="npc_kai",
        name="阿凯",
        role="掮客",
        traits={"loyalty": 90},
    )
    context.set("world", world)
    context.set("npcs", [profile])
    context.mark_completed("01_world")
    context.add_warning("示例告警")

    path = context.save(tmp_path / "cycles" / "cycle_002" / "context.json")
    loaded = ContextStore.load(path)

    assert isinstance(loaded.get("world"), WorldBible)
    assert loaded.get("world") == world
    assert isinstance(loaded.get("npcs")[0], NPCProfile)
    assert loaded.get("npcs")[0].traits == {"loyalty": 90}
    assert loaded.completed_steps == ["01_world"]
    assert loaded.warnings == ["示例告警"]
    assert loaded.cycle == 2
