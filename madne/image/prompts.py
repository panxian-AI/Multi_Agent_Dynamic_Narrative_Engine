"""从叙事数据结构构造稳定的文生图提示词。"""

from __future__ import annotations

from ..models import NPCProfile, SceneOutline, WorldBible

DEFAULT_NEGATIVE_PROMPT = (
    "lowres, blurry, bad anatomy, bad hands, extra fingers, extra limbs, "
    "deformed, disfigured, watermark, text, signature, speech bubble, "
    "comic panel borders, collage, multiple views, duplicate character, "
    "inconsistent costume, nsfw"
)


def build_negative_prompt(extra: str = "") -> str:
    parts = [DEFAULT_NEGATIVE_PROMPT]
    if extra.strip():
        parts.append(extra.strip())
    return ", ".join(parts)


def select_involved_npcs(
    scene: SceneOutline,
    npcs: list[NPCProfile],
) -> list[NPCProfile]:
    if not scene.involved_npcs:
        return [npc for npc in npcs if npc.is_major][:3]
    involved = set(scene.involved_npcs)
    return [npc for npc in npcs if npc.npc_id in involved]


def build_scene_prompt(
    scene: SceneOutline,
    world: WorldBible | None,
    npcs: list[NPCProfile],
    style: str = "",
) -> str:
    """把场景、动作、世界与角色外观拼装为单条提示词。

    中文叙事文本保留原样（部分本地模型可直接理解），同时附加英文风格标签
    与角色外观描述，提升 SDXL / ComfyUI 工作流的人物一致性。
    """
    parts: list[str] = []
    if style.strip():
        parts.append(style.strip())
    if scene.visual_prompt.strip():
        parts.append(scene.visual_prompt.strip())
    parts.append(f"Scene: {scene.setting.strip()}")
    parts.append(f"Story moment: {scene.summary.strip()}")
    if scene.actions:
        parts.append("Action: " + "; ".join(scene.actions[:4]))
    if world is not None:
        parts.append(f"World: {world.era}, {world.tech_level}")

    involved = select_involved_npcs(scene, npcs)
    if involved:
        character_parts = []
        for npc in involved:
            appearance = npc.appearance.strip() or npc.role.strip()
            character_parts.append(f"{npc.name} ({appearance})")
        parts.append("Characters: " + "; ".join(character_parts))

    parts.append(
        "Single wide cinematic shot, coherent character identities, "
        "consistent costume and facial features, no text in image"
    )
    return ", ".join(part.strip(" ,") for part in parts if part.strip())
