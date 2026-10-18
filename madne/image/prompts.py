"""从叙事数据结构构造稳定的文生图提示词。"""

from __future__ import annotations

from ..models import NPCProfile, SceneOutline, WorldBible

DEFAULT_NEGATIVE_PROMPT = (
    "lowres, blurry, bad anatomy, bad hands, extra fingers, extra limbs, "
    "deformed, disfigured, watermark, text, signature, speech bubble, "
    "comic panel borders, collage, multiple views, duplicate character, "
    "inconsistent costume, nsfw"
)


def build_negative_prompt(extra: str = "", max_chars: int | None = None) -> str:
    parts = [DEFAULT_NEGATIVE_PROMPT]
    if extra.strip():
        parts.append(extra.strip())
    text = ", ".join(parts)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip(" ,")
    return text


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
    max_chars: int | None = None,
) -> str:
    """把场景、动作、世界与角色外观拼装为单条提示词。

    中文叙事文本保留原样（部分本地模型可直接理解），同时附加英文风格标签
    与角色外观描述，提升 SDXL / ComfyUI 工作流的人物一致性。

    ``max_chars`` 用于混元轻量版等有提示词长度上限的接口：优先保留
    场景、人物与情节，再截断到指定字符数。
    """
    involved = select_involved_npcs(scene, npcs)
    character_parts = [
        f"{npc.name} ({npc.appearance.strip() or npc.role.strip()})"
        for npc in involved
    ]

    if max_chars is not None:
        compact_parts = [
            f"场景：{scene.setting.strip()}",
            (
                "人物：" + "；".join(character_parts)
                if character_parts
                else ""
            ),
            f"情节：{scene.summary.strip()}",
            style.strip()[:80],
            scene.visual_prompt.strip(),
            "高清，细节丰富，电影感，画面干净，不含文字",
        ]
        prompt = "；".join(part for part in compact_parts if part)
        return prompt[:max_chars].rstrip("；, ")

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
    if character_parts:
        parts.append("Characters: " + "; ".join(character_parts))

    parts.append(
        "Single wide cinematic shot, coherent character identities, "
        "consistent costume and facial features, no text in image"
    )
    return ", ".join(part.strip(" ,") for part in parts if part.strip())
