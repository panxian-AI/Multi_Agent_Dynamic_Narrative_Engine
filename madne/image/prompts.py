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


# --------------------------------------------------------------------- SD 本地后端
# CLIP 文本编码器上下文只有 77 token：完整长 prompt 会把角色外观与风格标签
# 截掉，导致人物不一致。这里按优先级做 token 预算组装：
#   style / visual_prompt（画面主描述） > 角色外观 > 场景简述 > 质量收尾标签
# 中文长段（story moment / action / world）对 SD 类模型的 CLIP 贡献极小，
# 默认省略，只保留短场景关键词。

_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK 统一表意文字
    (0x3400, 0x4DBF),   # 扩展 A
    (0x3000, 0x303F),   # CJK 标点
    (0xFF00, 0xFFEF),   # 全角字符
)


def _estimate_clip_tokens(text: str) -> int:
    """粗略估算 CLIP token 数：CJK 每字 ≈2 token，其余 ≈每 4 字符 1 token。"""
    cjk = sum(
        1
        for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES)
    )
    other = len(text) - cjk
    return cjk * 2 + max(1, (other + 3) // 4)


def _clip(text: str, budget_tokens: int) -> str:
    """按 token 预算截断单段文本，尽量在词边界收尾。

    用二分找满足预算的最大前缀；预算过小（纯中文不足 2 字）时返回空串。
    """
    text = text.strip()
    if budget_tokens <= 0:
        return ""
    if _estimate_clip_tokens(text) <= budget_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _estimate_clip_tokens(text[:mid]) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo].rstrip("，。；、,. ;:-—")
    # 去掉残留的半截英文词与悬挂标点
    while cut and cut[-1].isascii() and cut[-1].isalnum():
        cut = cut[:-1]
    return cut.rstrip("，。；、,. ;:-")


def build_sd_prompt(
    scene: SceneOutline,
    npcs: list[NPCProfile],
    style: str = "",
    max_tokens: int = 72,
) -> str:
    """面向 SD / CLIP 77-token 上限的提示词组装。

    优先级：镜头描述 → 角色外观（最多 3 个）→ 场景关键词 → 风格标签 →
    质量收尾标签。中文叙事长段省略；预算不足时丢弃低优先级段落而不是
    无差别硬截断，保证最重要的信息总是保留。
    """
    budget = max(20, int(max_tokens))
    segments: list[str] = []

    def _try_add(text: str, cap: int, min_budget: int) -> None:
        nonlocal budget
        if not text or budget < min_budget:
            return
        seg = _clip(text, min(cap, budget - min_budget))
        tokens = _estimate_clip_tokens(seg)
        if tokens < 3 or tokens > budget:
            return
        segments.append(seg)
        budget -= tokens + 2  # 段间 ", " 连接开销

    _try_add(scene.visual_prompt.strip(), 30, 18)

    involved = select_involved_npcs(scene, npcs)[:3]
    for npc in involved:
        desc = npc.appearance.strip() or npc.role.strip()
        desc = f"{npc.name}，{desc}" if npc.name else desc
        _try_add(desc, 16, 10)

    _try_add(style.strip(), 12, 8)
    _try_add(scene.setting.strip(), 14, 8)
    _try_add("cinematic, highly detailed, no text", 10, 0)

    return ", ".join(segments)
