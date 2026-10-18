"""M9 NPC 系统：档案持久化 + 规则驱动的 traits 演变。"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import (
    CompressedMemory,
    CreationBrief,
    NPCProfile,
    WorldBible,
)

PLAYER_NPC_ID = "npc_linmo"

# 预定义事件 -> 权重增量映射表；复杂事件后续可由 LLM 判定（见 README 扩展点）。
EVENT_RULES: tuple[tuple[tuple[str, ...], dict[str, int]], ...] = (
    (
        ("盟友", "合作", "结盟", "帮助", "信任"),
        {"loyalty": 10, "honesty": 5, "selfishness": -5},
    ),
    (
        ("威胁", "审讯", "欺骗", "背叛", "出卖"),
        {"loyalty": -15, "honesty": -5, "selfishness": 10},
    ),
    (
        ("救助", "保护", "救援", "救下"),
        {"loyalty": 15, "selfishness": -10},
    ),
)

DEFAULT_ROSTER: tuple[NPCProfile, ...] = (
    NPCProfile(
        npc_id=PLAYER_NPC_ID,
        name="林默",
        role="私家侦探（玩家角色）",
        is_major=True,
        traits={"honesty": 75, "selfishness": 40, "loyalty": 60, "curiosity": 85},
        appearance=(
            "Chinese male private detective in his early 30s, short black hair, "
            "tired eyes, worn charcoal trench coat, cybernetic left eye"
        ),
    ),
    NPCProfile(
        npc_id="npc_suli",
        name="苏离",
        role="失踪的记忆商人",
        is_major=True,
        traits={"honesty": 35, "selfishness": 80, "loyalty": 30, "curiosity": 65},
        appearance=(
            "elegant Chinese woman in her 30s, silver bob haircut, "
            "black high-collar memory-broker suit, neon-blue data earrings"
        ),
    ),
    NPCProfile(
        npc_id="npc_kai",
        name="阿凯",
        role="记忆黑市掮客 / 潜在盟友",
        is_major=True,
        traits={"honesty": 60, "selfishness": 45, "loyalty": 85, "curiosity": 50},
        appearance=(
            "streetwise Chinese man in his late 20s, undercut hair, "
            "patched dark leather jacket, glowing memory-chip pendant"
        ),
    ),
    NPCProfile(
        npc_id="npc_vera",
        name="维拉",
        role="企业合规官（背景角色）",
        is_major=False,
        traits={"honesty": 50, "selfishness": 65, "loyalty": 40, "curiosity": 55},
        appearance=(
            "corporate compliance officer, sharp gray tailored business suit, "
            "slicked-back black hair, silver corporate ID pin"
        ),
    ),
)


class NPCSystem:
    """主要角色池持久化；规则优先，LLM 辅助留作扩展点。"""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._profiles: dict[str, NPCProfile] = {}
        self.last_changes: list[str] = []

    def load(self) -> "NPCSystem":
        if self.storage_path is None or not self.storage_path.exists():
            return self
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            profiles = [NPCProfile.model_validate(item) for item in payload]
        except (json.JSONDecodeError, ValueError, TypeError):
            return self
        self._profiles = {profile.npc_id: profile for profile in profiles}
        return self

    def save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [profile.model_dump(mode="json") for profile in self._profiles.values()]
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def initialize(
        self,
        brief: CreationBrief | None = None,
        world: WorldBible | None = None,
    ) -> list[NPCProfile]:
        """首次进入会话时实例化角色池；已有档案则直接返回。"""
        if not self._profiles:
            self._profiles = {profile.npc_id: profile for profile in DEFAULT_ROSTER}
            self.save()
        return self.all_profiles()

    def set_profiles(self, profiles: list[NPCProfile]) -> list[NPCProfile]:
        """用角色抽取 AI 的产出替换/初始化角色池并持久化。

        LLM 在后续回合可能漏掉 appearance / reference_image；若新值为空，
        继承旧档案，避免生图失去跨回合人物一致性。
        """
        if not profiles:
            return self.all_profiles()
        merged: list[NPCProfile] = []
        for profile in profiles:
            previous = self._profiles.get(profile.npc_id)
            if previous is not None:
                updates: dict[str, str] = {}
                if not profile.appearance and previous.appearance:
                    updates["appearance"] = previous.appearance
                if not profile.reference_image and previous.reference_image:
                    updates["reference_image"] = previous.reference_image
                if updates:
                    profile = profile.model_copy(update=updates)
            merged.append(profile)
        self._profiles = {profile.npc_id: profile for profile in merged}
        self.save()
        return self.all_profiles()

    def all_profiles(self) -> list[NPCProfile]:
        return list(self._profiles.values())

    def get_profiles(self, npc_ids: list[str] | None = None) -> list[NPCProfile]:
        if npc_ids is None:
            return self.all_profiles()
        return [
            self._profiles[npc_id]
            for npc_id in npc_ids
            if npc_id in self._profiles
        ]

    def update_traits(self, npc_id: str, deltas: dict[str, int]) -> NPCProfile:
        if npc_id not in self._profiles:
            raise KeyError(f"未知 NPC: {npc_id}")
        profile = self._profiles[npc_id]
        traits = dict(profile.traits)
        for trait, delta in deltas.items():
            traits[trait] = max(0, min(100, traits.get(trait, 50) + int(delta)))
        updated = profile.model_copy(update={"traits": traits})
        self._profiles[npc_id] = updated
        return updated

    def apply_player_choice(
        self,
        choice: str,
        involved_npcs: list[str] | None = None,
    ) -> list[NPCProfile]:
        """按规则表更新相关 NPC 权重，并返回最新角色池。"""
        self.last_changes = []
        deltas = self._match_rule(choice)
        if not deltas:
            return self.all_profiles()

        target_ids = involved_npcs or [
            profile.npc_id
            for profile in self._profiles.values()
            if profile.is_major and profile.npc_id != PLAYER_NPC_ID
        ]
        for npc_id in target_ids:
            if npc_id == PLAYER_NPC_ID or npc_id not in self._profiles:
                continue
            updated = self.update_traits(npc_id, deltas)
            detail = ", ".join(f"{key}{value:+d}" for key, value in deltas.items())
            self.last_changes.append(f"{updated.name}: {detail}")

        self.save()
        return self.all_profiles()

    @staticmethod
    def _match_rule(choice: str) -> dict[str, int] | None:
        for keywords, deltas in EVENT_RULES:
            if any(keyword in choice for keyword in keywords):
                return deltas
        return None

    def to_compressed_hint(self) -> CompressedMemory | None:
        """预留：将关系变化写入压缩记忆的辅助入口。"""
        return None
