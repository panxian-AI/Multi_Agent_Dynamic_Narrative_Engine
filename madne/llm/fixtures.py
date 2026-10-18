"""Mock LLM 的预设响应（赛博朋克记忆买卖 Demo 场景）。"""

from __future__ import annotations

import json
from typing import Any


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


_EXAMPLE_BRIEF = {
    "genre": "赛博朋克侦探",
    "style": "黑色电影 / 冷硬派",
    "core_elements": ["记忆买卖", "侦探调查", "赛博朋克都市", "企业阴谋"],
    "tone": "冷峻、悬疑、带道德困境",
    "constraints": ["不存在超自然力量", "分支数量不超过 5 个", "保持侦探叙事视角"],
}

_PLANNER_DECISION = {
    "template": "cold_start",
    "priority_focus": ["记忆交易的道德困境", "侦探与记忆商人的博弈"],
    "pacing": "medium",
}

_PLANNER_CONTINUATION = {
    "template": "continuation",
    "priority_focus": ["兑现与阿凯的同盟", "追查苏离的交易对象"],
    "pacing": "medium",
}

_WORLD_BIBLE = {
    "era": "2087 年，新港市",
    "tech_level": "神经接口与记忆封装技术成熟，义体普及，但强人工智能被严格限制",
    "social_structure": "巨型企业掌控记忆交易网络，市政机构沦为附庸；黑市、记忆掮客与私家侦探构成灰色地带",
    "geography": "新港市分为上层企业穹顶区、中层商业区与底层锈带；记忆黑市藏于旧城数据墓园下方",
    "rules": [
        "记忆可被提取、封装、买卖，但需要神经接口且存在排斥风险",
        "被交易记忆会留下可追踪的神经签名",
        "企业拥有记忆专利，未经授权复制属于重罪",
    ],
    "boundaries": [
        "不存在超自然力量或魔法",
        "记忆不能被凭空创造，只能来自真实经历或人工合成",
        "主要角色不得无动机背叛既有关系",
        "死亡不可逆转，不存在复活技术",
    ],
}

_STORY_FRAMEWORK = {
    "main_plot": "私家侦探林默受神秘委托人雇佣，寻找失踪的记忆商人苏离；调查逐步揭开企业利用被盗记忆操控政商的阴谋。",
    "key_beats": [
        {
            "beat_id": "beat_01",
            "title": "雨夜委托",
            "summary": "林默在旧城咖啡馆接下寻找苏离的委托，发现委托人提供的记忆片段自相矛盾。",
            "order": 1,
            "status": "pending",
        },
        {
            "beat_id": "beat_02",
            "title": "记忆黑市",
            "summary": "林默潜入记忆黑市，从掮客阿凯处得知苏离最后一次交易涉及企业高管。",
            "order": 2,
            "status": "pending",
        },
        {
            "beat_id": "beat_03",
            "title": "被篡改的证人",
            "summary": "关键证人的记忆被改写，林默必须决定相信证据还是相信直觉。",
            "order": 3,
            "status": "pending",
        },
    ],
    "branches": [
        {
            "branch_id": "A",
            "description": "独自潜入记忆黑市，追查苏离最后的交易记录",
            "risk_level": "high",
            "requirements": ["需要伪造神经接口权限"],
        },
        {
            "branch_id": "B",
            "description": "寻求盟友阿凯的帮助，交换情报并结成临时同盟",
            "risk_level": "medium",
            "requirements": ["需要付出情报作为代价"],
        },
        {
            "branch_id": "C",
            "description": "正面接触企业高管，以调查委托为筹码试探对方",
            "risk_level": "high",
            "requirements": ["需要暴露侦探身份"],
        },
    ],
}

_SCENE_OUTLINE = {
    "scene_id": "scene_001",
    "summary": "林默在雨夜咖啡馆收到一段自相矛盾的记忆片段，决定前往记忆黑市寻找苏离的下落。",
    "setting": "新港市旧城，一间霓虹灯失修的咖啡馆；雨水沿玻璃幕墙流下，街对面的全息广告循环播放记忆银行的促销信息。",
    "actions": [
        "林默检查委托人留下的记忆芯片，记录神经签名编号",
        "他避开街口的巡逻无人机，穿过锈带进入数据墓园",
        "在地下入口，他用一枚旧义体零件换取通行口令",
    ],
    "events": [
        "记忆片段中出现了一个本不该存在的第二视角",
        "黑市掮客阿凯认出了苏离的神经签名，但拒绝免费提供信息",
        "远处传来企业安保无人机的扫描脉冲，交易被迫中断",
    ],
    "involved_npcs": ["npc_suli", "npc_kai"],
}

_DIALOGUE_SCRIPT = {
    "scene_id": "scene_001",
    "lines": [
        {
            "speaker": "林默",
            "text": "这段记忆有两个视角。委托人没告诉我，当时还有第二个人在场。",
            "inner_monologue": "要么委托人在撒谎，要么这段记忆被人剪过。",
            "emotion": "冷静而警觉",
        },
        {
            "speaker": "阿凯",
            "text": "神经签名不会说谎，但卖签名的人会。苏离最后出现在数据墓园，和一个穿企业制服的人在一起。",
            "inner_monologue": "这笔情报值多少钱，得看这个侦探有多急。",
            "emotion": "试探",
        },
        {
            "speaker": "林默",
            "text": "价格呢？",
            "inner_monologue": "先让他开价，才能判断他到底知道多少。",
            "emotion": "克制",
        },
        {
            "speaker": "阿凯",
            "text": "一枚未登记的记忆芯片，外加你欠我一次人情。别急着答应，安保无人机的扫描脉冲还有四十秒到。",
            "inner_monologue": "如果他答应得太快，说明苏离的事比我想的更麻烦。",
            "emotion": "急促",
        },
    ],
}

_REVIEW_PASS = {"passed": True, "issues": [], "suggested_fix": None}

_COMPRESSED_MEMORY = {
    "session_id": "",
    "player_preferences": ["偏好通过寻求盟友推进调查", "倾向谨慎试探而非正面冲突"],
    "unresolved_hooks": [
        "苏离最后一笔交易的对象是谁",
        "委托人的记忆片段为何存在第二视角",
    ],
    "relationship_changes": ["林默与掮客阿凯建立临时合作关系"],
    "long_term_goals": ["查清记忆黑市背后的企业操控网络"],
}

DEFAULT_FIXTURES: dict[str, list[str]] = {
    "intent": [_dumps(_EXAMPLE_BRIEF)],
    # 第一次选择 cold_start，之后持续选择 continuation（模拟已有世界观的续写）。
    "planner": [_dumps(_PLANNER_DECISION), _dumps(_PLANNER_CONTINUATION)],
    "world": [_dumps(_WORLD_BIBLE)],
    "framework": [_dumps(_STORY_FRAMEWORK)],
    "filler": [_dumps(_SCENE_OUTLINE)],
    "script": [_dumps(_DIALOGUE_SCRIPT)],
    "guardian_world": [_dumps(_REVIEW_PASS)],
    "guardian_framework": [_dumps(_REVIEW_PASS)],
    "guardian_scene": [_dumps(_REVIEW_PASS)],
    "memory_compress": [_dumps(_COMPRESSED_MEMORY)],
}
