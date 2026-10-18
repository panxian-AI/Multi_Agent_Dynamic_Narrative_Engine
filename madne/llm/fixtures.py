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

_STORY_FRAMEWORK_CONTINUATION = {
    "main_plot": "林默与掮客阿凯结成临时同盟，利用阿凯掌握的黑市渠道追查苏离最后的交易对象，并逐步逼近操控记忆买卖的企业网络。",
    "key_beats": [
        {
            "beat_id": "beat_01",
            "title": "雨夜委托",
            "summary": "林默接下寻找苏离的委托，并确认委托人的记忆片段存在第二视角。",
            "order": 1,
            "status": "completed",
        },
        {
            "beat_id": "beat_02",
            "title": "盟友的代价",
            "summary": "林默选择向盟友阿凯寻求帮助，以一份未登记的记忆芯片换取黑市交易记录。",
            "order": 2,
            "status": "active",
        },
        {
            "beat_id": "beat_03",
            "title": "数据墓园追踪",
            "summary": "两人潜入数据墓园，找到苏离最后的交易终端，却发现记录已被远程抹除。",
            "order": 3,
            "status": "pending",
        },
    ],
    "branches": [
        {
            "branch_id": "B1",
            "description": "与盟友阿凯共享线索，继续追查被抹除的交易记录",
            "risk_level": "medium",
            "requirements": ["兑现与阿凯的情报交换"],
        },
        {
            "branch_id": "B2",
            "description": "独自潜入数据墓园深处，寻找被隐藏的备份",
            "risk_level": "high",
            "requirements": ["需要伪造维护权限"],
        },
        {
            "branch_id": "B3",
            "description": "通过阿凯联系记忆黑市的中间人，换取企业交易名单",
            "risk_level": "medium",
            "requirements": ["需要支付额外记忆券"],
        },
    ],
}

_CAST_ROSTER = {
    "characters": [
        {
            "npc_id": "npc_linmo",
            "name": "林默",
            "role": "私家侦探（玩家角色）",
            "is_major": True,
            "traits": {
                "honesty": 75,
                "selfishness": 40,
                "loyalty": 60,
                "curiosity": 85,
            },
            "appearance": "Chinese male private detective in his early 30s, short black hair, tired eyes, worn charcoal trench coat over a high-collar shirt, cybernetic left eye with a faint amber glow",
        },
        {
            "npc_id": "npc_suli",
            "name": "苏离",
            "role": "失踪的记忆商人",
            "is_major": True,
            "traits": {
                "honesty": 35,
                "selfishness": 80,
                "loyalty": 30,
                "curiosity": 65,
            },
            "appearance": "elegant Chinese woman in her 30s, silver bob haircut, sharp cheekbones, black high-collar memory-broker suit, neon-blue data earrings",
        },
        {
            "npc_id": "npc_kai",
            "name": "阿凯",
            "role": "记忆黑市掮客 / 潜在盟友",
            "is_major": True,
            "traits": {
                "honesty": 60,
                "selfishness": 45,
                "loyalty": 85,
                "curiosity": 50,
            },
            "appearance": "streetwise Chinese man in his late 20s, undercut hair with shaved sides, patched dark leather jacket, glowing memory-chip pendant around his neck",
        },
        {
            "npc_id": "npc_vera",
            "name": "维拉",
            "role": "企业合规官（背景角色）",
            "is_major": False,
            "traits": {
                "honesty": 50,
                "selfishness": 65,
                "loyalty": 40,
                "curiosity": 55,
            },
            "appearance": "corporate compliance officer, sharp gray tailored business suit, slicked-back black hair, cold expression, silver corporate ID pin",
        },
    ]
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
    "visual_prompt": "rain-soaked cyberpunk cafe at night, neon signs reflected in wet glass, a tired detective in a charcoal trench coat examining a glowing memory chip at a corner table",
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

_SCENE_OUTLINE_CONTINUATION = {
    "scene_id": "scene_002",
    "summary": "林默与阿凯在记忆黑市的地下仓库汇合，用一枚未登记芯片换取苏离的交易记录，却发现记录被人远程抹除。",
    "setting": "新港市旧城数据墓园下方，一间由废弃服务器机房改造的地下仓库；成排的记忆存储柜闪着暗红色指示灯，空气里弥漫着冷却液的刺鼻气味。",
    "actions": [
        "林默把未登记的记忆芯片放在交易台上，阿凯检查芯片的神经签名",
        "阿凯调出苏离最后一次交易的终端日志，逐条比对时间戳",
        "仓库角落的旧终端突然自动重启，日志被远程清空",
        "林默拔下终端存储模块，交给阿凯做离线镜像",
    ],
    "events": [
        "阿凯确认芯片来源干净，同意交换苏离的交易记录",
        "日志显示苏离最后接触的买家使用企业级加密通道",
        "远程抹除指令来自数据墓园上层的中继节点",
        "两人决定在天亮前潜入中继节点寻找备份",
    ],
    "involved_npcs": ["npc_kai"],
    "visual_prompt": "underground server-room memory market, rows of red-lit memory storage cabinets, two men exchanging a data chip across a steel table, cold cyan and red lighting",
}

_DIALOGUE_SCRIPT_CONTINUATION = {
    "scene_id": "scene_002",
    "lines": [
        {
            "speaker": "阿凯",
            "text": "芯片没问题，签名是干净的。苏离最后一笔交易就在这台终端上，但有人比我们快了一步。",
            "inner_monologue": "日志被清得这么干净，说明对方知道我们会来。",
            "emotion": "压低声音",
        },
        {
            "speaker": "林默",
            "text": "远程抹除走的是上层中继节点。备份还在吗？",
            "inner_monologue": "只要镜像还在，抹除指令就抹不掉物理残留。",
            "emotion": "冷静",
        },
        {
            "speaker": "阿凯",
            "text": "存储模块里有离线镜像，但取出来会触发机房的警报。你有十分钟。",
            "inner_monologue": "这次人情欠大了，得让他记住。",
            "emotion": "急促",
        },
        {
            "speaker": "林默",
            "text": "十分钟够了。你盯着门口，我拆模块。",
            "inner_monologue": "阿凯的忠诚值不值得信，得看他在警报响起后会不会留下。",
            "emotion": "果断",
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

_CACHE_ACTIONS = (
    "追查对应分支指向的记忆交易中间人，并承担暴露行踪的风险",
    "与相关黑市掮客交换情报，确认对方的要价与底线",
    "潜入相关中继节点，寻找被抹除记录的离线备份",
)


def make_choice_cache(
    branches: list[str],
    descriptions: dict[str, str] | None = None,
) -> dict:
    branch_descriptions = descriptions or {}
    options = []
    risks = ("medium", "high", "low")
    for root in branches:
        context = branch_descriptions.get(root, "")
        # 避免多轮预取时不断嵌套“基于…”前缀
        prefix = (
            f"承接「{context[:28]}」，"
            if context and "基于「" not in context and "承接「" not in context
            else ""
        )
        for index, action in enumerate(_CACHE_ACTIONS, 1):
            next_id = f"{root}{index}"
            options.append(
                {
                    "path": [root, next_id],
                    "description": f"{prefix}{action}（分支 {next_id}）",
                    "risk_level": risks[(index - 1) % len(risks)],
                    "requirements": [f"需要为分支 {next_id} 付出相应代价"],
                }
            )
            for child_index in range(1, 4):
                child_id = f"{next_id}{chr(96 + child_index)}"
                options.append(
                    {
                        "path": [root, next_id, child_id],
                        "description": (
                            f"在 {next_id} 之后继续深入：锁定新的证人或备份，"
                            f"并处理暴露风险（分支 {child_id}）"
                        ),
                        "risk_level": "medium",
                        "requirements": [f"需要先完成 {next_id}"],
                    }
                )
    return {
        "session_id": "",
        "cycle": 0,
        "framework_hash": "",
        "options": options,
    }


_CHOICE_CACHE_COLD = make_choice_cache(["A", "B", "C"])
_CHOICE_CACHE_CONTINUATION = make_choice_cache(["B1", "B2", "B3"])

DEFAULT_FIXTURES: dict[str, list[str]] = {
    "intent": [_dumps(_EXAMPLE_BRIEF)],
    # 第一次选择 cold_start，之后持续选择 continuation（模拟已有世界观的续写）。
    "planner": [_dumps(_PLANNER_DECISION), _dumps(_PLANNER_CONTINUATION)],
    "world": [_dumps(_WORLD_BIBLE)],
    "framework": [_dumps(_STORY_FRAMEWORK), _dumps(_STORY_FRAMEWORK_CONTINUATION)],
    "npc": [_dumps(_CAST_ROSTER)],
    "filler": [_dumps(_SCENE_OUTLINE), _dumps(_SCENE_OUTLINE_CONTINUATION)],
    "script": [_dumps(_DIALOGUE_SCRIPT), _dumps(_DIALOGUE_SCRIPT_CONTINUATION)],
    "choice_cache": [
        _dumps(_CHOICE_CACHE_COLD),
        _dumps(_CHOICE_CACHE_CONTINUATION),
    ],
    "guardian_world": [_dumps(_REVIEW_PASS)],
    "guardian_framework": [_dumps(_REVIEW_PASS)],
    "guardian_scene": [_dumps(_REVIEW_PASS)],
    "memory_compress": [_dumps(_COMPRESSED_MEMORY)],
}
