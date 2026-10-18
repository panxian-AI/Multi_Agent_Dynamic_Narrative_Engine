"""跨 Agent 复用的提示词片段。"""

from __future__ import annotations

from ..models import ReviewReport, WorldBible


def world_redlines(world: WorldBible) -> str:
    """世界红线摘要：注入后续所有创作 Agent 的系统提示词。"""
    lines = ["【世界红线（不可违背）】"]
    lines.extend(f"- {boundary}" for boundary in world.boundaries)
    if world.rules:
        lines.append("【世界规则】")
        lines.extend(f"- {rule}" for rule in world.rules)
    return "\n".join(lines)


def format_review_feedback(report: ReviewReport) -> str:
    """把 Guardian 的审查结果转换为可回灌给生成模块的修订要求。"""
    lines = ["GuardianAI 判定本次产出未通过，请按以下要求重写："]
    for issue in report.issues:
        lines.append(f"- [{issue.category}/{issue.severity}] {issue.detail}")
    if report.suggested_fix:
        lines.append(f"建议修复方向：{report.suggested_fix}")
    lines.append("保留原有合格内容，只修正上述问题，重新输出完整 JSON。")
    return "\n".join(lines)


MEMORY_COMPRESSION_SYSTEM = """# AGENT_ID: memory_compress
你是记忆压缩 AI。请从给定的事件流中提炼长期有效的压缩记忆，供后续创作使用。

要求：
1. 只提炼，不创作新剧情。
2. 四个列表分别记录：玩家偏好、未解伏笔、人物关系变化、玩家长期目标。
3. 合并重复信息，保留最新状态；每条尽量简短。
4. 严格输出 CompressedMemory JSON。"""
