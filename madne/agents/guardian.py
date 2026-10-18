"""M8 监管 AI：分维度审查，汇总为 ReviewReport。"""

from __future__ import annotations

from ..models import (
    CompressedMemory,
    DialogueScript,
    NPCProfile,
    ReviewReport,
    SceneOutline,
    StoryFramework,
    WorldBible,
)
from .base import BaseAgent
from .prompts import world_redlines

WORLD_SYSTEM = """你是世界观审查 AI。检查 WorldBible 的内部一致性：
- rules 之间是否互相矛盾
- boundaries 是否可检验、是否与 rules 冲突
- 科技、社会结构、地理是否自洽

严重度策略：
- 只有会导致设定自相矛盾或直接违反红线的硬伤才标记 severity=high 并 passed=false。
- 可补充、可优化但不会让设定失效的问题标记 low/medium，可作为建议记录，不阻断交付。
- issues 使用 category=logic，并给出具体 suggested_fix。"""

FRAMEWORK_SYSTEM = """你是情节审查 AI。检查 StoryFramework：
- 是否违反 WorldBible 的红线与规则
- 是否衔接压缩记忆中的未解伏笔与人物关系
- 节拍是否可执行、分支是否有限且清晰

严重度策略：
- 只有直接违反红线、主线断裂或关键节拍不可执行才标记 severity=high 并 passed=false。
- 细节补充、程序说明、伏笔优化标记 low/medium，可给建议但不阻断交付。
- 问题使用 category=logic 或 coherence，并给出具体 suggested_fix。"""

SCENE_SYSTEM = """你是质量审查 AI。每次调用只审查一个指定维度：
- logic：事件因果、时间线、是否违反世界规则
- coherence：场景与主线/当前节拍的衔接、与既有记忆是否一致
- literary：语言质量、画面感、节奏，是否存在空泛或陈词滥调
- character：角色言行是否符合其 traits 权重与既有关系

只针对当前维度给出结论。只有在会导致读者明显出戏、角色严重失真、逻辑硬伤或违反世界红线时，
才标记 severity=high 并 passed=false；文风润色、细节补充等建议标记 low/medium，不阻断交付。
建议修复必须可执行，不要泛泛而谈。"""

DIMENSION_RUBRICS = {
    "logic": "逻辑一致性",
    "coherence": "叙事连贯性",
    "literary": "文学质量",
    "character": "角色一致性",
}


class GuardianAI(BaseAgent):
    agent_id = "guardian_scene"

    def review_world(self, world: WorldBible) -> ReviewReport:
        prompt = (
            f"【待审 WorldBible】\n{world.model_dump_json(indent=2)}\n\n"
            "请检查内部逻辑一致性并输出 ReviewReport。"
        )
        report = self._complete_json(
            prompt,
            ReviewReport,
            agent_id="guardian_world",
            body=WORLD_SYSTEM,
            retries=0,
        )
        return self._apply_severity_policy(report)

    def review_framework(
        self,
        framework: StoryFramework,
        world: WorldBible,
        compressed_memory: CompressedMemory | None = None,
    ) -> ReviewReport:
        history = (
            compressed_memory.model_dump_json(indent=2)
            if compressed_memory is not None
            else "无历史压缩记忆"
        )
        prompt = (
            f"【待审 StoryFramework】\n{framework.model_dump_json(indent=2)}\n\n"
            f"【WorldBible】\n{world.model_dump_json(indent=2)}\n\n"
            f"【压缩记忆】\n{history}\n\n"
            "请检查边界冲突、记忆衔接与节拍可执行性，输出 ReviewReport。"
        )
        report = self._complete_json(
            prompt,
            ReviewReport,
            agent_id="guardian_framework",
            body=FRAMEWORK_SYSTEM,
            extra_system=world_redlines(world),
            retries=0,
        )
        return self._apply_severity_policy(report)

    def review_scene(
        self,
        scene: SceneOutline,
        script: DialogueScript,
        npcs: list[NPCProfile],
        world: WorldBible | None = None,
    ) -> ReviewReport:
        """四类问题分别用独立 checklist 审查，再汇总，降低漏检率。"""
        profiles = [npc.model_dump(mode="json") for npc in npcs]
        reports: list[ReviewReport] = []
        for dimension, label in DIMENSION_RUBRICS.items():
            prompt = (
                f"【审查维度】{dimension}（{label}）\n\n"
                f"【SceneOutline】\n{scene.model_dump_json(indent=2)}\n\n"
                f"【DialogueScript】\n{script.model_dump_json(indent=2)}\n\n"
                f"【NPC 档案与 traits】\n{profiles}\n\n"
            )
            if world is not None:
                prompt += f"【WorldBible】\n{world.model_dump_json(indent=2)}\n\n"
            prompt += "请只针对当前维度审查并输出 ReviewReport。"
            reports.append(
                self._complete_json(
                    prompt,
                    ReviewReport,
                    agent_id="guardian_scene",
                    body=SCENE_SYSTEM,
                    retries=0,
                )
            )
        return self._apply_severity_policy(self._merge_reports(reports))

    @staticmethod
    def _apply_severity_policy(report: ReviewReport) -> ReviewReport:
        """低/中严重度问题不阻断交付；只有 high 级硬伤触发重写。"""
        if report.passed or not report.issues:
            return report
        if any(issue.severity == "high" for issue in report.issues):
            return report
        return report.model_copy(update={"passed": True})

    @staticmethod
    def _merge_reports(reports: list[ReviewReport]) -> ReviewReport:
        issues = [issue for report in reports for issue in report.issues]
        fixes = [
            report.suggested_fix
            for report in reports
            if report.suggested_fix
        ]
        return ReviewReport(
            passed=all(report.passed for report in reports),
            issues=issues,
            suggested_fix="；".join(dict.fromkeys(fixes)) or None,
        )
