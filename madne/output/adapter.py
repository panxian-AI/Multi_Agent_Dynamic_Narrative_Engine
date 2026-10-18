"""M11 输出适配器：打包并校验游戏引擎 JSON。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..exceptions import EngineError
from ..models import (
    DialogueScript,
    GameOutput,
    Illustration,
    SceneOutline,
    StoryFramework,
    WorldBible,
)


class OutputAdapter:
    """只把通过 Guardian 审查的内容打包为 GameOutput。"""

    def build(
        self,
        *,
        session_id: str,
        scene: SceneOutline,
        script: DialogueScript,
        framework: StoryFramework,
        world: WorldBible,
        illustration: Illustration | None = None,
    ) -> GameOutput:
        choices = framework.branches[:5]
        if not choices:
            raise EngineError("StoryFramework 没有任何可选分支，无法生成 GameOutput")

        world_hash = hashlib.sha256(
            world.model_dump_json().encode("utf-8")
        ).hexdigest()
        output = GameOutput(
            scene=scene,
            script=script,
            choices=choices,
            world_hash=world_hash,
            session_id=session_id,
            illustration=illustration,
        )
        issues = self.validate_for_engine(output.to_engine_dict())
        if issues:
            raise EngineError("GameOutput 引擎校验失败: " + "；".join(issues))
        return output

    @staticmethod
    def validate_for_engine(payload: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        for key in ("scene", "script", "choices", "world_hash", "session_id"):
            if key not in payload:
                issues.append(f"缺少必备字段: {key}")
        if issues:
            return issues

        scene = payload.get("scene")
        script = payload.get("script")
        choices = payload.get("choices")
        illustration = payload.get("illustration")

        if illustration is not None:
            if not isinstance(illustration, dict):
                issues.append("illustration 结构非法")
            else:
                for key in ("scene_id", "backend", "prompt", "relative_path", "url"):
                    if not illustration.get(key):
                        issues.append(f"illustration.{key} 缺失")
                if (
                    isinstance(scene, dict)
                    and illustration.get("scene_id")
                    and illustration.get("scene_id") != scene.get("scene_id")
                ):
                    issues.append("illustration.scene_id 与 scene.scene_id 不一致")

        if not isinstance(scene, dict) or not scene.get("scene_id"):
            issues.append("scene.scene_id 缺失")
        if not isinstance(script, dict):
            issues.append("script 结构非法")
        else:
            lines = script.get("lines")
            if not isinstance(lines, list) or not lines:
                issues.append("script.lines 不能为空")
            else:
                for index, line in enumerate(lines):
                    if not isinstance(line, dict):
                        issues.append(f"script.lines[{index}] 结构非法")
                        continue
                    if not line.get("speaker"):
                        issues.append(f"script.lines[{index}].speaker 缺失")
                    if not line.get("text"):
                        issues.append(f"script.lines[{index}].text 缺失")
            if isinstance(scene, dict) and script.get("scene_id") != scene.get("scene_id"):
                issues.append("scene 与 script 的 scene_id 不一致")

        if not isinstance(choices, list):
            issues.append("choices 必须为数组")
        else:
            if len(choices) < 1:
                issues.append("choices 至少需要 1 个选项")
            if len(choices) > 5:
                issues.append("choices 数量不能超过 5")
            for index, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    issues.append(f"choices[{index}] 结构非法")
                    continue
                if not choice.get("id"):
                    issues.append(f"choices[{index}].id 缺失")
                if not choice.get("text"):
                    issues.append(f"choices[{index}].text 缺失")
        return issues

    @staticmethod
    def write(output: GameOutput, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(output.to_engine_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target
