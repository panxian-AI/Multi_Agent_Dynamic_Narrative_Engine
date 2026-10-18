"""Agent 基类：统一系统提示词、JSON 解析与一次纠错重试。"""

from __future__ import annotations

import json
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ValidationError

from ..exceptions import AgentOutputError
from ..llm.base import LLMClient, extract_json

ModelT = TypeVar("ModelT", bound=BaseModel)


class BaseAgent:
    """所有 Agent 只通过 LLMClient 与 Pydantic 契约交互。"""

    agent_id: ClassVar[str]
    SYSTEM: ClassVar[str] = ""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def _system(
        self,
        agent_id: str | None = None,
        body: str | None = None,
        extra: str | None = None,
    ) -> str:
        parts = [
            f"# AGENT_ID: {agent_id or self.agent_id}",
            body if body is not None else self.SYSTEM,
        ]
        if extra:
            parts.append(extra)
        parts.append("只输出合法 JSON，不要输出 Markdown 代码块、注释或任何额外说明。")
        return "\n\n".join(part for part in parts if part)

    def _complete_json(
        self,
        prompt: str,
        model: type[ModelT],
        *,
        agent_id: str | None = None,
        body: str | None = None,
        extra_system: str | None = None,
        feedback: str | None = None,
        retries: int = 1,
    ) -> ModelT:
        schema_json = json.dumps(model.model_json_schema(), ensure_ascii=False)
        base_prompt = f"{prompt}\n\n【输出 JSON Schema】\n{schema_json}"
        if feedback:
            base_prompt += f"\n\n【修订要求】\n{feedback}"

        attempt_prompt = base_prompt
        last_error: Exception | None = None

        for _ in range(retries + 1):
            raw = self.llm.complete(
                attempt_prompt,
                system=self._system(agent_id=agent_id, body=body, extra=extra_system),
                json_schema=model.model_json_schema(),
            )
            try:
                payload = json.loads(extract_json(raw))
                return model.model_validate(payload)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                attempt_prompt = (
                    base_prompt
                    + f"\n\n【解析失败】\n上一次输出未通过校验：{exc}\n"
                    "请修正问题，重新输出完整 JSON。"
                )

        raise AgentOutputError(
            f"{agent_id or self.agent_id} 输出无法通过 {model.__name__} 校验: {last_error}"
        )
