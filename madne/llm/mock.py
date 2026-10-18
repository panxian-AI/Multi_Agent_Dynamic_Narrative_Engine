"""Mock LLM：按 Agent 标识返回预设 JSON，支持脚本化响应序列。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from ..exceptions import MockResponseMissingError
from .base import parse_agent_id
from .fixtures import DEFAULT_FIXTURES, make_choice_cache


@dataclass(slots=True)
class MockCall:
    agent_id: str
    prompt: str
    system: str | None


class MockLLMClient:
    """从预设模板返回响应，供测试与 Demo 使用。

    每个 Agent 的响应是一个队列：
    - 队列长度 > 1 时，每次调用弹出队首；
    - 队列只剩 1 条时，持续复用最后一条。
    这样默认 Demo 可以无限调用，测试也能注入“失败一次再成功”的序列。
    """

    def __init__(self, fixtures: dict[str, Sequence[str]] | None = None) -> None:
        source = fixtures if fixtures is not None else DEFAULT_FIXTURES
        self._queues: dict[str, list[str]] = {
            agent_id: list(responses) for agent_id, responses in source.items()
        }
        self.calls: list[MockCall] = []

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> str:
        agent_id = parse_agent_id(system) or "unknown"
        self.calls.append(MockCall(agent_id=agent_id, prompt=prompt, system=system))

        if agent_id == "choice_cache":
            return self._dynamic_choice_cache(prompt)

        queue = self._queues.get(agent_id)
        if not queue:
            available = ", ".join(sorted(self._queues)) or "<empty>"
            raise MockResponseMissingError(
                f"MockLLMClient 没有 Agent '{agent_id}' 的预设响应；可用: {available}"
            )
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]

    def push_responses(
        self,
        agent_id: str,
        *responses: str,
        front: bool = False,
    ) -> None:
        """注入脚本化响应；front=True 时插入队首。"""
        existing = self._queues.get(agent_id, [])
        incoming = list(responses)
        self._queues[agent_id] = incoming + existing if front else existing + incoming

    def call_count(self, agent_id: str | None = None) -> int:
        if agent_id is None:
            return len(self.calls)
        return sum(1 for call in self.calls if call.agent_id == agent_id)

    def calls_for(self, agent_id: str) -> list[MockCall]:
        return [call for call in self.calls if call.agent_id == agent_id]

    @staticmethod
    def _dynamic_choice_cache(prompt: str) -> str:
        marker = "【当前分支（必须作为 path 根节点）】"
        start = prompt.find(marker)
        branches: list[str] = []
        descriptions: dict[str, str] = {}
        if start != -1:
            rest = prompt[start + len(marker) :]
            end = rest.find("【")
            section = rest[:end] if end != -1 else rest
            try:
                branch_data = json.loads(section.strip())
                for item in branch_data:
                    branch_id = item.get("branch_id")
                    if branch_id:
                        branches.append(branch_id)
                        descriptions[branch_id] = item.get("description", "")
            except (json.JSONDecodeError, TypeError, AttributeError):
                branches = list(
                    dict.fromkeys(
                        re.findall(r'"branch_id"\s*:\s*"([^"]+)"', section)
                    )
                )
        return json.dumps(
            make_choice_cache(branches, descriptions),
            ensure_ascii=False,
        )
