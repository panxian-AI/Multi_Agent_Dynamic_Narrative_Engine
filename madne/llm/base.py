"""LLMClient 抽象（M0.1）。"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """所有 LLM 调用的统一入口，便于切换模型或注入 Mock。"""

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> str: ...


def extract_json(text: str) -> str:
    """从可能包含 Markdown 代码块或解释文字的响应中提取 JSON 文本。"""
    if not text or not text.strip():
        raise ValueError("LLM 返回为空")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)

    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        raise ValueError("LLM 响应中未找到 JSON 对象或数组")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("LLM 响应中的 JSON 不完整")
    return text[start : end + 1]


def parse_agent_id(system: str | None) -> str | None:
    """从系统提示词中的 # AGENT_ID: xxx 标记解析 Agent 标识。"""
    if not system:
        return None
    match = re.search(r"#\s*AGENT_ID:\s*([A-Za-z0-9_]+)", system)
    return match.group(1) if match else None
