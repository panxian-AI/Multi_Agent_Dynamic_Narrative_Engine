"""OpenAI 兼容 Chat Completions 客户端（无第三方 HTTP 依赖）。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..exceptions import ConfigError, LLMError


class OpenAICompatClient:
    """读取 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 的标准实现。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ConfigError("使用真实 LLM 时必须提供 LLM_API_KEY")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        if json_schema is not None:
            # 兼容多数 OpenAI 兼容端点的 JSON 模式；schema 约束已写入 prompt。
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise LLMError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM 连接失败: {exc.reason}") from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 响应结构异常: {raw[:500]}") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM 返回了空内容")
        return content
