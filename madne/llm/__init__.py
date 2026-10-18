"""LLM 客户端工厂。"""

from ..config import Settings
from ..exceptions import ConfigError
from .base import LLMClient, extract_json, parse_agent_id
from .mock import MockLLMClient
from .openai_compat import OpenAICompatClient

__all__ = [
    "LLMClient",
    "MockLLMClient",
    "OpenAICompatClient",
    "create_llm_client",
    "extract_json",
    "parse_agent_id",
]


def create_llm_client(settings: Settings, mock: bool | None = None) -> LLMClient:
    use_mock = settings.use_mock if mock is None else mock
    if use_mock:
        return MockLLMClient()
    if not settings.llm_api_key:
        raise ConfigError("未设置 LLM_API_KEY；可使用 --mock 或设置环境变量")
    return OpenAICompatClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )
