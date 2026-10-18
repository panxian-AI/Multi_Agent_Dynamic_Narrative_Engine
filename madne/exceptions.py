"""统一异常类型，便于上层按类别处理。"""


class MADNEError(Exception):
    """MADNE 基础异常。"""


class ConfigError(MADNEError):
    """配置缺失或非法。"""


class LLMError(MADNEError):
    """LLM 调用失败。"""


class MockResponseMissingError(LLMError):
    """MockLLMClient 未找到对应 Agent 的预设响应。"""


class AgentError(MADNEError):
    """Agent 执行失败。"""


class AgentOutputError(AgentError):
    """Agent 输出无法通过结构化校验。"""


class IntentParseError(AgentError):
    """意图解析失败（重试后仍不合法）。"""


class EngineError(MADNEError):
    """工作流引擎/上下文状态错误。"""
