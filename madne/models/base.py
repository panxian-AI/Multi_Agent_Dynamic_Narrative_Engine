"""所有数据模型的公共基类。"""

from pydantic import BaseModel, ConfigDict


class MADNEModel(BaseModel):
    """忽略 LLM 可能附带的额外字段，字符串自动去除首尾空白。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
