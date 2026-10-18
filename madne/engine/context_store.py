"""共享上下文存储：Agent 间唯一的数据交换介质，支持落盘与恢复。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..exceptions import EngineError
from ..models import model_by_name

MODEL_TAG = "__madne_model__"
DATA_TAG = "data"


def encode_value(value: Any) -> Any:
    """递归编码，保留 Pydantic 模型类型，供恢复时还原。"""
    if isinstance(value, BaseModel):
        return {
            MODEL_TAG: type(value).__name__,
            DATA_TAG: value.model_dump(mode="json"),
        }
    if isinstance(value, dict):
        return {key: encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_value(item) for item in value]
    return value


def decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if MODEL_TAG in value:
            type_name = value[MODEL_TAG]
            model_cls = model_by_name(type_name)
            if model_cls is None:
                raise EngineError(f"ContextStore 无法识别持久化模型类型: {type_name}")
            return model_cls.model_validate(value.get(DATA_TAG))
        return {key: decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


class ContextStore:
    """dict 风格的会话上下文，附带步骤完成标记与告警。"""

    FORMAT_VERSION = 1

    def __init__(
        self,
        session_id: str = "",
        cycle: int = 0,
        data: dict[str, Any] | None = None,
        completed_steps: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.cycle = cycle
        self._data: dict[str, Any] = dict(data or {})
        self.completed_steps: list[str] = list(completed_steps or [])
        self.warnings: list[str] = list(warnings or [])

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data

    def require(self, key: str) -> Any:
        if key not in self._data:
            raise EngineError(
                f"上下文缺少必需键 '{key}'（session={self.session_id}, cycle={self.cycle}）"
            )
        return self._data[key]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def keys(self) -> list[str]:
        return list(self._data)

    def mark_completed(self, step_id: str) -> None:
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)

    def is_completed(self, step_id: str) -> bool:
        return step_id in self.completed_steps

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def snapshot(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "session_id": self.session_id,
            "cycle": self.cycle,
            "completed_steps": list(self.completed_steps),
            "warnings": list(self.warnings),
            "data": {key: encode_value(value) for key, value in self._data.items()},
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ContextStore":
        source = Path(path)
        if not source.exists():
            raise EngineError(f"上下文文件不存在: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EngineError(f"上下文文件格式错误: {source}") from exc

        data = {
            key: decode_value(value)
            for key, value in payload.get("data", {}).items()
        }
        return cls(
            session_id=payload.get("session_id", ""),
            cycle=int(payload.get("cycle", 0)),
            data=data,
            completed_steps=payload.get("completed_steps", []),
            warnings=payload.get("warnings", []),
        )
