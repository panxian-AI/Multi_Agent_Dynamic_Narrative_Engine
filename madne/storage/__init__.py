"""持久化层：统一使用 SQLite 存储会话状态。"""

from .database import SessionDatabase

__all__ = ["SessionDatabase"]
