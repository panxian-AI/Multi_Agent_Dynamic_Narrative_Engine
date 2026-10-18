from pydantic import Field

from .base import MADNEModel


class PlayerInput(MADNEModel):
    """玩家的一次输入。"""

    text: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
