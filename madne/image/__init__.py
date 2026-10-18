"""图像生成子系统：统一客户端 + 多后端 + 人物一致性。"""

from .base import ImageGenClient, ImageRequest, ImageResult
from .factory import FallbackImageGenerator, create_image_client
from .prompts import build_negative_prompt, build_scene_prompt

__all__ = [
    "FallbackImageGenerator",
    "ImageGenClient",
    "ImageRequest",
    "ImageResult",
    "create_image_client",
    "build_scene_prompt",
    "build_negative_prompt",
]
