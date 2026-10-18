"""图像客户端工厂：按 IMG_BACKEND 选择后端，并在失败时降级 Mock。"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from .base import ImageGenClient, ImageRequest, ImageResult
from .comfyui import ComfyUIImageGenerator
from .diffusers import DiffusersImageGenerator
from .hunyuan import OpenAICompatImageGenerator
from .mock import MockImageGenerator


class FallbackImageGenerator:
    """主后端失败时自动用 Mock 出占位图，保证文本回合不被中断。"""

    def __init__(
        self,
        primary: ImageGenClient,
        fallback: ImageGenClient | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or MockImageGenerator()

    @property
    def backend_name(self) -> str:
        return self.primary.backend_name

    def health(self) -> dict[str, Any]:
        info = dict(self.primary.health())
        info.setdefault("fallback", self.fallback.backend_name)
        return info

    def generate(self, request: ImageRequest) -> ImageResult:
        try:
            return self.primary.generate(request)
        except Exception as exc:  # 生图失败不得中断文本回合
            result = self.fallback.generate(request)
            result.warnings.insert(
                0,
                f"{self.primary.backend_name} 生成失败，已自动降级 mock：{exc}",
            )
            return result


def create_image_client(settings: Settings) -> ImageGenClient:
    """根据配置创建图像后端；未知配置安全回退到 Mock。"""
    backend = (settings.img_backend or "mock").strip().lower()

    if backend == "comfyui":
        return FallbackImageGenerator(
            ComfyUIImageGenerator(
                base_url=settings.comfyui_url,
                checkpoint=settings.comfyui_checkpoint,
                workflow_path=settings.comfyui_workflow,
                timeout=settings.comfyui_timeout,
            )
        )

    if backend == "diffusers":
        return FallbackImageGenerator(
            DiffusersImageGenerator(
                model=settings.img_model,
                cache_dir=settings.model_cache_dir,
                allow_cpu=settings.img_allow_cpu,
                ip_adapter_dir=settings.img_ip_adapter_dir,
            )
        )

    if backend in {"seedream", "hunyuan", "openai", "openai_compat"}:
        return FallbackImageGenerator(
            OpenAICompatImageGenerator(
                api_key=settings.img_api_key or "",
                base_url=settings.img_base_url,
                model=settings.img_api_model,
                timeout=settings.img_api_timeout,
                max_retries=settings.img_api_retries,
                size=settings.img_api_size,
                backend_name=(
                    "openai" if backend == "openai_compat" else backend
                ),
            )
        )

    if backend not in {"mock", ""}:
        return MockImageGenerator(
            fallback_reason=(
                f"未知 IMG_BACKEND='{settings.img_backend}'，已回退 mock。"
                "可用值：mock / seedream / hunyuan / comfyui / diffusers"
            )
        )
    return MockImageGenerator()
