"""图像客户端工厂：按 IMG_BACKEND 选择后端，并在失败时降级 Mock。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings
from ..exceptions import ImageTimeoutError
from .base import ImageGenClient, ImageRequest, ImageResult
from .comfyui import ComfyUIImageGenerator
from .diffusers import DiffusersImageGenerator
from .downloads import find_local_checkpoint
from .hunyuan import ARK_BASE_URL, ARK_MODEL, OpenAICompatImageGenerator
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
        except ImageTimeoutError:
            # 用户设置的单张时限已到：本回合直接不出图，不返回占位图。
            raise
        except Exception as exc:  # 其他生图失败可降级占位图
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
                timeout=settings.img_timeout,
            )
        )

    if backend == "diffusers":
        checkpoint = settings.img_model_path
        if checkpoint is not None and not Path(checkpoint).is_file():
            checkpoint = None
        if checkpoint is None:
            checkpoint = find_local_checkpoint(
                settings.img_model, settings.model_cache_dir
            )
        return FallbackImageGenerator(
            DiffusersImageGenerator(
                model=settings.img_model,
                cache_dir=settings.model_cache_dir,
                allow_cpu=settings.img_allow_cpu,
                ip_adapter_dir=settings.img_ip_adapter_dir,
                timeout=settings.img_timeout,
                hf_endpoint=settings.hf_endpoint,
                local_checkpoint=checkpoint,
            )
        )

    if backend in {
        "seedream",
        "hunyuan",
        "openai",
        "openai_compat",
        "ark",
        "doubao",
        "volcengine",
    }:
        is_ark = backend in {"ark", "doubao", "volcengine"}
        base_url = settings.img_base_url.strip()
        if is_ark and not base_url:
            base_url = ARK_BASE_URL
        model = settings.img_api_model.strip() or ARK_MODEL
        size = settings.img_api_size.strip()
        if is_ark and not size:
            size = "2K"
        return FallbackImageGenerator(
            OpenAICompatImageGenerator(
                api_key=settings.img_api_key or "",
                base_url=base_url,
                model=model,
                timeout=settings.img_timeout,
                max_retries=settings.img_api_retries,
                size=size,
                backend_name=(
                    "ark"
                    if is_ark
                    else ("openai" if backend == "openai_compat" else backend)
                ),
                watermark=settings.img_api_watermark,
                sequential_generation=settings.img_api_sequential,
                stream=settings.img_api_stream,
            )
        )

    if backend not in {"mock", ""}:
        return MockImageGenerator(
            fallback_reason=(
                f"未知 IMG_BACKEND='{settings.img_backend}'，已回退 mock。"
                "可用值：mock / seedream / ark / doubao / hunyuan / "
                "comfyui / diffusers"
            )
        )
    return MockImageGenerator()
