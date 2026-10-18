"""ImageGenClient 抽象与请求/结果契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class ImageRequest:
    """一次场景插图的生成请求。"""

    prompt: str
    output_path: Path
    negative_prompt: str = ""
    width: int = 1024
    height: int = 768
    seed: int = 0
    steps: int = 28
    guidance_scale: float = 5.0
    reference_images: list[Path] = field(default_factory=list)
    consistency_mode: str = "self_attn"
    style: str = ""


@dataclass(slots=True)
class ImageResult:
    """生成结果；``path`` 为实际落盘的 PNG。"""

    path: Path
    backend: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    consistency_mode: str = "self_attn"
    reference_images: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class ImageGenClient(Protocol):
    """所有图像后端的统一入口，业务代码不得直接请求具体服务。"""

    backend_name: str

    def generate(self, request: ImageRequest) -> ImageResult: ...

    def health(self) -> dict[str, Any]: ...
