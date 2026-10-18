"""零依赖 Mock 图像后端：生成可打开的占位 PNG，供开发/CI/无 GPU 环境使用。"""

from __future__ import annotations

import colorsys
import random
import struct
import zlib
from pathlib import Path
from typing import Any

from .base import ImageRequest, ImageResult

_MAX_SIDE = 512


def _chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)


def _hsv(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    red, green, blue = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return int(red * 255), int(green * 255), int(blue * 255)


def render_placeholder_png(width: int, height: int, seed: int) -> bytes:
    """按 seed 生成一张赛博朋克风格渐变占位图（纯标准库）。"""
    rng = random.Random(seed)
    hue = rng.random()
    top_color = _hsv(hue, 0.72, 0.10)
    horizon_color = _hsv(hue + 0.08, 0.78, 0.60)
    glow_color = _hsv(hue + 0.5, 0.45, 1.0)
    building_color = _hsv(hue + 0.02, 0.85, 0.13)
    window_color = _hsv(hue + 0.12, 0.35, 1.0)

    center_x = int(width * rng.uniform(0.25, 0.75))
    center_y = int(height * rng.uniform(0.22, 0.42))
    radius = max(24.0, max(width, height) * rng.uniform(0.22, 0.34))
    dx_sq = [((x - center_x) / radius) ** 2 for x in range(width)]

    column_tops = [height] * width
    column_colors: list[tuple[int, int, int]] = [building_color] * width
    x = 0
    while x < width:
        block_width = rng.randint(max(6, width // 28), max(10, width // 8))
        top = rng.randint(int(height * 0.52), int(height * 0.82))
        shade = _hsv(hue + 0.02, 0.85, rng.uniform(0.10, 0.20))
        for column in range(x, min(width, x + block_width)):
            column_tops[column] = top
            column_colors[column] = shade
        x += block_width + rng.randint(1, max(2, width // 45))

    raw = bytearray()
    for y in range(height):
        t = y / max(1, height - 1)
        base_r = int(top_color[0] + (horizon_color[0] - top_color[0]) * t)
        base_g = int(top_color[1] + (horizon_color[1] - top_color[1]) * t)
        base_b = int(top_color[2] + (horizon_color[2] - top_color[2]) * t)
        dy_sq = ((y - center_y) / radius) ** 2
        row = bytearray()
        row.append(0)  # PNG filter: None
        for px in range(width):
            glow = 1.0 - (dx_sq[px] + dy_sq)
            glow = glow * glow if glow > 0.0 else 0.0
            red = base_r + (glow_color[0] - base_r) * glow * 0.75
            green = base_g + (glow_color[1] - base_g) * glow * 0.75
            blue = base_b + (glow_color[2] - base_b) * glow * 0.75

            building_top = column_tops[px]
            if y >= building_top:
                shade_r, shade_g, shade_b = column_colors[px]
                mix = 0.82
                red = red * (1.0 - mix) + shade_r * mix
                green = green * (1.0 - mix) + shade_g * mix
                blue = blue * (1.0 - mix) + shade_b * mix
                # 稀疏的窗灯，避免每像素都做取模
                if (
                    y > building_top + 4
                    and y < height - 4
                    and (px % 11) in (0, 1)
                    and ((px // 11) * 13 + (y // 9) * 7 + seed) % 17 == 0
                ):
                    red, green, blue = window_color

            row.extend(
                (
                    max(0, min(255, int(red))),
                    max(0, min(255, int(green))),
                    max(0, min(255, int(blue))),
                )
            )
        raw.extend(row)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )


class MockImageGenerator:
    """生成占位插图；无 Key / 无 GPU / CI 环境下始终可用。"""

    backend_name = "mock"

    def __init__(self, fallback_reason: str | None = None) -> None:
        self.fallback_reason = fallback_reason

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "available": True,
            "detail": self.fallback_reason or "占位图后端（零依赖）",
        }

    def generate(self, request: ImageRequest) -> ImageResult:
        scale = min(1.0, _MAX_SIDE / max(request.width, request.height, 1))
        width = max(64, int(request.width * scale))
        height = max(64, int(request.height * scale))
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(
            render_placeholder_png(width, height, request.seed)
        )

        warnings = ["mock 后端仅生成占位图，未使用真实扩散模型"]
        if self.fallback_reason:
            warnings.insert(0, self.fallback_reason)
        if (width, height) != (request.width, request.height):
            warnings.append(
                f"占位图已按 {_MAX_SIDE}px 上限缩放为 {width}x{height}"
            )
        return ImageResult(
            path=request.output_path,
            backend=self.backend_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=width,
            height=height,
            consistency_mode=request.consistency_mode,
            reference_images=list(request.reference_images),
            warnings=warnings,
        )
