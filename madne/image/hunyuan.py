"""Seedream / 豆包 / OpenAI 兼容图像生成后端。

默认按 ``POST {base_url}/images/generations`` 调用，支持返回 URL 或
base64 图片。火山方舟（Ark）使用
``https://ark.cn-beijing.volces.com/api/v3``，并额外发送
``sequential_image_generation`` / ``stream`` / ``watermark`` 字段。
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..exceptions import ImageGenError, ImageTimeoutError
from .base import ImageRequest, ImageResult

DEFAULT_BASE_URL = ""
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "doubao-seedream-4-0-250828"
ARK_BACKENDS = frozenset({"ark", "doubao", "volcengine"})
DEFAULT_MODEL = ARK_MODEL


class _Retryable(Exception):
    """网络抖动、限流或 5xx，允许退避重试。"""


class OpenAICompatImageGenerator:
    """OpenAI 兼容的图像生成后端（Seedream / 混元 / 第三方网关）。"""

    backend_name = "seedream"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
        max_retries: int = 2,
        size: str = "",
        backend_name: str = "seedream",
        watermark: bool = True,
        sequential_generation: str = "disabled",
        stream: bool = False,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        self.model = (model or DEFAULT_MODEL).strip()
        self.timeout = max(10, int(timeout))
        self.max_retries = max(0, int(max_retries))
        self.size = (size or "").strip()
        self.backend_name = (backend_name or "seedream").strip()
        self.watermark = bool(watermark)
        self.sequential_generation = (
            sequential_generation or "disabled"
        ).strip()
        self.stream = bool(stream)

    # ------------------------------------------------------------------ 兼容判定
    def _is_ark(self) -> bool:
        """火山方舟 / 豆包 Seedream 需要额外的请求字段。"""
        return (
            self.backend_name.lower() in ARK_BACKENDS
            or "ark.cn-beijing.volces.com" in self.base_url.lower()
            or self.model.lower().startswith("doubao-seedream")
        )

    # ------------------------------------------------------------------ 健康检查
    def health(self) -> dict[str, Any]:
        configured = bool(self.api_key and self.base_url)
        if not self.api_key:
            detail = "未配置生图 API Key"
        elif not self.base_url:
            detail = "未配置生图 Base URL"
        else:
            detail = f"{self.base_url} · {self.model}"
        return {
            "backend": self.backend_name,
            "available": configured,
            "detail": detail,
        }

    # ------------------------------------------------------------------ 生成入口
    def generate(self, request: ImageRequest) -> ImageResult:
        if not self.api_key:
            raise ImageGenError(
                "未配置生图 API Key；请在「API 配置」窗口填写"
            )
        if not self.base_url:
            raise ImageGenError(
                "未配置生图 Base URL；请填写服务商控制台提供的 OpenAI 兼容地址"
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._generate_once(request)
            except ImageGenError as exc:
                # 部分 OpenAI 兼容网关不接受 negative_prompt / response_format，
                # 400 时依次去掉这两个字段重试。
                if "HTTP 400" in str(exc):
                    fallbacks = (
                        {
                            "include_negative": False,
                            "include_response_format": True,
                        },
                        {
                            "include_negative": False,
                            "include_response_format": False,
                        },
                    )
                    for options in fallbacks:
                        try:
                            return self._generate_once(request, **options)
                        except ImageGenError:
                            continue
                raise
            except _Retryable as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(1 if attempt == 0 else 3)

        raise ImageGenError(
            f"Seedream 生图失败（已重试 {self.max_retries} 次）：{last_error}"
        )

    # ------------------------------------------------------------------ 内部实现
    def _generate_once(
        self,
        request: ImageRequest,
        include_negative: bool = True,
        include_response_format: bool = True,
    ) -> ImageResult:
        is_ark = self._is_ark()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "size": self.size
            or ("2K" if is_ark else f"{request.width}x{request.height}"),
        }
        if is_ark:
            # 对齐火山方舟 / 豆包 Seedream 的 REST 请求体。
            payload["sequential_image_generation"] = (
                self.sequential_generation
            )
            payload["stream"] = self.stream
            payload["watermark"] = self.watermark
            if include_response_format:
                payload["response_format"] = "url"
        else:
            payload["n"] = 1
            if include_negative and request.negative_prompt.strip():
                payload["negative_prompt"] = request.negative_prompt.strip()
            if include_response_format:
                payload["response_format"] = "url"

        data = self._post_json(self._endpoint(), payload)
        image_bytes = self._extract_image(data)
        self._save_png(image_bytes, request.output_path)
        size = _image_size(image_bytes) or (request.width, request.height)

        return ImageResult(
            path=request.output_path,
            backend=self.backend_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=size[0],
            height=size[1],
            consistency_mode=request.consistency_mode,
            reference_images=list(request.reference_images),
            warnings=[],
        )

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if not base:
            return ""
        lower = base.lower()
        if lower.endswith(
            ("/images/generations", "/image/generations", "/generations")
        ):
            return base
        if lower.endswith(("/v1", "/v1beta", "/api/v3")):
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

    def endpoint(self) -> str:
        """返回实际请求地址，供配置界面展示与排障。"""
        return self._endpoint()

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, image/*",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                raw = response.read()
                content_type = (
                    response.headers.get("Content-Type") or ""
                ).lower()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except OSError:
                pass
            hint = ""
            if exc.code == 404 and self._is_ark():
                hint = (
                    "；请确认 Base URL 为 "
                    "https://ark.cn-beijing.volces.com/api/v3，"
                    "且模型中已开通 doubao-seedream"
                )
            elif (
                exc.code == 404
                and "seedream" in self.model.lower()
                and "hunyuan.cloud.tencent.com" in url.lower()
            ):
                hint = (
                    "；当前仍是混元默认 Base URL，Seedream 模型需要填写"
                    "腾讯云/服务商控制台提供的 OpenAI 兼容地址"
                )
            if exc.code in (401, 403):
                raise ImageGenError(
                    f"生图鉴权失败（HTTP {exc.code}）[{url}]：{body[:300]}"
                ) from exc
            if exc.code == 429 or exc.code >= 500:
                raise _Retryable(
                    f"HTTP {exc.code} [{url}]: {body[:200]}"
                ) from exc
            raise ImageGenError(
                f"生图请求失败（HTTP {exc.code}）[{url}]："
                f"{body[:500]}{hint}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                raise ImageTimeoutError(
                    f"生图 API 超时（>{self.timeout}s，{url}）"
                ) from exc
            raise _Retryable(f"网络错误 [{url}]：{exc}") from exc

        if content_type.startswith("image/"):
            return {"_raw": raw}
        if not raw:
            raise ImageGenError("生图返回空响应")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            if _looks_like_image(raw):
                return {"_raw": raw}
            raise ImageGenError(f"生图返回非 JSON：{raw[:300]!r}")
        if isinstance(payload, list):
            return {"data": payload}
        return payload

    def _extract_image(self, data: dict[str, Any]) -> bytes:
        if data.get("_raw"):
            return data["_raw"]

        candidates: list[str] = []
        items = data.get("data")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                for key in ("url", "b64_json", "image"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())
                        break

        for key in ("ResultImage", "result", "images", "image", "output"):
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, list) and value:
                value = value[0]
            if isinstance(value, dict):
                value = (
                    value.get("url")
                    or value.get("b64_json")
                    or value.get("image")
                )
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

        if not candidates:
            raise ImageGenError(
                f"生图响应中没有图片：{json.dumps(data, ensure_ascii=False)[:300]}"
            )

        value = candidates[0]
        if value.startswith(("http://", "https://")):
            return self._download(value)
        if value.startswith("data:image"):
            value = value.split(",", 1)[-1]
        try:
            return base64.b64decode(value)
        except (ValueError, TypeError) as exc:
            raise ImageGenError(
                f"无法解析生图返回的 base64 图片：{exc}"
            ) from exc

    def _download(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                if data:
                    return data
                last_error = ImageGenError("下载结果图为空")
            except (urllib.error.URLError, OSError) as exc:
                if "timed out" in str(exc).lower():
                    raise ImageTimeoutError(
                        "下载生图结果超时，本回合不生成插图"
                    ) from exc
                last_error = exc
            if attempt < 2:
                time.sleep(1)
        raise ImageGenError(f"下载生图结果失败：{last_error}")

    @staticmethod
    def _save_png(data: bytes, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            output_path.write_bytes(data)
            return

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImageGenError(
                "需要 Pillow 将生图返回的图片转换为 PNG"
            ) from exc

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.convert("RGB").save(output_path, format="PNG")
        except Exception as exc:
            raise ImageGenError(f"无法转换生图返回图片：{exc}") from exc


def _looks_like_image(data: bytes) -> bool:
    return data.startswith(
        (
            b"\x89PNG\r\n\x1a\n",
            b"\xff\xd8\xff",
            b"GIF8",
            b"RIFF",
        )
    )


def _image_size(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return image.size
    except Exception:
        return None


# 兼容旧名称 / 旧配置
HunyuanImageGenerator = OpenAICompatImageGenerator
