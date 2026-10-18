"""diffusers 本地后端：SDXL 基座 + 可选 IP-Adapter 参考图身份注入。

依赖为可选安装（``pip install torch diffusers transformers accelerate``），
未安装或无 GPU 时由工厂自动降级为 Mock，不影响文本管线。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ..exceptions import ImageGenError
from .base import ImageRequest, ImageResult


class DiffusersImageGenerator:
    """本地 GPU 推理；模型权重懒加载到 ``MODEL_CACHE_DIR``。"""

    backend_name = "diffusers"

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir: str | Path = "models",
        allow_cpu: bool = False,
        ip_adapter_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.allow_cpu = allow_cpu
        self.ip_adapter_dir = (
            Path(ip_adapter_dir) if ip_adapter_dir is not None else None
        )
        self._pipeline: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._torch: Any = None
        self._ip_adapter_loaded = False

    # ------------------------------------------------------------------ 健康检查
    def health(self) -> dict[str, Any]:
        missing = [
            name
            for name in ("torch", "diffusers")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return {
                "backend": self.backend_name,
                "available": False,
                "detail": "缺少依赖: " + ", ".join(missing),
            }

        gpu: str | None = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu = f"cuda:{torch.cuda.get_device_name(0)}"
            elif (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                gpu = "mps"
        except Exception as exc:  # 初始化失败也视为不可用
            return {
                "backend": self.backend_name,
                "available": False,
                "detail": f"torch 初始化失败: {exc}",
            }

        if gpu is None and not self.allow_cpu:
            return {
                "backend": self.backend_name,
                "available": False,
                "detail": "未检测到 GPU；设置 IMG_ALLOW_CPU=1 可使用 CPU（非常慢）",
            }
        return {
            "backend": self.backend_name,
            "available": True,
            "detail": f"{gpu or 'cpu'} · {self.model}",
        }

    # ------------------------------------------------------------------ 生成入口
    def generate(self, request: ImageRequest) -> ImageResult:
        pipe, device, torch_module = self._pipeline_or_raise()
        warnings: list[str] = []

        # CUDA 需要 CUDA generator；MPS/CPU 使用 CPU generator。
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "generator": torch_module.Generator(
                device=generator_device
            ).manual_seed(request.seed),
        }

        if request.reference_images:
            if request.consistency_mode == "id_injection":
                self._apply_reference_injection(
                    pipe, request.reference_images, kwargs, warnings
                )
            else:
                warnings.append(
                    "self_attn 模式通过固定 seed + 稳定外观提示保持一致性，"
                    "参考图未注入模型；需要身份注入请设置 IMG_CONSISTENCY_MODE=id_injection"
                )

        try:
            images = pipe(**kwargs).images
        except Exception as exc:
            raise ImageGenError(f"diffusers 生成失败: {exc}") from exc

        if not images:
            raise ImageGenError("diffusers 未返回图片")
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(request.output_path)

        return ImageResult(
            path=request.output_path,
            backend=self.backend_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=request.width,
            height=request.height,
            consistency_mode=request.consistency_mode,
            reference_images=list(request.reference_images),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ 内部实现
    def _pipeline_or_raise(self) -> tuple[Any, Any, Any]:
        if self._pipeline is not None:
            return self._pipeline, self._device, self._torch

        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise ImageGenError(
                "diffusers 后端需要安装 torch / diffusers / transformers / accelerate"
            ) from exc

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
            dtype = torch.float32
        elif self.allow_cpu:
            device = torch.device("cpu")
            dtype = torch.float32
        else:
            raise ImageGenError(
                "未检测到 GPU；设置 IMG_ALLOW_CPU=1 可使用 CPU（非常慢）"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "torch_dtype": dtype,
            "use_safetensors": True,
        }
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"

        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model, **load_kwargs
            )
        except Exception:
            # 部分模型没有 fp16 variant，回退到默认权重
            load_kwargs.pop("variant", None)
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.model, **load_kwargs
                )
            except Exception as exc:
                raise ImageGenError(f"加载 diffusers 模型失败: {exc}") from exc

        pipe = pipe.to(device)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._pipeline = pipe
        self._device = device
        self._torch = torch
        return pipe, device, torch

    def _apply_reference_injection(
        self,
        pipe: Any,
        reference_images: list[Path],
        kwargs: dict[str, Any],
        warnings: list[str],
    ) -> None:
        if self.ip_adapter_dir is None or not hasattr(pipe, "load_ip_adapter"):
            warnings.append(
                "id_injection 需要配置 IMG_IP_ADAPTER_DIR（IP-Adapter 权重目录）且 "
                "diffusers 版本支持 load_ip_adapter；本次已回退 self_attn。"
            )
            return

        try:
            if not self._ip_adapter_loaded:
                try:
                    pipe.load_ip_adapter(
                        str(self.ip_adapter_dir),
                        subfolder="sdxl_models",
                        weight_name="ip-adapter_sdxl.bin",
                    )
                except Exception:
                    pipe.load_ip_adapter(str(self.ip_adapter_dir))
                self._ip_adapter_loaded = True
            pipe.set_ip_adapter_scale(0.8)

            from PIL import Image

            pil_images = [
                Image.open(path).convert("RGB")
                for path in reference_images
                if Path(path).exists()
            ]
            if not pil_images:
                warnings.append("参考图文件不存在，已回退 self_attn。")
                return
            kwargs["ip_adapter_image"] = pil_images
        except Exception as exc:
            warnings.append(f"IP-Adapter 身份注入失败，已回退 self_attn：{exc}")
