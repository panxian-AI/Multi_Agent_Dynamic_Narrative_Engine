"""diffusers 本地后端：自动选择 SD1.5 / SD2.1 / SDXL 文本生图管线。

依赖为可选安装（``pip install torch diffusers transformers accelerate``），
未安装或无 GPU 时由工厂自动降级为 Mock，不影响文本管线。
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

from ..exceptions import ImageGenError, ImageTimeoutError
from ..hf_env import configure_hf_env, resolve_endpoint
from .base import ImageRequest, ImageResult
from .downloads import download_model, is_model_cached


class DiffusersImageGenerator:
    """本地 GPU 推理；模型权重懒加载到 ``MODEL_CACHE_DIR``。

    ``timeout`` 为单张图片的生成上限（秒，默认 60）。模型首次下载/加载
    不计入该上限，加载完成后的去噪过程会在步间检查超时。
    """

    backend_name = "diffusers"

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir: str | Path = "models",
        allow_cpu: bool = False,
        ip_adapter_dir: str | Path | None = None,
        timeout: int = 60,
        hf_endpoint: str | None = None,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.allow_cpu = allow_cpu
        self.hf_endpoint = hf_endpoint or ""
        self.ip_adapter_dir = (
            Path(ip_adapter_dir) if ip_adapter_dir is not None else None
        )
        self.timeout = max(1, int(timeout))
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
        cached = self._model_cached()
        endpoint = resolve_endpoint(self.hf_endpoint) or "https://huggingface.co"
        detail = (
            f"{gpu or 'cpu'} · {self.model} · 单张≤{self.timeout}s · {endpoint}"
        )
        if not cached:
            detail += (
                " · 模型未缓存（首次生成会下载；可用 "
                "`madne models download` 预下载）"
            )
        return {
            "backend": self.backend_name,
            "available": True,
            "model_cached": cached,
            "cache_dir": str(self.cache_dir),
            "hf_endpoint": endpoint,
            "detail": detail,
        }

    def _model_cached(self) -> bool:
        """只查本地缓存，不触发网络下载。"""
        return is_model_cached(self.model, self.cache_dir)

    # ------------------------------------------------------------------ 生成入口
    def generate(self, request: ImageRequest) -> ImageResult:
        pipe, device, torch_module = self._pipeline_or_raise(request.on_status)
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

        # 单张生成上限：在去噪步之间检查，超时立即中止。
        deadline = time.monotonic() + self.timeout
        total_steps = max(1, int(request.steps))

        def _report(current: int, total: int) -> None:
            if request.on_progress is None:
                return
            try:
                request.on_progress(current, total)
            except Exception:
                pass

        def _on_step_end(
            _pipe: Any,
            step_index: int,
            _timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            if time.monotonic() > deadline:
                raise ImageTimeoutError(
                    f"本地 SD 生成超时（>{self.timeout}s，"
                    f"已完成 {step_index} 步）"
                )
            _report(step_index + 1, total_steps)
            return callback_kwargs

        _report(0, total_steps)
        kwargs["callback_on_step_end"] = _on_step_end

        try:
            images = pipe(**kwargs).images
        except ImageGenError:
            raise
        except TypeError as exc:
            # 老版本 diffusers 不接受 callback_on_step_end：移除后重试。
            if "callback_on_step_end" not in str(exc):
                raise ImageGenError(f"diffusers 生成失败: {exc}") from exc
            kwargs.pop("callback_on_step_end", None)
            try:
                images = pipe(**kwargs).images
            except Exception as fallback_exc:
                raise ImageGenError(
                    f"diffusers 生成失败: {fallback_exc}"
                ) from fallback_exc
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
    def _pipeline_or_raise(
        self, on_status: Any = None
    ) -> tuple[Any, Any, Any]:
        if self._pipeline is not None:
            return self._pipeline, self._device, self._torch

        def _status(message: str) -> None:
            if on_status is None:
                return
            try:
                on_status(message)
            except Exception:
                pass

        # 必须在导入 diffusers/huggingface_hub 之前写入镜像配置。
        configure_hf_env(self.hf_endpoint or None)

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
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

        # 先把权重下完（只拉必要文件，支持断点续传与镜像回退），
        # 之后加载时只读本地缓存，避免逐个文件联网校验导致卡住。
        if not self._model_cached():
            _status(
                "本地未缓存模型，开始下载（首次约 2-3GB；"
                "国内网络会自动切换到 hf-mirror.com）…"
            )
            download = download_model(
                self.model,
                self.cache_dir,
                endpoint=self.hf_endpoint or None,
                log=_status,
            )
            _status(f"模型已就绪：{download.summary()}")

        load_kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "torch_dtype": dtype,
            "use_safetensors": True,
            "local_files_only": True,
        }
        if dtype == torch.float16:
            load_kwargs["variant"] = "fp16"

        _status(f"正在加载模型到 {device.type}（首次加载通常 10-60 秒）…")

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model, **load_kwargs
            )
        except Exception as exc:
            # 只对「本地缺文件 / 缺少 safetensors / fp16 variant」做兼容回退；
            # 网络/鉴权错误必须直接抛出，避免重复下载十几 GB。
            message = str(exc).lower()
            if load_kwargs.get("local_files_only") and any(
                token in message
                for token in (
                    "local_files_only",
                    "does not appear to have",
                    "no file",
                    "not found",
                    "cannot find",
                )
            ):
                load_kwargs.pop("local_files_only", None)
            elif "safetensors" in message:
                load_kwargs["use_safetensors"] = False
                load_kwargs.pop("variant", None)
                load_kwargs.pop("local_files_only", None)
            elif "variant" in message or "fp16" in message:
                load_kwargs.pop("variant", None)
                load_kwargs.pop("local_files_only", None)
            else:
                raise ImageGenError(
                    f"加载 diffusers 模型失败: {exc}"
                ) from exc
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.model, **load_kwargs
                )
            except Exception as fallback_exc:
                raise ImageGenError(
                    f"加载 diffusers 模型失败: {fallback_exc}"
                ) from fallback_exc

        pipe = pipe.to(device)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        # 8GB 级笔记本 GPU 上降低显存峰值，尽量避免 OOM 拖慢单张耗时。
        for method in (
            "enable_attention_slicing",
            "enable_vae_slicing",
            "enable_vae_tiling",
        ):
            try:
                getattr(pipe, method)()
            except Exception:
                pass
        if device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
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
