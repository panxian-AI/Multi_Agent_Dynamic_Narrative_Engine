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
from .downloads import (
    bundled_config_dir,
    cached_snapshot_dir,
    download_model,
    is_model_cached,
)


def _available_ram_gb() -> float | None:
    """返回系统可用物理内存（GB）；查询失败返回 None（不阻塞生成）。"""
    try:  # Windows
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            return status.ullAvailPhys / (1024**3)
    except Exception:
        pass
    try:  # Linux / macOS
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return None


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
        local_checkpoint: str | Path | None = None,
        upscale: int = 2,
        hires_steps: int = 20,
        hires_strength: float = 0.4,
        hires_guidance: float = 1.5,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.allow_cpu = allow_cpu
        self.hf_endpoint = hf_endpoint or ""
        self.local_checkpoint = (
            Path(local_checkpoint) if local_checkpoint is not None else None
        )
        self.ip_adapter_dir = (
            Path(ip_adapter_dir) if ip_adapter_dir is not None else None
        )
        self.timeout = max(1, int(timeout))
        # Hires-fix：底图按 ≤512 生成，再 img2img 放大到目标尺寸补细节。
        # 蒸馏模型（sd-turbo）直接生成 >512 的图会构图崩坏，必须走二次采样。
        self.upscale = max(0, int(upscale))
        self.hires_steps = max(1, int(hires_steps))
        self.hires_strength = min(0.99, max(0.05, float(hires_strength)))
        self.hires_guidance = max(0.0, float(hires_guidance))
        self._pipeline: Any = None
        self._img2img_pipeline: Any = None
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

        if self._has_local_checkpoint():
            assert self.local_checkpoint is not None
            return {
                "backend": self.backend_name,
                "available": True,
                "model_cached": True,
                "model_file": str(self.local_checkpoint),
                "detail": (
                    f"{gpu or 'cpu'} · 本地单文件 "
                    f"{self.local_checkpoint.name} · 单张≤{self.timeout}s"
                    + self._hires_detail()
                ),
            }

        cached = self._model_cached()
        endpoint = resolve_endpoint(self.hf_endpoint) or "https://huggingface.co"
        detail = (
            f"{gpu or 'cpu'} · {self.model} · 单张≤{self.timeout}s · {endpoint}"
            + self._hires_detail()
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
        return self._has_local_checkpoint() or is_model_cached(
            self.model, self.cache_dir
        )

    def _hires_detail(self) -> str:
        if self.upscale >= 2:
            return f" · hires {self.upscale}x（底图≤512 二次采样）"
        return ""

    def _fit_prompt_to_clip(        self, pipe: Any, prompt: str, warnings: list[str]
    ) -> str:
        """CLIP 上下文只有 77 token：超限时用真实 tokenizer 截断并告警。

        SD 类模型会把超出部分直接丢弃（最重要的收尾标签最先被截掉），
        这里显式截断至少能让"保留哪些"可控，并让用户从 warnings 得知。
        """
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            return prompt
        try:
            ids = tokenizer(
                prompt, add_special_tokens=True, truncation=False
            ).input_ids
        except Exception:
            return prompt
        max_len = int(getattr(tokenizer, "model_max_length", 0) or 77)
        if len(ids) <= max_len:
            return prompt
        try:
            kept = tokenizer.decode(
                ids[: max_len - 1], skip_special_tokens=True
            ).strip()
        except Exception:
            kept = prompt[: 4 * max_len]
        warnings.append(
            f"提示词 {len(ids)} tokens 超出 CLIP {max_len} token 上限，"
            "已截断（超出部分不参与生图）；建议压缩场景描述"
        )
        return kept or prompt

    def _has_local_checkpoint(self) -> bool:
        return (
            self.local_checkpoint is not None
            and self.local_checkpoint.is_file()
        )

    # ------------------------------------------------------------------ 生成入口
    def generate(self, request: ImageRequest) -> ImageResult:
        pipe, device, torch_module = self._pipeline_or_raise(request.on_status)
        warnings: list[str] = []

        prompt_text = self._fit_prompt_to_clip(pipe, request.prompt, warnings)

        target_w, target_h = int(request.width), int(request.height)
        base_w, base_h = self._base_size(target_w, target_h)

        # CUDA 需要 CUDA generator；MPS/CPU 使用 CPU generator。
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        kwargs: dict[str, Any] = {
            "prompt": prompt_text,
            "negative_prompt": request.negative_prompt,
            "width": base_w,
            "height": base_h,
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
        image = images[0]

        # Hires-fix：底图 ≤512，img2img 放大到目标尺寸补细节。
        if (base_w, base_h) != (target_w, target_h):
            image = self._hires_pass(
                pipe,
                request,
                image,
                target_w,
                target_h,
                generator_device,
                deadline,
                _report,
                torch_module,
                kwargs.get("ip_adapter_image"),
                warnings,
                prompt_text,
            )

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(request.output_path)

        return ImageResult(
            path=request.output_path,
            backend=self.backend_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=target_w,
            height=target_h,
            consistency_mode=request.consistency_mode,
            reference_images=list(request.reference_images),
            warnings=warnings,
        )

    # ------------------------------------------------------------------ 内部实现
    def _base_size(self, width: int, height: int) -> tuple[int, int]:
        """计算底图尺寸：目标边长超过 512 时缩到 ≤512（保持长宽比，8 的倍数）。

        蒸馏底模（sd-turbo）只在 512 附近成立，直接生成大图会重复纹理、
        构图崩坏；放大必须由 hires 二次采样完成。
        """
        max_side = max(int(width), int(height))
        if self.upscale < 2 or max_side <= 512:
            return int(width), int(height)
        scale = 512.0 / max_side
        base_w = min(int(width), max(256, round(width * scale / 8) * 8))
        base_h = min(int(height), max(256, round(height * scale / 8) * 8))
        return base_w, base_h

    def _hires_pass(
        self,
        pipe: Any,
        request: ImageRequest,
        base_image: Any,
        width: int,
        height: int,
        generator_device: str,
        deadline: float,
        _report: Any,
        torch_module: Any,
        ip_adapter_image: Any,
        warnings: list[str],
        prompt: str,
    ) -> Any:
        """img2img 二次采样：低强度重绘放大图，补出细节。"""
        try:
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            warnings.append(f"hires 二次采样不可用，已使用底图：{exc}")
            return base_image

        if self._img2img_pipeline is None:
            self._img2img_pipeline = AutoPipelineForImage2Image.from_pipe(
                pipe
            )
        img2img = self._img2img_pipeline
        try:
            img2img.set_progress_bar_config(disable=True)
        except Exception:
            pass

        # img2img 管线不接受 width/height（会被 **kwargs 静默吞掉，
        # 输出永远等于输入图尺寸）——必须先把底图放大到目标尺寸。
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            warnings.append(f"hires 二次采样不可用（缺 Pillow）：{exc}")
            return base_image
        init_image = base_image.resize(
            (int(width), int(height)), PILImage.LANCZOS
        )

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": request.negative_prompt,
            "image": init_image,
            "strength": self.hires_strength,
            "num_inference_steps": self.hires_steps,
            "guidance_scale": self.hires_guidance,
            "generator": torch_module.Generator(
                device=generator_device
            ).manual_seed(request.seed),
            "callback_on_step_end": lambda _p, step, _t, cb: (
                self._hires_step_check(deadline, step)
                or _report(step + 1, self.hires_steps)
                or cb
            ),
        }
        if ip_adapter_image is not None:
            kwargs["ip_adapter_image"] = ip_adapter_image

        try:
            out = img2img(**kwargs).images[0]
            if out.size != (int(width), int(height)):
                warnings.append(
                    f"hires 输出尺寸 {out.size} 与目标 "
                    f"{(int(width), int(height))} 不符，已缩放对齐"
                )
                out = out.resize(
                    (int(width), int(height)), PILImage.LANCZOS
                )
            return out
        except TypeError as exc:
            if "callback_on_step_end" not in str(exc):
                raise ImageGenError(f"hires 二次采样失败: {exc}") from exc
            kwargs.pop("callback_on_step_end", None)
            try:
                return img2img(**kwargs).images[0]
            except Exception as fallback_exc:
                raise ImageGenError(
                    f"hires 二次采样失败: {fallback_exc}"
                ) from fallback_exc
        except ImageTimeoutError:
            raise
        except Exception as exc:
            warnings.append(f"hires 二次采样失败，已使用底图：{exc}")
            return base_image

    @staticmethod
    def _hires_step_check(deadline: float, step_index: int) -> None:
        if time.monotonic() > deadline:
            raise ImageTimeoutError(
                f"本地 SD hires 二次采样超时（已完成 {step_index} 步）"
            )
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
            # 必须带上真实原因：打包版最常见的是缺少 requests/filelock 等
            # 发行版元数据（diffusers 用 importlib.metadata 校验版本），
            # 只报“需要安装依赖”会把排查方向带偏。
            raise ImageGenError(
                f"diffusers 后端依赖导入失败：{type(exc).__name__}: {exc}。"
                "源码运行请安装 torch / diffusers / transformers / accelerate；"
                "打包版请确认已 --copy-metadata requests / filelock / numpy。"
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

        # 加载 2.5GB 级权重需要约 4-5GB 可用内存峰值；不足时 from_single_file
        # 会抛 MemoryError（常表现为空异常消息），这里预检并给出明确指引。
        available_gb = _available_ram_gb()
        if available_gb is not None and available_gb < 3.5:
            raise ImageGenError(
                f"系统可用内存仅 {available_gb:.1f} GB，不足以加载 "
                "SD 权重（约需 4-5 GB 峰值）。请关闭占内存的程序"
                "（如大型游戏、浏览器）后重试，或改用云端图像后端。"
            )

        # 本地单文件（sd_turbo.safetensors 等）：不联网、不用 diffusers 缓存。
        if self._has_local_checkpoint():
            pipe = self._load_local_checkpoint(dtype, _status)
            return self._finalize(pipe, device, torch)

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

        return self._finalize(pipe, device, torch)

    def _load_local_checkpoint(
        self, dtype: Any, _status: Any
    ) -> Any:
        """用 ``from_single_file`` 加载本地 .safetensors / .ckpt。"""
        assert self.local_checkpoint is not None
        checkpoint = self.local_checkpoint

        try:
            from diffusers import (
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
            )
        except ImportError as exc:  # 与主路径保持一致的错误信息
            raise ImageGenError(
                "diffusers 后端需要安装 torch / diffusers / transformers / accelerate"
            ) from exc

        pipeline_cls: Any = StableDiffusionPipeline
        # 优先用 HF 缓存里的快照配置；没有缓存时退回随包内置的配置目录，
        # 这样“exe + 单个 safetensors”也能完全离线加载。
        config_dir = cached_snapshot_dir(self.model, self.cache_dir)
        if config_dir is None:
            config_dir = bundled_config_dir(self.model)
        if config_dir is not None:
            try:
                import json

                index = json.loads(
                    (config_dir / "model_index.json").read_text(
                        encoding="utf-8"
                    )
                )
                if index.get("_class_name") == "StableDiffusionXLPipeline":
                    pipeline_cls = StableDiffusionXLPipeline
            except Exception:
                pass

        _status(
            f"使用本地单文件模型：{checkpoint.name}；"
            "正在加载（首次通常 10-40 秒）…"
        )
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": checkpoint.suffix.lower() == ".safetensors",
        }
        if config_dir is not None:
            # 复用已下载的组件配置，彻底离线。
            kwargs["config"] = str(config_dir)
            kwargs["local_files_only"] = True

        try:
            return pipeline_cls.from_single_file(
                str(checkpoint), **kwargs
            )
        except Exception as exc:
            raise ImageGenError(
                f"加载本地单文件模型失败（{checkpoint}）："
                f"{type(exc).__name__}: {exc or '(无异常消息)'}。"
                "若缺少管线配置，请先执行 `madne models download`，"
                "或设置 IMG_MODEL_PATH 指向有效的权重文件。"
            ) from exc

    def _finalize(
        self, pipe: Any, device: Any, torch_module: Any
    ) -> tuple[Any, Any, Any]:
        """公共后处理：上设备、降显存、缓存管线。"""
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
                torch_module.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass

        self._pipeline = pipe
        self._device = device
        self._torch = torch_module
        return pipe, device, torch_module

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
