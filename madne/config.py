"""运行配置：全部通过环境变量注入，便于本地与容器部署。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .hf_env import configure_hf_env


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析：真实环境变量优先，不覆盖已有值。"""
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_project_dotenv() -> None:
    """只允许源码/CLI 读取项目 .env。

    打包后的 exe 不读取 .env：避免构建者本机的 Key 被自动带进分发包，
    桌面端只使用「API 配置」窗口保存的本机配置。
    """
    if getattr(sys, "frozen", False):
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_dotenv(candidate)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_path(name: str, default: Path | None = None) -> Path | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return Path(raw.strip())


DEFAULT_IMG_STYLE = (
    "cinematic concept art, highly detailed, dramatic volumetric lighting, "
    "rich color grading, coherent character design, wide shot"
)


@dataclass(frozen=True, slots=True)
class Settings:
    """全局配置。frozen 便于在模块间安全共享。"""

    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 8192
    use_mock: bool = True
    data_dir: Path = Path("runtime")
    db_path: Path | None = None
    memory_compress_every: int = 10
    review_max_rewrites: int = 2
    prefetch_choices: bool = True

    # ---- 动态规划链 ----
    dynamic_chain: bool = True
    max_plan_steps: int = 8
    plan_repair_limit: int = 1

    # ---- 图像生成（IMG_BACKEND: mock / diffusers / ark / seedream / comfyui）----
    img_enabled: bool = True
    img_backend: str = "diffusers"
    img_backend_explicit: bool = False
    img_timeout: int = 60
    img_width: int = 512
    img_height: int = 512
    img_steps: int = 4
    img_guidance: float = 0.0
    img_seed: int | None = None
    img_style: str = DEFAULT_IMG_STYLE
    img_consistency_mode: str = "self_attn"
    img_metrics: bool = False

    # ComfyUI
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_checkpoint: str = "sd_xl_base_1.0.safetensors"
    comfyui_workflow: Path | None = None
    comfyui_timeout: int = 60

    # Seedream / OpenAI 兼容图像接口
    # Base URL 必须由部署方填写服务商提供的 OpenAI 兼容地址，
    # 不能默认成混元地址（Seedream 模型会返回 404）。
    img_api_key: str | None = None
    img_base_url: str = ""
    img_api_model: str = "doubao-seedream-4-0-250828"
    img_api_size: str = ""
    img_api_timeout: int = 60
    img_api_retries: int = 2
    # 火山方舟 / 豆包 Seedream 专用参数
    img_api_watermark: bool = True
    img_api_sequential: str = "disabled"
    img_api_stream: bool = False

    # diffusers（本地 GPU / CPU）
    img_model: str = "stabilityai/sd-turbo"
    model_cache_dir: Path = Path("models")
    img_allow_cpu: bool = False
    img_ip_adapter_dir: Path | None = None
    # 空 = 先官方、失败自动回退 hf-mirror；指定则只用该端点。
    hf_endpoint: str = ""
    # None = 自动（镜像端点默认关闭 Xet）；True/False 强制指定。
    hf_disable_xet: bool | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        _load_project_dotenv()
        api_key = os.getenv("LLM_API_KEY") or None
        # 未显式指定 MADNE_MOCK 时：没有 Key 自动使用 Mock，保证 Demo 可跑。
        use_mock = _env_bool("MADNE_MOCK", default=api_key is None)
        img_backend_env = (os.getenv("IMG_BACKEND") or "").strip().lower()
        data_dir = (
            os.getenv("MADNE_DATA_DIR")
            or os.getenv("DATA_DIR")
            or "runtime"
        )
        settings = cls(
            llm_api_key=api_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_max_tokens=max(256, _env_int("LLM_MAX_TOKENS", 8192)),
            use_mock=use_mock,
            data_dir=Path(data_dir),
            db_path=_env_path("MADNE_DB_PATH"),
            memory_compress_every=max(1, _env_int("MADNE_MEMORY_COMPRESS_EVERY", 10)),
            review_max_rewrites=max(0, _env_int("MADNE_REVIEW_MAX_REWRITES", 2)),
            prefetch_choices=_env_bool("MADNE_PREFETCH_CHOICES", default=True),
            dynamic_chain=_env_bool("MADNE_DYNAMIC_CHAIN", default=True),
            max_plan_steps=max(3, _env_int("MADNE_MAX_PLAN_STEPS", 8)),
            plan_repair_limit=max(0, _env_int("MADNE_PLAN_REPAIR_LIMIT", 1)),
            img_enabled=_env_bool("IMG_ENABLED", default=True),
            img_backend=img_backend_env or "diffusers",
            img_backend_explicit=bool(img_backend_env),
            img_timeout=max(10, _env_int("IMG_TIMEOUT", 60)),
            img_width=max(256, _env_int("IMG_WIDTH", 512)),
            img_height=max(256, _env_int("IMG_HEIGHT", 512)),
            img_steps=max(1, _env_int("IMG_STEPS", 4)),
            img_guidance=max(0.0, _env_float("IMG_GUIDANCE", 0.0)),
            img_seed=_env_optional_int("IMG_SEED"),
            img_style=os.getenv("IMG_STYLE") or DEFAULT_IMG_STYLE,
            img_consistency_mode=(
                os.getenv("IMG_CONSISTENCY_MODE") or "self_attn"
            ).strip().lower(),
            img_api_key=(
                os.getenv("IMG_API_KEY")
                or os.getenv("ARK_API_KEY")
                or os.getenv("HUNYUAN_API_KEY")
                or None
            ),
            img_base_url=(
                os.getenv("IMG_BASE_URL")
                or os.getenv("ARK_BASE_URL")
                or os.getenv("HUNYUAN_BASE_URL")
                or ""
            ),
            img_api_model=(
                os.getenv("IMG_API_MODEL")
                or os.getenv("ARK_IMAGE_MODEL")
                or os.getenv("SEEDREAM_IMAGE_MODEL")
                or os.getenv("HUNYUAN_IMAGE_MODEL")
                or "doubao-seedream-4-0-250828"
            ),
            img_api_size=(
                os.getenv("IMG_API_SIZE")
                or os.getenv("ARK_IMAGE_SIZE")
                or ""
            ),
            img_api_timeout=max(10, _env_int("IMG_API_TIMEOUT", 60)),
            img_api_retries=max(0, _env_int("IMG_API_RETRIES", 2)),
            img_api_watermark=_env_bool("IMG_API_WATERMARK", default=True),
            img_api_sequential=(
                os.getenv("IMG_API_SEQUENTIAL") or "disabled"
            ).strip(),
            img_api_stream=_env_bool("IMG_API_STREAM", default=False),
            img_metrics=_env_bool("IMG_METRICS", default=False),
            comfyui_url=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
            comfyui_checkpoint=os.getenv(
                "COMFYUI_CHECKPOINT", "sd_xl_base_1.0.safetensors"
            ),
            comfyui_workflow=_env_path("COMFYUI_WORKFLOW"),
            comfyui_timeout=max(10, _env_int("COMFYUI_TIMEOUT", 60)),
            img_model=os.getenv(
                "IMG_MODEL", "stabilityai/sd-turbo"
            ),
            model_cache_dir=_env_path("MODEL_CACHE_DIR", Path("models"))
            or Path("models"),
            img_allow_cpu=_env_bool("IMG_ALLOW_CPU", default=False),
            img_ip_adapter_dir=_env_path("IMG_IP_ADAPTER_DIR"),
            hf_endpoint=(
                os.getenv("MADNE_HF_ENDPOINT")
                or os.getenv("HF_ENDPOINT")
                or ""
            ).strip(),
            hf_disable_xet=(
                None
                if not (os.getenv("HF_HUB_DISABLE_XET") or "").strip()
                else _env_bool("HF_HUB_DISABLE_XET", default=True)
            ),
        )
        # huggingface_hub 在 import 时缓存 HF_ENDPOINT，越早生效越好。
        configure_hf_env(settings.hf_endpoint or None, settings.hf_disable_xet)
        return settings
