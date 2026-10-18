"""运行配置：全部通过环境变量注入，便于本地与容器部署。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


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
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        # 打包后的 exe：只读取 exe 同目录的 .env，避免误读外部目录的 Key
        candidates.append(Path(sys.executable).resolve().parent / ".env")
    else:
        candidates.extend(
            [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
        )

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
    memory_compress_every: int = 10
    review_max_rewrites: int = 2
    prefetch_choices: bool = True

    # ---- 图像生成（IMG_BACKEND: mock / comfyui / diffusers）----
    img_enabled: bool = True
    img_backend: str = "mock"
    img_width: int = 1024
    img_height: int = 768
    img_steps: int = 28
    img_guidance: float = 5.0
    img_seed: int | None = None
    img_style: str = DEFAULT_IMG_STYLE
    img_consistency_mode: str = "self_attn"
    img_metrics: bool = False

    # ComfyUI
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_checkpoint: str = "sd_xl_base_1.0.safetensors"
    comfyui_workflow: Path | None = None
    comfyui_timeout: int = 180

    # diffusers（本地 GPU / CPU）
    img_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_cache_dir: Path = Path("models")
    img_allow_cpu: bool = False
    img_ip_adapter_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        _load_project_dotenv()
        api_key = os.getenv("LLM_API_KEY") or None
        # 未显式指定 MADNE_MOCK 时：没有 Key 自动使用 Mock，保证 Demo 可跑。
        use_mock = _env_bool("MADNE_MOCK", default=api_key is None)
        data_dir = (
            os.getenv("MADNE_DATA_DIR")
            or os.getenv("DATA_DIR")
            or "runtime"
        )
        return cls(
            llm_api_key=api_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_max_tokens=max(256, _env_int("LLM_MAX_TOKENS", 8192)),
            use_mock=use_mock,
            data_dir=Path(data_dir),
            memory_compress_every=max(1, _env_int("MADNE_MEMORY_COMPRESS_EVERY", 10)),
            review_max_rewrites=max(0, _env_int("MADNE_REVIEW_MAX_REWRITES", 2)),
            prefetch_choices=_env_bool("MADNE_PREFETCH_CHOICES", default=True),
            img_enabled=_env_bool("IMG_ENABLED", default=True),
            img_backend=(os.getenv("IMG_BACKEND") or "mock").strip().lower(),
            img_width=max(256, _env_int("IMG_WIDTH", 1024)),
            img_height=max(256, _env_int("IMG_HEIGHT", 768)),
            img_steps=max(1, _env_int("IMG_STEPS", 28)),
            img_guidance=max(0.0, _env_float("IMG_GUIDANCE", 5.0)),
            img_seed=_env_optional_int("IMG_SEED"),
            img_style=os.getenv("IMG_STYLE") or DEFAULT_IMG_STYLE,
            img_consistency_mode=(
                os.getenv("IMG_CONSISTENCY_MODE") or "self_attn"
            ).strip().lower(),
            img_metrics=_env_bool("IMG_METRICS", default=False),
            comfyui_url=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
            comfyui_checkpoint=os.getenv(
                "COMFYUI_CHECKPOINT", "sd_xl_base_1.0.safetensors"
            ),
            comfyui_workflow=_env_path("COMFYUI_WORKFLOW"),
            comfyui_timeout=max(10, _env_int("COMFYUI_TIMEOUT", 180)),
            img_model=os.getenv(
                "IMG_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
            ),
            model_cache_dir=_env_path("MODEL_CACHE_DIR", Path("models"))
            or Path("models"),
            img_allow_cpu=_env_bool("IMG_ALLOW_CPU", default=False),
            img_ip_adapter_dir=_env_path("IMG_IP_ADAPTER_DIR"),
        )
