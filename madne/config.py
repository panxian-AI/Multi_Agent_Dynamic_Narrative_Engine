"""运行配置：全部通过环境变量注入，便于本地与容器部署。"""

from __future__ import annotations

import os
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
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
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


@dataclass(frozen=True, slots=True)
class Settings:
    """全局配置。frozen 便于在模块间安全共享。"""

    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    use_mock: bool = True
    data_dir: Path = Path("runtime")
    memory_compress_every: int = 10
    review_max_rewrites: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        _load_project_dotenv()
        api_key = os.getenv("LLM_API_KEY") or None
        # 未显式指定 MADNE_MOCK 时：没有 Key 自动使用 Mock，保证 Demo 可跑。
        use_mock = _env_bool("MADNE_MOCK", default=api_key is None)
        return cls(
            llm_api_key=api_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            use_mock=use_mock,
            data_dir=Path(os.getenv("MADNE_DATA_DIR", "runtime")),
            memory_compress_every=max(1, _env_int("MADNE_MEMORY_COMPRESS_EVERY", 10)),
            review_max_rewrites=max(0, _env_int("MADNE_REVIEW_MAX_REWRITES", 2)),
        )
