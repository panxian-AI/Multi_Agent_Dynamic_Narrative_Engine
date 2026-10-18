"""用户级 API 配置：保存在本机用户目录，便于分发测试。

Windows 默认路径：%APPDATA%\\MADNE\\config.json
其他平台回退：~/.madne/config.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import Settings

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 8192


@dataclass(slots=True)
class ApiConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    admin_unlocked: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())


def config_dir() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "MADNE"
    return Path.home() / ".madne"


def config_path() -> Path:
    return config_dir() / "config.json"


def load_api_config() -> ApiConfig:
    path = config_path()
    if not path.exists():
        return ApiConfig()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ApiConfig()

    try:
        max_tokens = int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS)
    except (TypeError, ValueError):
        max_tokens = DEFAULT_MAX_TOKENS

    raw_admin = payload.get("admin_unlocked", False)
    admin_unlocked = (
        raw_admin.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(raw_admin, str)
        else bool(raw_admin)
    )

    return ApiConfig(
        api_key=str(payload.get("api_key") or "").strip(),
        base_url=str(payload.get("base_url") or DEFAULT_BASE_URL).strip(),
        model=str(payload.get("model") or DEFAULT_MODEL).strip(),
        max_tokens=max(256, max_tokens),
        admin_unlocked=admin_unlocked,
    )


def save_api_config(config: ApiConfig) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def clear_api_config() -> None:
    path = config_path()
    if path.exists():
        path.unlink()


def apply_api_config(
    settings: Settings,
    config: ApiConfig | None = None,
) -> Settings:
    config = config or load_api_config()
    if not config.configured:
        return settings
    return replace(
        settings,
        llm_api_key=config.api_key,
        llm_base_url=config.base_url or settings.llm_base_url,
        llm_model=config.model or settings.llm_model,
        llm_max_tokens=max(256, config.max_tokens or settings.llm_max_tokens),
    )


def load_app_settings() -> Settings:
    """GUI / EXE 使用的配置：本机用户 API 配置优先于 .env。"""
    return apply_api_config(Settings.from_env())
