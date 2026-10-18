"""用户级 API 配置：保存在本机，便于分发测试。

源码 / CLI：%APPDATA%\\MADNE\\config.json（其他平台 ~/.madne/config.json）
打包 EXE：exe 同目录 config\\config.json（便携模式，不继承构建机配置）
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import Settings

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 8192

DEFAULT_IMG_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"
DEFAULT_IMG_MODEL = "hunyuan-image-lite"
DEFAULT_IMG_WIDTH = 1280
DEFAULT_IMG_HEIGHT = 720


@dataclass(slots=True)
class ApiConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    admin_unlocked: bool = False

    # 生图 API（混元生图轻量版 / OpenAI 兼容图像接口）
    img_enabled: bool = True
    img_api_key: str = ""
    img_base_url: str = DEFAULT_IMG_BASE_URL
    img_model: str = DEFAULT_IMG_MODEL
    img_width: int = DEFAULT_IMG_WIDTH
    img_height: int = DEFAULT_IMG_HEIGHT

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def img_configured(self) -> bool:
        return self.img_enabled and bool(self.img_api_key.strip())


def config_dir() -> Path:
    """打包版使用便携配置目录，避免继承构建机的用户级 Key。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "config"
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

    raw_img_enabled = payload.get("img_enabled", True)
    img_enabled = (
        raw_img_enabled.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(raw_img_enabled, str)
        else bool(raw_img_enabled)
    )

    try:
        img_width = max(256, int(payload.get("img_width") or DEFAULT_IMG_WIDTH))
    except (TypeError, ValueError):
        img_width = DEFAULT_IMG_WIDTH
    try:
        img_height = max(
            256, int(payload.get("img_height") or DEFAULT_IMG_HEIGHT)
        )
    except (TypeError, ValueError):
        img_height = DEFAULT_IMG_HEIGHT

    return ApiConfig(
        api_key=str(payload.get("api_key") or "").strip(),
        base_url=str(payload.get("base_url") or DEFAULT_BASE_URL).strip(),
        model=str(payload.get("model") or DEFAULT_MODEL).strip(),
        max_tokens=max(256, max_tokens),
        admin_unlocked=admin_unlocked,
        img_enabled=img_enabled,
        img_api_key=str(payload.get("img_api_key") or "").strip(),
        img_base_url=str(
            payload.get("img_base_url") or DEFAULT_IMG_BASE_URL
        ).strip(),
        img_model=str(payload.get("img_model") or DEFAULT_IMG_MODEL).strip(),
        img_width=img_width,
        img_height=img_height,
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
    updates: dict[str, object] = {}

    if config.configured:
        updates.update(
            llm_api_key=config.api_key,
            llm_base_url=config.base_url or settings.llm_base_url,
            llm_model=config.model or settings.llm_model,
            llm_max_tokens=max(
                256, config.max_tokens or settings.llm_max_tokens
            ),
            use_mock=False,
        )

    if config.img_enabled and config.img_api_key.strip():
        updates.update(
            img_enabled=True,
            img_backend="hunyuan",
            img_api_key=config.img_api_key.strip(),
            img_base_url=config.img_base_url.strip() or DEFAULT_IMG_BASE_URL,
            img_api_model=config.img_model.strip() or DEFAULT_IMG_MODEL,
            img_width=max(256, config.img_width or DEFAULT_IMG_WIDTH),
            img_height=max(256, config.img_height or DEFAULT_IMG_HEIGHT),
        )
    elif not config.img_enabled:
        updates["img_enabled"] = False

    if not updates:
        return settings
    return replace(settings, **updates)


def load_app_settings() -> Settings:
    """GUI / EXE 使用的配置。

    打包版只使用 exe 同目录的便携配置，不读取 .env 或系统环境里的 Key，
    从根源上避免把构建者自己的 API 自动带进分发包。
    """
    settings = Settings.from_env()
    if getattr(sys, "frozen", False):
        settings = replace(
            settings,
            llm_api_key=None,
            img_api_key=None,
            img_backend="mock",
            use_mock=True,
        )
    return apply_api_config(settings)
