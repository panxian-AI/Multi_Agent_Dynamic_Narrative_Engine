"""Hugging Face 下载环境：镜像端点与 Xet 传输开关。

``huggingface_hub`` 在 **import 时** 读取 ``HF_ENDPOINT`` / ``HF_HUB_DISABLE_XET``
并缓存为常量，所以配置必须尽早执行（``Settings.from_env`` 会调用
``configure_hf_env``）。若 hub 已经被导入，这里会同步修补它的常量，
避免「设置了镜像却仍然直连 huggingface.co」。
"""

from __future__ import annotations

import os

DEFAULT_HF_MIRROR = "https://hf-mirror.com"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
#: hub 默认读超时只有 10 秒，国内慢速链路易在 90%+ 处反复超时。
DEFAULT_DOWNLOAD_TIMEOUT = "120"
DEFAULT_ETAG_TIMEOUT = "30"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_true(raw: str | None) -> bool:
    if not raw:
        return False
    return raw.strip().lower() in _TRUE_VALUES


def is_official_endpoint(endpoint: str) -> bool:
    """是否直连 Hugging Face 官方站点。"""
    normalized = endpoint.strip().rstrip("/").lower()
    return normalized in {
        DEFAULT_HF_ENDPOINT,
        "https://huggingface.co",
        "http://huggingface.co",
    }


def resolve_endpoint(explicit: str | None = None) -> str:
    """显式参数 > ``MADNE_HF_ENDPOINT`` > ``HF_ENDPOINT`` > 空（官方默认）。"""
    for candidate in (
        explicit,
        os.getenv("MADNE_HF_ENDPOINT"),
        os.getenv("HF_ENDPOINT"),
    ):
        if candidate and candidate.strip():
            return candidate.strip().rstrip("/")
    return ""


def endpoint_candidates(explicit: str | None = None) -> list[str]:
    """下载时依次尝试的端点。

    - 用户配置了非官方端点：只用它，尊重用户选择；
    - 未配置 / 官方地址：先官方，失败后自动切到 ``hf-mirror.com``。
    """
    configured = resolve_endpoint(explicit)
    # 官方地址等同于「未配置」：保留失败后回退镜像的能力。
    if configured and not is_official_endpoint(configured):
        return [configured]
    return [DEFAULT_HF_ENDPOINT, DEFAULT_HF_MIRROR]


def configure_hf_env(
    endpoint: str | None = None,
    disable_xet: bool | None = None,
) -> dict[str, str]:
    """把镜像/Xet 配置写入环境变量，并修补已导入的 ``huggingface_hub``。

    返回实际生效的配置，便于日志与健康检查展示。
    """
    applied: dict[str, str] = {}
    resolved = resolve_endpoint(endpoint)

    if resolved:
        os.environ["HF_ENDPOINT"] = resolved
        applied["HF_ENDPOINT"] = resolved

    if disable_xet is None:
        raw = os.getenv("HF_HUB_DISABLE_XET")
        if raw is not None and raw.strip() != "":
            disable_xet = _is_true(raw)
        else:
            # 镜像的 Xet CAS 端点不完整，会返回 401；非官方端点默认关闭。
            disable_xet = bool(resolved) and not is_official_endpoint(resolved)

    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        applied["HF_HUB_DISABLE_XET"] = "1"

    # 用户显式设置时优先；否则放宽默认值，适配慢速网络。
    for key, value in (
        ("HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_DOWNLOAD_TIMEOUT),
        ("HF_HUB_ETAG_TIMEOUT", DEFAULT_ETAG_TIMEOUT),
    ):
        if not (os.getenv(key) or "").strip():
            os.environ[key] = value
            applied[key] = value

    _patch_hub_constants(resolved or None, disable_xet)
    return applied


def _patch_hub_constants(endpoint: str | None, disable_xet: bool) -> None:
    """hub 已导入时，环境变量不会再生效，需要直接改常量。"""
    try:
        import huggingface_hub.constants as hf_constants
    except Exception:  # hub 未安装：由后端自己报缺依赖
        return

    if endpoint:
        base = endpoint.rstrip("/")
    else:
        base = DEFAULT_HF_ENDPOINT
    hf_constants.ENDPOINT = base
    if hasattr(hf_constants, "HUGGINGFACE_CO_URL_TEMPLATE"):
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            f"{base}/{{repo_id}}/resolve/{{revision}}/{{filename}}"
        )

    if hasattr(hf_constants, "HF_HUB_DISABLE_XET"):
        hf_constants.HF_HUB_DISABLE_XET = bool(disable_xet)

    for name, env_name, default in (
        ("HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_DOWNLOAD_TIMEOUT),
        ("HF_HUB_ETAG_TIMEOUT", "HF_HUB_ETAG_TIMEOUT", DEFAULT_ETAG_TIMEOUT),
    ):
        if hasattr(hf_constants, name):
            try:
                setattr(
                    hf_constants,
                    name,
                    int(os.getenv(env_name) or default),
                )
            except (TypeError, ValueError):
                pass
