"""Hugging Face 权重下载计划的纯函数测试（不联网）。"""

import json
import os

import pytest

from madne.hf_env import (
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_MIRROR,
    configure_hf_env,
    endpoint_candidates,
    resolve_endpoint,
)
from madne.image.downloads import (
    _DiskProgress,
    is_model_cached,
    select_files,
)

_HF_KEYS = (
    "HF_ENDPOINT",
    "MADNE_HF_ENDPOINT",
    "HF_HUB_DISABLE_XET",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _restore_hf_state():
    saved = {key: os.environ.get(key) for key in _HF_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.ENDPOINT = DEFAULT_HF_ENDPOINT
        if hasattr(hub_constants, "HUGGINGFACE_CO_URL_TEMPLATE"):
            hub_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
                DEFAULT_HF_ENDPOINT
                + "/{repo_id}/resolve/{revision}/{filename}"
            )
        if hasattr(hub_constants, "HF_HUB_DISABLE_XET"):
            hub_constants.HF_HUB_DISABLE_XET = False
        hub_constants.HF_HUB_DOWNLOAD_TIMEOUT = 10
        hub_constants.HF_HUB_ETAG_TIMEOUT = 10
    except Exception:
        pass


def test_select_files_prefers_fp16_and_skips_extra_formats():
    repo_files = [
        "model_index.json",
        "README.md",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder/pytorch_model.bin",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.bin",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ]

    selected = select_files(repo_files)

    assert selected is not None
    assert "model_index.json" in selected
    assert "tokenizer/vocab.json" in selected
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in selected
    assert "text_encoder/model.fp16.safetensors" in selected
    # 非 fp16 与 bin/文档都不应出现在下载计划里
    assert "unet/diffusion_pytorch_model.safetensors" not in selected
    assert "text_encoder/pytorch_model.bin" not in selected
    assert "README.md" not in selected


def test_select_files_falls_back_to_safetensors_then_bin():
    repo_files = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.bin",
        "vae/diffusion_pytorch_model.bin",
    ]

    selected = select_files(repo_files) or []

    assert "unet/diffusion_pytorch_model.safetensors" in selected
    assert "unet/diffusion_pytorch_model.bin" not in selected
    # 只有 bin 的组件必须保留，否则加载会缺权重
    assert "vae/diffusion_pytorch_model.bin" in selected


def test_select_files_ignores_root_single_file_checkpoint():
    repo_files = [
        "model_index.json",
        "sd_turbo.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
    ]

    selected = select_files(repo_files) or []

    assert "sd_turbo.safetensors" not in selected
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in selected


def test_select_files_without_model_index_keeps_full_download():
    assert select_files(["README.md", "some_model.safetensors"]) is None


def test_endpoint_defaults_to_official_then_mirror(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("MADNE_HF_ENDPOINT", raising=False)

    assert endpoint_candidates(None) == [DEFAULT_HF_ENDPOINT, DEFAULT_HF_MIRROR]
    assert endpoint_candidates(DEFAULT_HF_ENDPOINT) == [
        DEFAULT_HF_ENDPOINT,
        DEFAULT_HF_MIRROR,
    ]
    assert endpoint_candidates(DEFAULT_HF_MIRROR) == [DEFAULT_HF_MIRROR]


def test_disk_progress_counts_newest_incomplete_only(tmp_path):
    blobs = tmp_path / "models--org--demo" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "done-hash").write_bytes(b"x" * 100)
    old = blobs / "retry-hash.old.incomplete"
    old.write_bytes(b"x" * 1000)
    os.utime(old, (1, 1))
    new = blobs / "retry-hash.new.incomplete"
    new.write_bytes(b"x" * 2000)

    progress = _DiskProgress("org/demo", tmp_path, 0, None)

    # 完成的 blob + 最新一次重试的分片；旧中断尝试不重复计数
    assert progress._disk_bytes() == 100 + 2000


def test_is_model_cached_requires_component_weights(tmp_path):
    snapshot = (
        tmp_path / "models--org--demo" / "snapshots" / "abc123"
    )
    for component in ("unet", "vae", "text_encoder"):
        (snapshot / component).mkdir(parents=True)
        (snapshot / component / "config.json").write_text(
            "{}", encoding="utf-8"
        )
    (snapshot / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionPipeline",
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "text_encoder": ["transformers", "CLIPTextModel"],
            }
        ),
        encoding="utf-8",
    )

    assert not is_model_cached("org/demo", tmp_path)

    for component in ("unet", "vae", "text_encoder"):
        (snapshot / component / "weights.fp16.safetensors").write_bytes(b"x")

    assert is_model_cached("org/demo", tmp_path)


def test_configure_hf_env_disable_xet_for_mirror(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    applied = configure_hf_env(DEFAULT_HF_MIRROR)

    assert applied["HF_ENDPOINT"] == DEFAULT_HF_MIRROR
    assert applied["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_ENDPOINT"] == DEFAULT_HF_MIRROR
    assert resolve_endpoint() == DEFAULT_HF_MIRROR
