import sys

from madne.config import Settings
from madne.user_config import (
    ApiConfig,
    DEFAULT_IMG_BASE_URL,
    DEFAULT_IMG_MODEL,
    apply_api_config,
    clear_api_config,
    config_dir,
    config_path,
    load_api_config,
    load_app_settings,
    save_api_config,
)


def test_api_config_roundtrip_and_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config = ApiConfig(
        api_key="sk-test-local",
        base_url="https://example.com/v1",
        model="test-model",
        max_tokens=1024,
    )

    save_api_config(config)

    assert config_path() == tmp_path / "MADNE" / "config.json"
    assert load_api_config() == config

    base = Settings(
        use_mock=True,
        llm_api_key=None,
        llm_base_url="https://default.example/v1",
        llm_model="default-model",
        llm_max_tokens=4096,
    )
    applied = apply_api_config(base, config)
    assert applied.llm_api_key == "sk-test-local"
    assert applied.llm_base_url == "https://example.com/v1"
    assert applied.llm_model == "test-model"
    assert applied.llm_max_tokens == 1024

    clear_api_config()
    assert not load_api_config().configured


def test_apply_api_config_without_key_keeps_base(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    base = Settings(use_mock=True, llm_api_key=None)
    assert apply_api_config(base, ApiConfig()) == base


def test_admin_unlock_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert load_api_config().admin_unlocked is False

    save_api_config(ApiConfig(api_key="sk-test-admin", admin_unlocked=True))
    assert load_api_config().admin_unlocked is True

    clear_api_config()
    assert load_api_config().admin_unlocked is False


def test_image_api_config_roundtrip_and_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    config = ApiConfig(
        api_key="sk-llm-test",
        img_api_key="sk-image-test",
        img_base_url="https://image.example/v1",
        img_model="hunyuan-image-lite",
        img_width=1280,
        img_height=720,
    )

    save_api_config(config)
    loaded = load_api_config()
    assert loaded.img_api_key == "sk-image-test"
    assert loaded.img_base_url == "https://image.example/v1"
    assert loaded.img_model == DEFAULT_IMG_MODEL
    assert (loaded.img_width, loaded.img_height) == (1280, 720)

    applied = apply_api_config(
        Settings(use_mock=True, llm_api_key=None, img_backend="mock"),
        loaded,
    )
    assert applied.llm_api_key == "sk-llm-test"
    assert applied.use_mock is False
    assert applied.img_enabled is True
    assert applied.img_backend == "hunyuan"
    assert applied.img_api_key == "sk-image-test"
    assert applied.img_base_url == "https://image.example/v1"
    assert applied.img_api_model == "hunyuan-image-lite"
    assert (applied.img_width, applied.img_height) == (1280, 720)


def test_image_api_disabled_keeps_text_config():
    config = ApiConfig(
        api_key="sk-llm-test",
        img_enabled=False,
        img_api_key="sk-image-test",
    )

    applied = apply_api_config(Settings(use_mock=True), config)

    assert applied.img_enabled is False
    assert applied.img_backend == "mock"


def test_frozen_config_is_portable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "MADNE.exe"))

    assert config_dir() == tmp_path / "config"


def test_frozen_app_settings_ignore_env_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "MADNE.exe"))
    monkeypatch.setenv("LLM_API_KEY", "sk-env-llm")
    monkeypatch.setenv("IMG_API_KEY", "sk-env-image")
    monkeypatch.setenv("MADNE_MOCK", "0")

    settings = load_app_settings()

    assert settings.llm_api_key is None
    assert settings.img_api_key is None
    assert settings.use_mock is True


def test_frozen_app_settings_reads_portable_config(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "MADNE.exe"))
    save_api_config(
        ApiConfig(
            api_key="sk-llm-portable",
            img_api_key="sk-image-portable",
            img_base_url="https://image.example/v1",
            img_model="hunyuan-image-lite",
            img_width=1920,
            img_height=1080,
        )
    )

    settings = load_app_settings()

    assert settings.llm_api_key == "sk-llm-portable"
    assert settings.use_mock is False
    assert settings.img_backend == "hunyuan"
    assert settings.img_api_key == "sk-image-portable"
    assert settings.img_base_url == "https://image.example/v1"
    assert (settings.img_width, settings.img_height) == (1920, 1080)
