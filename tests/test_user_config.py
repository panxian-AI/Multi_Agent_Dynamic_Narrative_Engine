from madne.config import Settings
from madne.user_config import (
    ApiConfig,
    apply_api_config,
    clear_api_config,
    config_path,
    load_api_config,
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
