from __future__ import annotations

from app.settings import LLMSettings, Settings, load_settings


def test_default_settings():
    s = Settings()
    assert s.app.port == 8000
    assert s.database.pool_size == 5
    assert s.redis.url == "redis://localhost:6379/0"
    assert s.llm.provider == "ollama"
    assert s.llm.model == "qwen3:8b"


def test_llm_settings_defaults():
    llm = LLMSettings()
    assert llm.provider == "ollama"
    assert "thinking" in llm.modes
    assert llm.modes["thinking"]["temperature"] == 0.2


def test_load_settings_from_config_dir():
    s = load_settings(config_dir="config")
    assert s.app.host == "0.0.0.0"
    # Provider depends on current config (may be ollama or openai)
    assert s.llm.provider in ("ollama", "openai")
    assert s.llm.model != ""


def test_load_settings_missing_dir():
    s = load_settings(config_dir="/nonexistent/path")
    # Should fall back to defaults
    assert s.app.port == 8000
    assert s.llm.provider == "ollama"
