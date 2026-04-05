from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    cors_origins: list[str] = ["*"]


class DatabaseSettings(BaseSettings):
    url: str = "postgresql+asyncpg://researcher:researcher_dev@localhost:5432/researcher"
    pool_size: int = 5
    echo: bool = False


class RedisSettings(BaseSettings):
    url: str = "redis://localhost:6379/0"


class LLMSettings(BaseSettings):
    provider: str = "ollama"
    model: str = "qwen3:8b"
    host: str = "http://localhost:11434"
    api_key: str = ""
    modes: dict[str, Any] = {
        "thinking": {"temperature": 0.2, "max_tokens": 4096},
        "fast": {"temperature": 0.7, "max_tokens": 2048},
    }


class Settings(BaseSettings):
    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    llm: LLMSettings = LLMSettings()

    # Search defaults
    default_sources: list[str] = ["arxiv", "semantic_scholar", "github", "papers_with_code"]
    max_results_per_source: int = 20


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_settings(config_dir: str = "config") -> Settings:
    """Load settings from config/*.yaml files, with env var overrides."""
    cfg_path = Path(config_dir)

    app_cfg = _load_yaml(cfg_path / "app.yaml")
    llm_cfg = _load_yaml(cfg_path / "llm.yaml")

    return Settings(
        app=AppSettings(**app_cfg),
        llm=LLMSettings(**llm_cfg),
    )
