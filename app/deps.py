from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.settings import Settings
from core.llm import LLM, LLMConfig, ModeConfig


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def build_llm(settings: Settings) -> LLM:
    modes_raw = settings.llm.modes
    modes = {}
    for name, cfg in modes_raw.items():
        if isinstance(cfg, dict):
            modes[name] = ModeConfig(**cfg)
        else:
            modes[name] = cfg

    llm_cfg = LLMConfig(
        provider=settings.llm.provider,
        model=settings.llm.model,
        host=settings.llm.host,
        api_key=settings.llm.api_key or None,
        modes=modes,
    )
    return LLM(llm_cfg)
