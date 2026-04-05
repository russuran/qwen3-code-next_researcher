from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import engine, init_engine
from app.settings import load_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings

    # Init database
    init_engine(
        settings.database.url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
    )
    logger.info("Database engine initialized")

    # Init Redis + event bus
    app.state.redis = aioredis.from_url(settings.redis.url, decode_responses=True)
    from app.services.event_bus import init_event_bus
    init_event_bus(app.state.redis)
    logger.info("Redis + event bus connected")

    yield

    # Shutdown
    await app.state.redis.aclose()
    if engine:
        await engine.dispose()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Researcher",
        version="0.2.0",
        description="Autonomous research agent platform",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    from app.api.health import router as health_router
    from app.api.runs import router as runs_router
    from app.api.brownfield import router as brownfield_router
    from app.api.replay import router as replay_router
    from app.dashboard.routes import router as dashboard_router

    app.include_router(health_router)
    app.include_router(runs_router, prefix="/runs", tags=["runs"])
    app.include_router(brownfield_router, prefix="/brownfield", tags=["brownfield"])
    app.include_router(replay_router, prefix="/replay", tags=["replay"])
    app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])

    return app


app = create_app()
