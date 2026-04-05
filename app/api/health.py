from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    checks = {"db": False, "redis": False}

    # Check Redis
    try:
        redis = request.app.state.redis
        if redis:
            await redis.ping()
            checks["redis"] = True
    except Exception:
        pass

    # Check DB
    try:
        from app.db.session import async_session_factory
        if async_session_factory:
            async with async_session_factory() as session:
                await session.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                )
                checks["db"] = True
    except Exception:
        pass

    all_ready = all(checks.values())
    return {"status": "ready" if all_ready else "degraded", **checks}
