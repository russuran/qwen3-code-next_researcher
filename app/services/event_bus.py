"""Event bus: Redis pub/sub for real-time event streaming."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None
_CHANNEL_PREFIX = "researcher:events:"


def init_event_bus(redis_client: aioredis.Redis) -> None:
    global _redis
    _redis = redis_client


async def publish_event(run_id: str, event_data: dict) -> None:
    if _redis is None:
        return
    channel = f"{_CHANNEL_PREFIX}{run_id}"
    try:
        await _redis.publish(channel, json.dumps(event_data, default=str))
    except Exception as e:
        logger.debug("Failed to publish event: %s", e)


async def subscribe_events(run_id: str) -> AsyncGenerator[dict, None]:
    if _redis is None:
        return
    channel = f"{_CHANNEL_PREFIX}{run_id}"
    pubsub = _redis.pubsub()
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    yield json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    continue
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
