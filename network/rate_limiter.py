"""Rate limiter: per-domain rate limiting with token bucket."""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class DomainRateLimiter:
    """Per-domain rate limiter using cooldown and semaphores."""

    def __init__(
        self,
        default_cooldown: float = 1.0,
        default_concurrency: int = 4,
    ) -> None:
        self._default_cooldown = default_cooldown
        self._default_concurrency = default_concurrency
        self._last_request: dict[str, float] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._cooldowns: dict[str, float] = {}

    def configure(self, domain: str, cooldown: float, concurrency: int) -> None:
        """Set custom rate limits for a domain."""
        self._cooldowns[domain] = cooldown
        self._semaphores[domain] = asyncio.Semaphore(concurrency)
        logger.debug("Configured %s: cooldown=%.1fs concurrency=%d", domain, cooldown, concurrency)

    async def acquire(self, domain: str) -> None:
        """Wait until the domain rate limit allows a new request."""
        sem = self._get_semaphore(domain)
        await sem.acquire()

        cooldown = self._cooldowns.get(domain, self._default_cooldown)
        last = self._last_request.get(domain, 0)
        elapsed = time.time() - last
        if elapsed < cooldown:
            await asyncio.sleep(cooldown - elapsed)

        self._last_request[domain] = time.time()

    def release(self, domain: str) -> None:
        """Release the semaphore for a domain after request completes."""
        sem = self._get_semaphore(domain)
        sem.release()

    def _get_semaphore(self, domain: str) -> asyncio.Semaphore:
        if domain not in self._semaphores:
            self._semaphores[domain] = asyncio.Semaphore(self._default_concurrency)
        return self._semaphores[domain]

    @property
    def tracked_domains(self) -> list[str]:
        return list(self._last_request.keys())
