"""Network policy: domain-level access rules and rate limiting."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class DomainPolicy:
    mode: str = "direct"  # direct, proxy, browser
    max_concurrency: int = 4
    retry_count: int = 2
    cooldown_sec: float = 1.0
    fallback_order: list[str] = ["direct"]


class RateLimiter:
    """Per-domain rate limiter using token bucket."""

    def __init__(self) -> None:
        self._last_request: dict[str, float] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def get_semaphore(self, domain: str, max_concurrency: int = 4) -> asyncio.Semaphore:
        if domain not in self._semaphores:
            self._semaphores[domain] = asyncio.Semaphore(max_concurrency)
        return self._semaphores[domain]

    async def wait(self, domain: str, cooldown: float = 1.0) -> None:
        last = self._last_request.get(domain, 0)
        elapsed = time.time() - last
        if elapsed < cooldown:
            await asyncio.sleep(cooldown - elapsed)
        self._last_request[domain] = time.time()


class NetworkLayer:
    """Unified network access with domain policies and rate limiting."""

    def __init__(self, policies: dict[str, dict[str, Any]] | None = None) -> None:
        self._policies: dict[str, DomainPolicy] = {}
        self._limiter = RateLimiter()

        if policies:
            for domain, cfg in policies.items():
                policy = DomainPolicy()
                policy.mode = cfg.get("mode", "direct")
                policy.max_concurrency = cfg.get("max_concurrency", 4)
                policy.cooldown_sec = cfg.get("cooldown_sec", 1.0)
                self._policies[domain] = policy

    def _get_policy(self, url: str) -> DomainPolicy:
        domain = urlparse(url).netloc
        return self._policies.get(domain, DomainPolicy())

    async def fetch(self, url: str, **kwargs) -> httpx.Response:
        policy = self._get_policy(url)
        domain = urlparse(url).netloc

        sem = self._limiter.get_semaphore(domain, policy.max_concurrency)
        async with sem:
            await self._limiter.wait(domain, policy.cooldown_sec)

            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                for attempt in range(policy.retry_count + 1):
                    try:
                        resp = await client.get(url, **kwargs)
                        if resp.status_code == 429:
                            wait = min(2 ** attempt * 2, 30)
                            logger.warning("Rate limited on %s, waiting %ds", domain, wait)
                            await asyncio.sleep(wait)
                            continue
                        return resp
                    except (httpx.ConnectError, httpx.TimeoutException) as e:
                        if attempt < policy.retry_count:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise

        raise httpx.ConnectError(f"All retries exhausted for {url}")
