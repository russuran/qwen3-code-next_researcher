"""Proxy health: monitors health of proxy endpoints."""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TEST_URL = "https://httpbin.org/ip"


class ProxyHealth:
    """Health checker for proxy endpoints."""

    def __init__(self, test_url: str = _TEST_URL, timeout: int = 10) -> None:
        self._test_url = test_url
        self._timeout = timeout
        self._results: dict[str, dict[str, Any]] = {}

    async def check(self, proxy_url: str) -> bool:
        """Check if a proxy endpoint is healthy."""
        start = time.time()
        try:
            async with httpx.AsyncClient(
                proxy=proxy_url, timeout=self._timeout,
            ) as client:
                resp = await client.get(self._test_url)
                latency = round(time.time() - start, 3)
                healthy = resp.status_code == 200
                self._results[proxy_url] = {
                    "healthy": healthy,
                    "latency_sec": latency,
                    "status_code": resp.status_code,
                    "checked_at": time.time(),
                }
                logger.debug(
                    "Proxy %s: healthy=%s latency=%.3fs",
                    proxy_url, healthy, latency,
                )
                return healthy
        except Exception as e:
            self._results[proxy_url] = {
                "healthy": False,
                "error": str(e),
                "checked_at": time.time(),
            }
            logger.warning("Proxy %s unhealthy: %s", proxy_url, e)
            return False

    def report(self) -> dict[str, dict[str, Any]]:
        """Return health status for all checked proxies."""
        return dict(self._results)

    def get_healthy(self) -> list[str]:
        """Return list of proxy URLs that passed the last check."""
        return [
            url for url, info in self._results.items()
            if info.get("healthy", False)
        ]
