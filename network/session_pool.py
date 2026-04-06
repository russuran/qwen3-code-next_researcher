"""Session pool: reusable HTTP session pool with per-domain clients."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SessionPool:
    """Maintains a pool of httpx.AsyncClient instances keyed by domain."""

    def __init__(
        self,
        timeout: int = 30,
        max_connections: int = 10,
        proxy_map: dict[str, str] | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_connections = max_connections
        self._proxy_map = proxy_map or {}
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def get_session(self, domain: str) -> httpx.AsyncClient:
        """Get or create a reusable async client for a domain."""
        if domain in self._clients:
            return self._clients[domain]

        proxy_url = self._proxy_map.get(domain)
        kwargs: dict[str, Any] = {
            "timeout": self._timeout,
            "follow_redirects": True,
            "limits": httpx.Limits(
                max_connections=self._max_connections,
                max_keepalive_connections=5,
            ),
        }
        if proxy_url:
            kwargs["proxy"] = proxy_url

        client = httpx.AsyncClient(**kwargs)
        self._clients[domain] = client
        logger.debug("Created session for %s (proxy=%s)", domain, proxy_url)
        return client

    async def close(self, domain: str | None = None) -> None:
        """Close session(s). If domain is None, close all."""
        if domain:
            client = self._clients.pop(domain, None)
            if client:
                await client.aclose()
        else:
            for d, client in self._clients.items():
                await client.aclose()
            self._clients.clear()
            logger.info("Closed all sessions")

    async def close_all(self) -> None:
        await self.close()

    @property
    def active_count(self) -> int:
        return len(self._clients)
