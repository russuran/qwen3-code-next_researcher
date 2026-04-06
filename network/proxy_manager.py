"""Proxy manager: manages proxy providers for domain-specific routing."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ProxyManager:
    """Manages proxy endpoints and maps domains to proxies."""

    def __init__(self, proxies: dict[str, str] | None = None) -> None:
        # domain_pattern -> proxy_url
        self._proxies: dict[str, str] = proxies or {}
        self._domain_overrides: dict[str, str] = {}

    def add_proxy(self, proxy_url: str, domains: list[str] | None = None) -> None:
        """Register a proxy, optionally restricted to specific domains."""
        if domains:
            for domain in domains:
                self._proxies[domain] = proxy_url
        else:
            self._proxies["*"] = proxy_url
        logger.info("Added proxy %s for %s", proxy_url, domains or ["*"])

    def remove_proxy(self, proxy_url: str) -> int:
        """Remove a proxy. Returns count of removed entries."""
        to_remove = [k for k, v in self._proxies.items() if v == proxy_url]
        for k in to_remove:
            del self._proxies[k]
        logger.info("Removed proxy %s (%d entries)", proxy_url, len(to_remove))
        return len(to_remove)

    def get_proxy(self, domain: str) -> str | None:
        """Get the best proxy URL for a given domain, or None for direct."""
        # Check domain-specific override first
        if domain in self._domain_overrides:
            return self._domain_overrides[domain]
        # Check exact match
        if domain in self._proxies:
            return self._proxies[domain]
        # Check wildcard
        return self._proxies.get("*")

    def set_override(self, domain: str, proxy_url: str) -> None:
        self._domain_overrides[domain] = proxy_url

    def list_proxies(self) -> dict[str, str]:
        return {**self._proxies, **self._domain_overrides}
