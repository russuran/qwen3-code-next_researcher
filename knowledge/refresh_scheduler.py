"""Refresh scheduler: selective refresh of stale sources."""
from __future__ import annotations

import logging
from typing import Any

from knowledge.source_registry import SourceRegistry
from knowledge.change_detector import ChangeDetector
from knowledge.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class RefreshScheduler:
    """Manages selective refresh of stale sources and caches."""

    def __init__(
        self,
        registry: SourceRegistry,
        change_detector: ChangeDetector,
        cache: CacheManager,
    ) -> None:
        self.registry = registry
        self.detector = change_detector
        self.cache = cache

    def get_stale_sources(self, max_age_hours: int = 24) -> list[dict[str, Any]]:
        stale = self.registry.get_stale(max_age_hours)
        return [{"url": s.url, "source_type": s.source_type, "last_fetched": s.last_fetched} for s in stale]

    def invalidate_stale(self, max_age_hours: int = 24) -> int:
        """Invalidate caches for stale sources. Returns count invalidated."""
        stale = self.registry.get_stale(max_age_hours)
        count = 0
        for source in stale:
            self.cache.invalidate("search", source.url)
            self.cache.invalidate("fetch", source.url)
            count += 1
        if count:
            logger.info("Invalidated %d stale source caches", count)
        return count

    def plan_refresh(self, max_age_hours: int = 24, budget: int = 10) -> list[dict[str, Any]]:
        """Plan which sources to refresh within budget."""
        stale = self.registry.get_stale(max_age_hours)
        # Prioritize by source type: web > github > arxiv (web changes more often)
        priority = {"web": 3, "github": 2, "arxiv": 1, "semantic_scholar": 1, "papers_with_code": 1}
        stale.sort(key=lambda s: -priority.get(s.source_type, 0))
        plan = []
        for source in stale[:budget]:
            plan.append({
                "url": source.url,
                "source_type": source.source_type,
                "action": "refetch",
            })
        return plan

    def refresh_all(self, max_age_hours: int = 24) -> dict[str, int]:
        """Full refresh: invalidate stale + return stats."""
        stale_count = self.invalidate_stale(max_age_hours)
        total_sources = self.registry.count()
        return {
            "total_sources": total_sources,
            "stale_invalidated": stale_count,
            "fresh": total_sources - stale_count,
        }
