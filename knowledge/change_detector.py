"""Change detector: compares current vs stored source metadata."""
from __future__ import annotations

import logging

from knowledge.source_registry import SourceRegistry

logger = logging.getLogger(__name__)


class ChangeResult:
    def __init__(self, url: str, changed: bool, reason: str = ""):
        self.url = url
        self.changed = changed
        self.reason = reason


class ChangeDetector:
    def __init__(self, registry: SourceRegistry) -> None:
        self.registry = registry

    def detect_changes(self, url: str, new_hash: str = "", new_etag: str = "") -> ChangeResult:
        record = self.registry.get(url)
        if record is None:
            return ChangeResult(url, True, "new source")

        if new_hash and record.content_hash and new_hash != record.content_hash:
            return ChangeResult(url, True, "content_hash changed")

        if new_etag and record.etag and new_etag != record.etag:
            return ChangeResult(url, True, "etag changed")

        return ChangeResult(url, False, "no change detected")

    def get_dirty_set(self, checks: list[dict[str, str]]) -> list[ChangeResult]:
        """Check multiple sources and return only changed ones."""
        dirty = []
        for check in checks:
            result = self.detect_changes(
                url=check["url"],
                new_hash=check.get("content_hash", ""),
                new_etag=check.get("etag", ""),
            )
            if result.changed:
                dirty.append(result)
        return dirty
