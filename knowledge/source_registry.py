"""Source registry: tracks known sources with versioning metadata.

Supports save/load to JSON for persistence between sessions.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SourceRecord(BaseModel):
    url: str
    source_type: str  # web, arxiv, github, pdf, api
    content_hash: str = ""
    etag: str = ""
    last_modified: str = ""
    commit_sha: str = ""
    last_fetched: str = ""
    metadata: dict[str, Any] = {}


class SourceRegistry:
    """In-memory source registry (backed by DB in production)."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceRecord] = {}

    def register(self, url: str, source_type: str, **kwargs) -> SourceRecord:
        record = SourceRecord(
            url=url,
            source_type=source_type,
            last_fetched=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        self._sources[url] = record
        logger.debug("Registered source: %s", url)
        return record

    def get(self, url: str) -> SourceRecord | None:
        return self._sources.get(url)

    def update_hash(self, url: str, content: str | bytes) -> bool:
        """Update content hash. Returns True if content changed."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        new_hash = hashlib.sha256(content).hexdigest()

        record = self._sources.get(url)
        if record is None:
            return True

        changed = record.content_hash != new_hash
        record.content_hash = new_hash
        record.last_fetched = datetime.now(timezone.utc).isoformat()
        return changed

    def get_stale(self, max_age_hours: int = 24) -> list[SourceRecord]:
        """Return sources not fetched within max_age_hours."""
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_hours * 3600
        stale = []
        for record in self._sources.values():
            if record.last_fetched:
                fetched_ts = datetime.fromisoformat(record.last_fetched).timestamp()
                if fetched_ts < cutoff:
                    stale.append(record)
            else:
                stale.append(record)
        return stale

    def list_all(self) -> list[SourceRecord]:
        return list(self._sources.values())

    def count(self) -> int:
        return len(self._sources)

    def save(self, path: str | Path) -> None:
        data = [rec.model_dump() for rec in self._sources.values()]
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d source records to %s", len(data), path)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for item in data:
                rec = SourceRecord(**item)
                self._sources[rec.url] = rec
            logger.info("Loaded %d source records from %s", len(data), path)
        except Exception as e:
            logger.warning("Failed to load source registry: %s", e)
