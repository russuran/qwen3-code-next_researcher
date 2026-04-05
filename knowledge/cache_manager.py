"""Cache manager: manages fetch/parse/extraction caches."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager:
    """File-system based cache (upgradable to Redis/MinIO later)."""

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict[str, Any]] = {}

    def _key(self, namespace: str, identifier: str) -> str:
        raw = f"{namespace}:{identifier}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, namespace: str, identifier: str) -> Any | None:
        key = self._key(namespace, identifier)

        # Memory cache first
        if key in self._memory:
            entry = self._memory[key]
            if entry.get("expires", float("inf")) > time.time():
                return entry["value"]
            del self._memory[key]

        # Disk cache
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("expires", float("inf")) > time.time():
                    self._memory[key] = data
                    return data["value"]
                path.unlink()
            except (json.JSONDecodeError, KeyError):
                path.unlink(missing_ok=True)

        return None

    def set(self, namespace: str, identifier: str, value: Any, ttl_seconds: int = 3600) -> None:
        key = self._key(namespace, identifier)
        entry = {
            "value": value,
            "expires": time.time() + ttl_seconds,
            "namespace": namespace,
        }
        self._memory[key] = entry

        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(entry, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write cache: %s", e)

    def invalidate(self, namespace: str, identifier: str) -> None:
        key = self._key(namespace, identifier)
        self._memory.pop(key, None)
        path = self._cache_dir / f"{key}.json"
        path.unlink(missing_ok=True)

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries in a namespace. Returns count removed."""
        count = 0
        to_remove = [k for k, v in self._memory.items() if v.get("namespace") == namespace]
        for k in to_remove:
            del self._memory[k]
            count += 1
        return count

    def clear(self) -> None:
        self._memory.clear()
        for f in self._cache_dir.glob("*.json"):
            f.unlink()
