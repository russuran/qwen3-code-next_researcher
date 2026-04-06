"""Patch registry: stores patch candidates with rollback support."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from repo_adaptation.patch_editor import PatchCandidate

logger = logging.getLogger(__name__)


class PatchRecord(BaseModel):
    patch: PatchCandidate
    applied: bool = False
    validated: bool = False
    metadata: dict[str, Any] = {}


class PatchRegistry:
    """Registry for managing patch candidates with rollback."""

    def __init__(self, storage_dir: str = "patches") -> None:
        self._storage = Path(storage_dir)
        self._storage.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, PatchRecord] = {}

    def register(self, patch: PatchCandidate, **metadata: Any) -> str:
        record = PatchRecord(patch=patch, metadata=metadata)
        self._records[patch.patch_id] = record
        self._persist(patch.patch_id, record)
        logger.info("Registered patch %s for %s", patch.patch_id, patch.file_path)
        return patch.patch_id

    def get(self, patch_id: str) -> PatchRecord | None:
        return self._records.get(patch_id)

    def rollback(self, patch_id: str, repo_path: str | Path) -> bool:
        """Rollback a patch by restoring the original file content."""
        record = self._records.get(patch_id)
        if not record or not record.applied:
            logger.warning("Cannot rollback %s: not applied", patch_id)
            return False

        target = Path(repo_path) / record.patch.file_path
        target.write_text(record.patch.original, encoding="utf-8")
        record.applied = False
        logger.info("Rolled back patch %s on %s", patch_id, record.patch.file_path)
        return True

    def list_patches(self, applied_only: bool = False) -> list[PatchRecord]:
        records = list(self._records.values())
        if applied_only:
            records = [r for r in records if r.applied]
        return records

    def _persist(self, patch_id: str, record: PatchRecord) -> None:
        path = self._storage / f"{patch_id}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
