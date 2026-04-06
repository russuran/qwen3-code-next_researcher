"""Aggregator: merges results from multiple DAG nodes at barrier points."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AggregatedResult(BaseModel):
    sources: list[dict[str, Any]] = []
    total_items: int = 0
    merged_text: str = ""
    metadata: dict[str, Any] = {}


class Aggregator:
    """Aggregates results from parallel DAG node executions."""

    def aggregate(self, results: list[Any]) -> AggregatedResult:
        """Merge a list of node results into a single aggregated result."""
        merged = AggregatedResult()
        text_parts: list[str] = []

        for result in results:
            if result is None:
                continue
            if isinstance(result, dict):
                self._merge_dict(result, merged, text_parts)
            elif isinstance(result, str):
                text_parts.append(result)
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        merged.sources.append(item)
                    merged.total_items += 1
            else:
                text_parts.append(str(result))

        merged.merged_text = "\n\n---\n\n".join(text_parts)
        merged.total_items = max(merged.total_items, len(merged.sources))
        logger.info(
            "Aggregated %d results: %d sources, %d chars text",
            len(results), len(merged.sources), len(merged.merged_text),
        )
        return merged

    @staticmethod
    def _merge_dict(
        data: dict[str, Any],
        merged: AggregatedResult,
        text_parts: list[str],
    ) -> None:
        if "results" in data and isinstance(data["results"], list):
            merged.sources.extend(data["results"])
        if "text" in data:
            text_parts.append(str(data["text"]))
        if "title" in data:
            merged.sources.append(data)
        merged.total_items += 1
