"""Hypothesis loop: auto-cycle through hypotheses with status tracking."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HypothesisStatus:
    DRAFT = "draft"
    RUNNING = "running"
    VALIDATED = "validated"
    REJECTED = "rejected"
    COMBINED = "combined"


class TrackedHypothesis(BaseModel):
    id: str
    title: str
    description: str = ""
    approach: str = ""
    expected_outcome: str = ""
    validation_method: str = ""
    priority: int = 0
    effort: str = "medium"
    based_on: list[str] = []
    status: str = HypothesisStatus.DRAFT
    research_run_id: str = ""
    validation_result: dict[str, Any] = {}
    created_at: str = ""
    updated_at: str = ""


class HypothesisRegistry(BaseModel):
    hypotheses: list[TrackedHypothesis] = []
    history: list[dict[str, Any]] = []

    def add(self, hyp: TrackedHypothesis) -> None:
        hyp.created_at = datetime.now(timezone.utc).isoformat()
        hyp.updated_at = hyp.created_at
        self.hypotheses.append(hyp)

    def get_next(self) -> TrackedHypothesis | None:
        """Get highest-priority draft hypothesis."""
        drafts = [h for h in self.hypotheses if h.status == HypothesisStatus.DRAFT]
        if not drafts:
            return None
        return sorted(drafts, key=lambda h: -h.priority)[0]

    def update_status(self, hyp_id: str, status: str, **kwargs) -> None:
        for h in self.hypotheses:
            if h.id == hyp_id:
                old = h.status
                h.status = status
                h.updated_at = datetime.now(timezone.utc).isoformat()
                for k, v in kwargs.items():
                    if hasattr(h, k):
                        setattr(h, k, v)
                self.history.append({
                    "hypothesis_id": hyp_id,
                    "from": old, "to": status,
                    "timestamp": h.updated_at,
                    **kwargs,
                })
                return

    def get_stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for h in self.hypotheses:
            counts[h.status] = counts.get(h.status, 0) + 1
        return counts

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HypothesisRegistry:
        p = Path(path)
        if p.exists():
            return cls.model_validate_json(p.read_text(encoding="utf-8"))
        return cls()


class HypothesisLoop:
    """Automated loop: pick top hypothesis -> research -> validate -> next."""

    def __init__(self, registry: HypothesisRegistry, run_callback, max_iterations: int = 5):
        self.registry = registry
        self.run_callback = run_callback  # async fn(topic: str) -> dict with eval results
        self.max_iterations = max_iterations

    async def run(self) -> HypothesisRegistry:
        iteration = 0
        while iteration < self.max_iterations:
            hyp = self.registry.get_next()
            if not hyp:
                logger.info("No more draft hypotheses. Loop complete.")
                break

            iteration += 1
            logger.info("Iteration %d: investigating H%s '%s'", iteration, hyp.id, hyp.title)

            # Mark as running
            self.registry.update_status(hyp.id, HypothesisStatus.RUNNING)

            try:
                # Build research query from hypothesis
                query = f"{hyp.title}: {hyp.approach}"
                result = await self.run_callback(query)

                # Evaluate result
                overall = result.get("overall", 0)
                if overall >= 0.6:
                    self.registry.update_status(
                        hyp.id, HypothesisStatus.VALIDATED,
                        validation_result=result,
                        research_run_id=result.get("run_id", ""),
                    )
                    logger.info("H%s validated (overall=%.2f)", hyp.id, overall)
                else:
                    self.registry.update_status(
                        hyp.id, HypothesisStatus.REJECTED,
                        validation_result=result,
                    )
                    logger.info("H%s rejected (overall=%.2f)", hyp.id, overall)

            except Exception as e:
                logger.error("H%s failed: %s", hyp.id, e)
                self.registry.update_status(
                    hyp.id, HypothesisStatus.REJECTED,
                    validation_result={"error": str(e)},
                )

        return self.registry
