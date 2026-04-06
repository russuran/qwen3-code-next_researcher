"""Patch ranker: ranks multiple patch candidates by quality metrics."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from repo_adaptation.patch_editor import PatchCandidate

logger = logging.getLogger(__name__)


class RankedPatch(BaseModel):
    patch: PatchCandidate
    score: float = 0.0
    rank: int = 0
    breakdown: dict[str, float] = {}


def rank_patches(
    candidates: list[PatchCandidate],
    test_results: dict[str, bool] | None = None,
    eval_metrics: dict[str, dict[str, float]] | None = None,
) -> list[RankedPatch]:
    """Rank patch candidates by quality, test results, and eval metrics.

    Args:
        candidates: Patch candidates to rank.
        test_results: Mapping of patch_id -> tests_passed.
        eval_metrics: Mapping of patch_id -> {metric: score}.

    Returns:
        Sorted list of RankedPatch (best first).
    """
    test_results = test_results or {}
    eval_metrics = eval_metrics or {}
    ranked: list[RankedPatch] = []

    for patch in candidates:
        breakdown: dict[str, float] = {}

        # Test pass score (0 or 1)
        tests_passed = test_results.get(patch.patch_id, False)
        breakdown["tests"] = 1.0 if tests_passed else 0.0

        # Diff size penalty (prefer smaller patches)
        diff_size = abs(len(patch.modified) - len(patch.original))
        breakdown["compactness"] = max(0.0, 1.0 - diff_size / 10_000)

        # External eval metrics
        metrics = eval_metrics.get(patch.patch_id, {})
        if metrics:
            breakdown["eval"] = sum(metrics.values()) / len(metrics)
        else:
            breakdown["eval"] = 0.5

        # Weighted score
        score = (
            breakdown["tests"] * 0.5
            + breakdown["compactness"] * 0.2
            + breakdown["eval"] * 0.3
        )
        ranked.append(RankedPatch(patch=patch, score=score, breakdown=breakdown))

    ranked.sort(key=lambda r: -r.score)
    for idx, item in enumerate(ranked):
        item.rank = idx + 1

    logger.info("Ranked %d patches, best score=%.3f", len(ranked), ranked[0].score if ranked else 0)
    return ranked
