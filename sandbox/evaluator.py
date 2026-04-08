"""Evaluator: quality metrics for research outputs and patches."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EvalMetrics(BaseModel):
    groundedness: float = 0.0      # 0-1: claims backed by sources
    coverage: float = 0.0          # 0-1: how many sub-questions answered
    source_diversity: float = 0.0  # 0-1: variety of source types
    factual_density: float = 0.0   # facts per 100 words
    code_presence: float = 0.0     # 0-1: code examples included
    citation_rate: float = 0.0     # 0-1: fraction of paragraphs with [N] citations
    overall: float = 0.0


class PatchMetrics(BaseModel):
    tests_pass: bool = False
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    diff_size: int = 0
    complexity_delta: float = 0.0


class Evaluator:
    """Evaluates research output quality."""

    def evaluate_report(
        self,
        report: str,
        sources: list[dict[str, Any]],
        plan_questions: list[str],
    ) -> EvalMetrics:
        metrics = EvalMetrics()

        # Source diversity
        source_types = set()
        for s in sources:
            source_types.add(s.get("source", s.get("type", "unknown")))
        metrics.source_diversity = min(len(source_types) / 4.0, 1.0)

        # Coverage: check if each question is addressed
        report_lower = report.lower()
        addressed = sum(1 for q in plan_questions if any(
            word in report_lower for word in q.lower().split()[:3]
        ))
        metrics.coverage = addressed / max(len(plan_questions), 1)

        # Code presence
        code_blocks = report.count("```")
        metrics.code_presence = min(code_blocks / 6.0, 1.0)

        # Factual density (rough: count numbers and named entities)
        words = report.split()
        facts = sum(1 for w in words if any(c.isdigit() for c in w))
        metrics.factual_density = min(facts / max(len(words), 1) * 100, 1.0)

        # Groundedness (rough: check reference links)
        links = report.count("http")
        metrics.groundedness = min(links / max(len(sources), 1), 1.0)

        # Citation rate: fraction of content paragraphs with [N] citations
        import re
        paragraphs = [p.strip() for p in report.split("\n\n") if len(p.strip()) > 50]
        cited = sum(1 for p in paragraphs if re.search(r"\[\d+\]", p))
        metrics.citation_rate = cited / max(len(paragraphs), 1)

        # Overall (weighted)
        metrics.overall = (
            metrics.groundedness * 0.20 +
            metrics.coverage * 0.20 +
            metrics.source_diversity * 0.10 +
            metrics.code_presence * 0.10 +
            metrics.factual_density * 0.15 +
            metrics.citation_rate * 0.25   # citations are critical
        )

        return metrics

    def evaluate_patch(self, diff: str, test_passed: bool) -> PatchMetrics:
        lines = diff.splitlines()
        added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        files = set()
        for l in lines:
            if l.startswith("diff --git"):
                parts = l.split()
                if len(parts) >= 4:
                    files.add(parts[3])

        return PatchMetrics(
            tests_pass=test_passed,
            files_changed=len(files),
            lines_added=added,
            lines_removed=removed,
            diff_size=len(diff),
        )
