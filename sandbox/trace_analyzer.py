"""Trace analyzer: analyzes execution traces for performance insights."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Bottleneck(BaseModel):
    phase: str
    duration_sec: float = 0.0
    description: str = ""

class Suggestion(BaseModel):
    area: str
    suggestion: str
    priority: str = "medium"

class TraceReport(BaseModel):
    total_events: int = 0
    total_duration_sec: float = 0.0
    bottlenecks: list[Bottleneck] = []
    suggestions: list[Suggestion] = []
    phase_durations: dict[str, float] = {}


class TraceAnalyzer:
    """Analyzes execution trace events to find bottlenecks and improvements."""

    def analyze(self, events: list[dict[str, Any]]) -> TraceReport:
        """Analyze a list of trace events and produce a report."""
        report = TraceReport(total_events=len(events))
        if not events:
            return report

        phase_times: dict[str, list[float]] = {}
        for event in events:
            phase = event.get("phase", "unknown")
            duration = float(event.get("duration_sec", 0))
            phase_times.setdefault(phase, []).append(duration)
            report.total_duration_sec += duration

        # Aggregate phase durations
        for phase, durations in phase_times.items():
            total = sum(durations)
            report.phase_durations[phase] = round(total, 2)

        # Identify bottlenecks (phases taking >30% of total time)
        for phase, total in report.phase_durations.items():
            if report.total_duration_sec > 0 and total / report.total_duration_sec > 0.3:
                report.bottlenecks.append(Bottleneck(
                    phase=phase, duration_sec=total,
                    description=f"{phase} took {total / report.total_duration_sec:.0%} of total",
                ))

        # Generate suggestions
        report.suggestions = self._generate_suggestions(report)

        logger.info(
            "Trace: %d events, %.1fs total, %d bottlenecks",
            report.total_events, report.total_duration_sec, len(report.bottlenecks),
        )
        return report

    @staticmethod
    def _generate_suggestions(report: TraceReport) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for bn in report.bottlenecks:
            if "fetch" in bn.phase.lower():
                hint, prio = "Consider adding caching or parallel fetching", "high"
            elif "search" in bn.phase.lower():
                hint, prio = "Reduce max_results or add query filters", "medium"
            else:
                hint, prio = f"Optimize {bn.phase}: {bn.duration_sec:.1f}s is significant", "medium"
            suggestions.append(Suggestion(area=bn.phase, suggestion=hint, priority=prio))
        return suggestions
